from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vision_models.base_model import BaseModel

DEBUG_MODE = False
_ZERO_NORM_WARNED: set[str] = set()


@dataclass
class PrototypeConfig:
    indoor_weight_min: float = 0.05 # sim_indoor <= indoor_weight_t_low이면 weight를 무조건 이 값으로 떨어뜨림
    indoor_weight_t_low: float = 0.15 # sim_indoor가 이 값 이하이면 “거의 실내가 아니다”로 보고 indoor_weight_min을 적용
    indoor_weight_t_high: float = 0.45 # sim_indoor가 이 값 이상이면 weight를 1로 설정
    indoor_sim_threshold: float = 0.25 # indoor_plausibility()에서 indoor_keep_debug = (sim_indoor >= indoor_sim_threshold)를 만들기 위한 디버그/참고용 판정 임계값
    min_room_sim_threshold: float = 0.20 # 방 할당에서 top1 방 유사도가 최소 이 값 이상이어야 room을 부여 이 값보다 낮으면 unkown
    room_margin_threshold: float = 0.05 # 방 할당에서 **top1과 top2의 차이(margin)가 이 값 이상이어야 room을 부여
    margin_weight_min: float = 0.10
    margin_t_low: float = 0.08
    margin_t_high: float = 0.1
    outdoor_sim_threshold: float = 0.25
    use_dual_prototype: bool = True
    outdoor_decay_alpha: float = 0.98
    outdoor_margin_threshold: float = 0.0


class SemanticPrototypeIndex:
    def __init__(
        self,
        model: BaseModel,
        config: Optional[PrototypeConfig] = None,
        indoor_prototypes: Optional[List[str]] = None,
        room_prototypes: Optional[Dict[str, List[str]]] = None,
        outdoor_prototypes: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        auto_build: bool = True,
    ) -> None:
        self.model = model
        self.config = config or PrototypeConfig()
        self._label_cache: Dict[str, np.ndarray] = {}
        self.cache_path = Path(cache_path or ".semantic_prototypes_cache.npz")

        self.indoor_prototypes = indoor_prototypes or [
            "interior walls and ceiling",
            "indoor ceiling light",
            "indoor fluorescent lighting",
            "interior tiled floor",
            "indoor room with walls and a ceiling",
            "a kitchen with cabinets and countertop",
            "a bathroom with tiles and sink",
            "a bedroom with a bed and wardrobe",
            "an office room with a desk",
            "a living room with a sofa and TV",
            "a microwave on a kitchen counter indoors",
            "a refrigerator in a kitchen indoors",
            "a toilet inside a bathroom indoors",
            "a sofa in a living room indoors",
            "a desk and monitor in an office indoors",
        ]
        self.outdoor_prototypes = outdoor_prototypes or [
            "outdoor street scene",
            "outside in a city",
            "road and sidewalk",
            "a vehicle on the road",
            "public transportation train station",
            "airplane in the sky",
            "ocean and whale",
            "mountains and forest",
            "traffic lights and crosswalk",
            "parking lot",
            "highway",
        ]
        self.room_prototypes = room_prototypes or {
            "bedroom": ["bedroom", "bed and pillow", "sleeping room"],
            "kitchen": ["kitchen", "fridge and microwave", "cooking room"],
            "bathroom": ["bathroom", "toilet and sink", "shower room"],
            "living room": ["living room", "sofa and tv", "lounge"],
            "office": ["office", "desk and monitor", "workspace"],
            "hallway": ["hallway", "corridor", "passage"],
            "dining room": ["dining room", "table and chair", "eating area"],
        }

        self.indoor_proto: Optional[np.ndarray] = None
        self.outdoor_proto: Optional[np.ndarray] = None
        self.indoor_embs: Optional[np.ndarray] = None
        self.outdoor_embs: Optional[np.ndarray] = None
        self.room_protos: Optional[np.ndarray] = None
        self.room_names: List[str] = []
        self._room_embeddings: Dict[str, np.ndarray] = {}
        self._indoor_embedding: Optional[np.ndarray] = None

        if auto_build:
            self.build_or_load(ignore_cache=False)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        feats = self.model.get_text_features(texts)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D text embeddings, got shape {feats.shape}")
        assert feats.shape[1] >= 256, (
            f"Unexpected embedding dim {feats.shape[1]}; expected real CLIP dim (e.g., 512). "
            "Check clip_dense.py."
        )
        norms = np.linalg.norm(feats, axis=1)
        for text, norm in zip(texts, norms):
            if norm < 1e-12:
                if text not in _ZERO_NORM_WARNED:
                    print(f"[WARN] zero-norm embedding for text='{text}'. Returning zeros.")
                    _ZERO_NORM_WARNED.add(text)
                if DEBUG_MODE:
                    raise ValueError(f"Zero-norm embedding for text='{text}'")
        return self._l2_normalize_batch(feats)

    def _label_to_text(self, label: str) -> str:
        return self.make_prompt(label)

    def make_prompt(self, label: str) -> str:
        s = self._canonical_label(label)
        article = "an" if s[:1] in ["a", "e", "i", "o", "u"] else "a"
        return f"a photo of {article} {s}"

    def _canonical_label(self, label: str) -> str:
        return label.strip().lower().replace("_", " ")

    def _get_label_embedding(self, label: str) -> np.ndarray:
        canonical = self._canonical_label(label)
        if canonical in self._label_cache:
            return self._label_cache[canonical]
        text = self._label_to_text(canonical)
        emb = self._embed_texts([text])[0]
        self._label_cache[canonical] = emb
        return emb

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return vec
        return vec / norm

    @staticmethod
    def _l2_normalize_batch(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vecs / norms
    
    def build_or_load(self, ignore_cache: bool = False) -> None:
        print(f"[DBG] build_or_load(ignore_cache={ignore_cache}): using cache = {not ignore_cache}")
        loaded_outdoor = False
        if not ignore_cache and self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=True)
            self.indoor_proto = data["indoor_proto"]
            self.room_protos = data["room_protos"]
            self.room_names = data["room_names"].tolist()
            if "indoor_embs" in data:
                self.indoor_embs = data["indoor_embs"]
            if "outdoor_proto" in data:
                self.outdoor_proto = data["outdoor_proto"]
                loaded_outdoor = True
            if "outdoor_embs" in data:
                self.outdoor_embs = data["outdoor_embs"]
        else:
            self.indoor_embs = self._embed_texts(self.indoor_prototypes)
            self.indoor_proto = self._l2_normalize(self.indoor_embs.mean(axis=0))

            self.outdoor_embs = self._embed_texts(self.outdoor_prototypes)
            outdoor_proto = np.mean(self.outdoor_embs, axis=0).astype(np.float32)
            outdoor_proto = outdoor_proto / (np.linalg.norm(outdoor_proto) + 1e-12)
            self.outdoor_proto = outdoor_proto

            room_names = []
            room_vecs = []
            for room_name, texts in self.room_prototypes.items():
                emb = self._embed_texts(texts).mean(axis=0)
                room_names.append(room_name)
                room_vecs.append(self._l2_normalize(emb))
            self.room_names = room_names
            self.room_protos = np.stack(room_vecs, axis=0)
            np.savez(
                self.cache_path,
                indoor_proto=self.indoor_proto,
                outdoor_proto=self.outdoor_proto,
                indoor_embs=self.indoor_embs,
                outdoor_embs=self.outdoor_embs,
                room_protos=self.room_protos,
                room_names=np.array(self.room_names),
            )

        if not loaded_outdoor or self.indoor_embs is None or self.outdoor_embs is None:
            self.indoor_embs = self._embed_texts(self.indoor_prototypes)
            self.indoor_proto = self._l2_normalize(self.indoor_embs.mean(axis=0))
            self.outdoor_embs = self._embed_texts(self.outdoor_prototypes)
            outdoor_proto = np.mean(self.outdoor_embs, axis=0).astype(np.float32)
            outdoor_proto = outdoor_proto / (np.linalg.norm(outdoor_proto) + 1e-12)
            self.outdoor_proto = outdoor_proto
            np.savez(
                self.cache_path,
                indoor_proto=self.indoor_proto,
                outdoor_proto=self.outdoor_proto,
                indoor_embs=self.indoor_embs,
                outdoor_embs=self.outdoor_embs,
                room_protos=self.room_protos,
                room_names=np.array(self.room_names),
            )

        self._indoor_embedding = self.indoor_proto
        self._room_embeddings = dict(zip(self.room_names, self.room_protos))

        assert self.indoor_proto is not None, "indoor_proto is None; build_or_load failed."
        assert self.outdoor_proto is not None, "outdoor_proto is None; build_or_load failed."
        assert self.indoor_embs is not None, "indoor_embs is None; build_or_load failed."
        assert self.outdoor_embs is not None, "outdoor_embs is None; build_or_load failed."
        assert self.room_protos is not None, "room_protos is None; build_or_load failed."
        assert np.isfinite(self.indoor_proto).all(), "indoor_proto contains NaN/Inf"
        assert np.isfinite(self.outdoor_proto).all(), "outdoor_proto contains NaN/Inf"
        assert np.all(np.isfinite(self.indoor_embs)), "indoor_embs contains NaN/Inf"
        assert np.all(np.isfinite(self.outdoor_embs)), "outdoor_embs contains NaN/Inf"
        assert np.linalg.norm(self.indoor_proto) > 0.1, (
            "indoor_proto norm too small (likely zero vector). "
            "Check text encoder outputs and cache."
        )
        assert np.linalg.norm(self.outdoor_proto) > 0.1, (
            "outdoor_proto norm too small (likely zero vector). "
            "Check text encoder outputs and cache."
        )
        assert np.all(np.linalg.norm(self.indoor_embs, axis=1) > 0.1), (
            "indoor_embs norm too small. Check text encoder outputs and cache."
        )
        assert np.all(np.linalg.norm(self.outdoor_embs, axis=1) > 0.1), (
            "outdoor_embs norm too small. Check text encoder outputs and cache."
        )
        assert np.all(np.linalg.norm(self.room_protos, axis=1) > 0.1), (
            "room prototype norm too small. Check text encoder outputs and cache."
        )

    def debug_dump(self, sample_labels: List[str]) -> None:
        np.set_printoptions(precision=6, suppress=False)

        def _stats(name: str, x: np.ndarray) -> None:
            nan_count = np.isnan(x).sum() if np.issubdtype(x.dtype, np.floating) else "n/a"
            print(
                f"[DBG] {name}: shape={x.shape} dtype={x.dtype} "
                f"norm={np.linalg.norm(x):.6f} min={np.min(x):.6f} max={np.max(x):.6f} "
                f"nan={nan_count}"
            )

        if self.indoor_proto is None or self.room_protos is None:
            print("[DBG] Prototypes not built. Call build_or_load().")
            return

        _stats("indoor_proto", self.indoor_proto)
        _stats("outdoor_proto", self.outdoor_proto)
        _stats("room_protos", self.room_protos)
        print(f"[DBG] embedding_dim: {self.indoor_proto.shape[0]}")
        room_norms = np.linalg.norm(self.room_protos, axis=1)
        print(f"[DBG] room_protos norms: min={room_norms.min():.6f} max={room_norms.max():.6f}")

        print("[DBG] sample label embeddings:")
        for label in sample_labels:
            canonical = self._canonical_label(label)
            text = self._label_to_text(canonical)
            emb = self._get_label_embedding(label).reshape(-1)
            nan_count = np.isnan(emb).sum()
            print(
                f"[DBG] label='{label}' text='{text}' emb_norm={np.linalg.norm(emb):.6f} "
                f"nan={nan_count} first5={emb[:5]}"
            )
            v = emb.reshape(-1)
            p = self.indoor_proto.reshape(-1)
            sims_in = self.indoor_embs @ v
            sims_out = self.outdoor_embs @ v
            sim_in = float(np.max(sims_in))
            sim_out = float(np.max(sims_out))
            margin = sim_in - sim_out
            in_idx = int(np.argmax(sims_in))
            out_idx = int(np.argmax(sims_out))
            print(
                f"[DBG] sim_indoor label='{label}' max={sim_in:.6f} "
                f"proto='{self.indoor_prototypes[in_idx]}'"
            )
            print(
                f"[DBG] sim_outdoor label='{label}' max={sim_out:.6f} "
                f"proto='{self.outdoor_prototypes[out_idx]}' "
                f"margin={margin:.6f}"
            )

            sims = self.room_protos @ v
            top_idx = np.argsort(sims)[::-1][:3]
            top_rooms = [(self.room_names[i], float(sims[i])) for i in top_idx]
            print(f"[DBG] top_rooms label='{label}': {top_rooms}")

    def indoor_weight_from_similarity(self, sim_indoor: float) -> float:
        sim = float(np.clip(sim_indoor, -1.0, 1.0))
        t_low = self.config.indoor_weight_t_low
        t_high = self.config.indoor_weight_t_high
        if t_high <= t_low:
            raise ValueError("indoor_weight_t_high must be > indoor_weight_t_low")
        if sim <= t_low:
            return float(self.config.indoor_weight_min)
        if sim >= t_high:
            return 1.0
        ratio = (sim - t_low) / (t_high - t_low)
        weight = self.config.indoor_weight_min + ratio * (1.0 - self.config.indoor_weight_min)
        return float(np.clip(weight, self.config.indoor_weight_min, 1.0))

    def indoor_weight_from_margin(self, margin: float) -> float:
        margin_val = float(np.clip(margin, -1.0, 1.0))
        t_low = self.config.margin_t_low
        t_high = self.config.margin_t_high
        if t_high <= t_low:
            raise ValueError("margin_t_high must be > margin_t_low")
        if margin_val <= t_low:
            return float(self.config.margin_weight_min)
        if margin_val >= t_high:
            return 1.0
        ratio = (margin_val - t_low) / (t_high - t_low)
        weight = self.config.margin_weight_min + ratio * (1.0 - self.config.margin_weight_min)
        return float(np.clip(weight, self.config.margin_weight_min, 1.0))

    def indoor_plausibility(self, label: str) -> Dict[str, Any]:
        emb = self._get_label_embedding(label)
        if self.indoor_proto is None:
            raise ValueError("indoor_proto not initialized. Call build_or_load().")
        if self.config.use_dual_prototype:
            if self.outdoor_proto is None:
                raise ValueError("outdoor_proto not initialized. Call build_or_load().")
            if self.indoor_embs is None or self.outdoor_embs is None:
                raise ValueError("indoor/outdoor embeddings not initialized. Call build_or_load().")
            sim_in = float(np.max(self.indoor_embs @ emb.reshape(-1)))
            sim_out = float(np.max(self.outdoor_embs @ emb.reshape(-1)))
            margin = sim_in - sim_out
            weight = self.indoor_weight_from_margin(margin)
            keep_debug = margin >= self.config.margin_t_low
            return {
                "sim_indoor": sim_in,
                "sim_outdoor": sim_out,
                "sim_margin": margin,
                "indoor_weight": weight,
                "indoor_keep_debug": keep_debug,
            }

        sim = float(np.dot(emb, self.indoor_proto))
        weight = self.indoor_weight_from_similarity(sim)
        keep_debug = sim >= self.config.indoor_sim_threshold
        return {
            "sim_indoor": sim,
            "indoor_weight": weight,
            "indoor_keep_debug": keep_debug,
        }

    def outdoor_decision(self, label: str) -> Dict[str, Any]:
        emb = self._get_label_embedding(label)
        if self.config.use_dual_prototype:
            if self.indoor_embs is None or self.outdoor_embs is None:
                raise ValueError("indoor/outdoor embeddings not initialized. Call build_or_load().")
            sim_in = float(np.max(self.indoor_embs @ emb.reshape(-1)))
            sim_out = float(np.max(self.outdoor_embs @ emb.reshape(-1)))
            margin = sim_in - sim_out
            is_outdoor = margin < self.config.outdoor_margin_threshold
            return {
                "sim_indoor": sim_in,
                "sim_outdoor": sim_out,
                "sim_margin": margin,
                "is_outdoor": is_outdoor,
            }

        if self.indoor_proto is None:
            raise ValueError("indoor_proto not initialized. Call build_or_load().")
        sim_in = float(np.dot(emb, self.indoor_proto))
        margin = sim_in
        is_outdoor = sim_in < self.config.outdoor_sim_threshold
        return {
            "sim_indoor": sim_in,
            "sim_outdoor": 0.0,
            "sim_margin": margin,
            "is_outdoor": is_outdoor,
        }

    def filter_labels(self, labels: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            results[label] = self.indoor_plausibility(label)
        return results

    def assign_room(self, label: str) -> Dict[str, Any]:
        emb = self._get_label_embedding(label)
        if self.room_protos is None:
            raise ValueError("room_protos not initialized. Call build_or_load().")
        sims = self.room_protos @ emb.reshape(-1)
        scores: List[Tuple[str, float]] = list(zip(self.room_names, sims.tolist()))
        scores.sort(key=lambda x: x[1], reverse=True)
        top1_room, top1_score = scores[0]
        top2_score = scores[1][1] if len(scores) > 1 else -1.0
        margin = float(top1_score - top2_score)

        assigned = "unknown"
        if top1_score >= self.config.min_room_sim_threshold and margin >= self.config.room_margin_threshold:
            assigned = top1_room

        return {
            "room": assigned,
            "score": float(top1_score),
            "top2_score": float(top2_score),
            "margin": margin,
        }

    def assign_rooms(self, labels: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            results[label] = self.assign_room(label)
        return results


def _get_label(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        return obj.get("label")
    return getattr(obj, "label", None)


def _get_confidence(obj: Any) -> Optional[float]:
    if isinstance(obj, dict):
        return obj.get("confidence")
    return getattr(obj, "confidence", None)


def _set_attr(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def annotate_object_nodes(
    proto_index: SemanticPrototypeIndex,
    object_nodes: List[Any],
) -> None:
    labels = []
    for obj in object_nodes:
        label = _get_label(obj)
        if label:
            labels.append(label)

    unique_labels = sorted(set(labels))
    if not unique_labels:
        return

    indoor_results = {label: proto_index.outdoor_decision(label) for label in unique_labels}
    room_results = {label: proto_index.assign_room(label) for label in unique_labels}

    for obj in object_nodes:
        label = _get_label(obj)
        if not label:
            continue
        indoor = indoor_results[label]
        room = room_results[label]

        conf = _get_confidence(obj)
        is_outdoor = indoor.get("is_outdoor", False)
        confidence_weighted = None
        indoor_weight = None
        if conf is not None:
            if is_outdoor:
                confidence_weighted = float(conf) * float(proto_index.config.outdoor_decay_alpha)
            else:
                confidence_weighted = float(conf)
            indoor_weight = confidence_weighted / float(conf) if conf > 0 else 1.0

        _set_attr(obj, "sim_indoor", indoor["sim_indoor"])
        _set_attr(obj, "sim_outdoor", indoor.get("sim_outdoor"))
        _set_attr(obj, "sim_margin", indoor.get("sim_margin"))
        _set_attr(obj, "indoor_weight", indoor_weight)
        _set_attr(obj, "confidence_weighted", confidence_weighted)
        _set_attr(obj, "room_type", room["room"])
        _set_attr(obj, "room_score", room["score"])
        _set_attr(obj, "room_margin", room["margin"])
        _set_attr(obj, "room_top2_score", room["top2_score"])
        _set_attr(obj, "is_outdoor", is_outdoor)


def semantic_goal_similarity(
    proto_index: SemanticPrototypeIndex,
    goal_label: str,
    object_nodes: List[Any],
    agg: str = "max",
    use_weighted_conf: bool = True,
) -> float:
    if not object_nodes:
        return 0.0

    goal_emb = proto_index._get_label_embedding(goal_label)
    sims = []
    for obj in object_nodes:
        label = _get_label(obj)
        if not label:
            continue
        obj_emb = proto_index._get_label_embedding(label)
        sim = float(np.dot(goal_emb, obj_emb))

        if use_weighted_conf:
            conf_weighted = None
            if isinstance(obj, dict):
                conf_weighted = obj.get("confidence_weighted")
                if conf_weighted is None:
                    conf_weighted = obj.get("confidence")
            else:
                conf_weighted = getattr(obj, "confidence_weighted", None)
                if conf_weighted is None:
                    conf_weighted = getattr(obj, "confidence", None)
            if conf_weighted is not None:
                sim *= float(conf_weighted)

        sims.append(sim)

    if not sims:
        return 0.0

    if agg == "mean":
        return float(np.mean(sims))
    return float(np.max(sims))
