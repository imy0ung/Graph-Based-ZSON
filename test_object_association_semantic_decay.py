from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from mapping.pose_graph import PoseGraph, ObjectNode
from semantic_prototypes import PrototypeConfig, SemanticPrototypeIndex
from vision_models.base_model import BaseModel


class MockTextModel(BaseModel):
    feature_dim = 256

    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        feats = []
        for text in texts:
            feats.append(self._encode(text))
        return torch.tensor(feats, dtype=torch.float32)

    def compute_similarity(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode(self, text: str) -> List[float]:
        t = text.lower()
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        vec[-1] = 0.1
        indoor_keywords = ["bed", "sofa", "couch", "microwave", "toilet", "chair", "table", "indoor", "room", "home"]
        outdoor_keywords = ["airplane", "train", "car", "tree", "street", "outdoor", "highway", "ocean", "mountain"]

        if any(k in t for k in indoor_keywords):
            vec[0] = 1.0
        if any(k in t for k in outdoor_keywords):
            vec[1] = 1.0

        if "bed" in t or "bedroom" in t:
            vec[2] = 1.0
        if "microwave" in t or "kitchen" in t or "fridge" in t:
            vec[3] = 1.0
        if "toilet" in t or "bathroom" in t:
            vec[4] = 1.0
        if "sofa" in t or "living room" in t:
            vec[5] = 1.0

        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec.tolist()
        return (vec / norm).tolist()


def _make_proto() -> SemanticPrototypeIndex:
    model = MockTextModel()
    config = PrototypeConfig(
        outdoor_decay_alpha=0.9,
        outdoor_margin_threshold=0.0,
        use_dual_prototype=True,
    )
    proto = SemanticPrototypeIndex(model, config=config, auto_build=False)
    proto.build_or_load(ignore_cache=True)
    return proto


def _make_pose_graph(proto: SemanticPrototypeIndex | None) -> PoseGraph:
    pg = PoseGraph()
    if proto is not None:
        pg.set_semantic_prototypes(proto)
    pg.add_pose(0.0, 0.0, 0.0)
    return pg


def test_hungarian_matching() -> None:
    print("\n[T1] Hungarian matching")
    proto = _make_proto()
    pg = _make_pose_graph(proto)
    pose_id = pg.pose_ids[-1]

    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    obs = [
        {"label": "chair", "position_w": np.array([0.0, 0.0, 0.0]), "confidence": 0.9, "embedding": emb_a},
        {"label": "table", "position_w": np.array([5.0, 0.0, 0.0]), "confidence": 0.8, "embedding": emb_b},
    ]
    res = pg.add_object_nodes_batch(pose_id, obs, distance_threshold=2.0, mahalanobis_threshold=3.0)
    ids = [node.id for node in res if node is not None]
    print("  initial ids:", ids)

    obs_swapped = [
        {"label": "table", "position_w": np.array([5.1, 0.1, 0.0]), "confidence": 0.8, "embedding": emb_b},
        {"label": "chair", "position_w": np.array([0.1, -0.1, 0.0]), "confidence": 0.9, "embedding": emb_a},
    ]
    matches, _, _ = pg._associate_observations_hungarian(
        obs_swapped,
        distance_threshold=2.0,
        mahalanobis_threshold=3.0,
        sim_threshold=-1.0,
        w_geo=1.0,
        w_app=0.0,
        max_candidates_per_obs=10,
    )
    print("  matches:", matches)
    res2 = pg.add_object_nodes_batch(pose_id, obs_swapped, distance_threshold=2.0, mahalanobis_threshold=3.0)
    ids2 = [node.id for node in res2 if node is not None]
    print("  updated ids:", ids2)
    assert set(ids) == set(ids2), "Object IDs changed across swapped observations"


def test_label_hysteresis() -> None:
    print("\n[T2] Label hysteresis")
    pg = _make_pose_graph(None)
    pose_id = pg.pose_ids[-1]

    emb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    labels = ["tv", "monitor", "tv", "monitor", "tv", "monitor"]
    obj_id = None
    for i, label in enumerate(labels):
        obs = [{"label": label, "position_w": np.array([1.0, 1.0, 0.0]), "confidence": 0.6, "embedding": emb}]
        res = pg.add_object_nodes_batch(pose_id, obs, distance_threshold=1.0, mahalanobis_threshold=3.0)
        obj = res[0]
        assert obj is not None
        obj_id = obj.id
        print(
            f"  step={i} label={label} final={obj.label_final} "
            f"margin={obj.label_margin:.3f} consec={obj.consecutive_best_label}"
        )
    final_label = pg.nodes[obj_id].label_final
    assert final_label == labels[0], "Label hysteresis failed; label flipped too easily"


def test_semantic_decay() -> None:
    print("\n[T3] Semantic decay")
    proto = _make_proto()
    proto.config.outdoor_decay_alpha = 0.95
    pg = _make_pose_graph(proto)
    pose_id = pg.pose_ids[-1]

    obs_outdoor = [{"label": "airplane", "position_w": np.array([2.0, 2.0, 0.0]), "confidence": 1.0}]
    res = pg.add_object_nodes_batch(pose_id, obs_outdoor, distance_threshold=1.0)
    obj = res[0]
    assert obj is not None
    confs = [obj.confidence]
    for _ in range(3):
        res = pg.add_object_nodes_batch(pose_id, obs_outdoor, distance_threshold=1.0)
        obj = res[0]
        assert obj is not None
        confs.append(obj.confidence)
    print("  outdoor confidences:", confs)
    assert confs[-1] < confs[0], "Outdoor confidence did not decay"

    obs_indoor = [{"label": "bed", "position_w": np.array([3.0, 3.0, 0.0]), "confidence": 1.0}]
    res = pg.add_object_nodes_batch(pose_id, obs_indoor, distance_threshold=1.0)
    obj = res[0]
    assert obj is not None
    confs_in = [obj.confidence]
    for _ in range(3):
        res = pg.add_object_nodes_batch(pose_id, obs_indoor, distance_threshold=1.0)
        obj = res[0]
        assert obj is not None
        confs_in.append(obj.confidence)
    print("  indoor confidences:", confs_in)
    assert confs_in[-1] >= confs_in[0], "Indoor confidence should not decay"


def test_embedding_norms() -> None:
    print("\n[T4] Embedding normalization")
    proto = _make_proto()
    pg = _make_pose_graph(proto)
    pose_id = pg.pose_ids[-1]
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    obs = [{"label": "chair", "position_w": np.array([0.5, 0.5, 0.0]), "confidence": 0.7, "embedding": emb}]
    res = pg.add_object_nodes_batch(pose_id, obs, distance_threshold=1.0)
    obj = res[0]
    assert obj is not None
    assert obj.embedding is not None
    norm = np.linalg.norm(obj.embedding)
    print("  embedding norm:", norm)
    assert np.isfinite(norm)
    assert abs(norm - 1.0) < 1e-3


if __name__ == "__main__":
    test_hungarian_matching()
    test_label_hysteresis()
    test_semantic_decay()
    test_embedding_norms()
