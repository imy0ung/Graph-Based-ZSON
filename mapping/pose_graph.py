from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import time
import uuid
from collections import defaultdict

import numpy as np
import rerun as rr
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

############## utility functions ###############
""" 
normalize_embedding
CLIP 계열 모델의 텍스트/비전 출력을 L2 정규화하는 함수
Args:
    x: Optional[np.ndarray]
Returns:
    Optional[np.ndarray]

world_to_local_2d
world 좌표를 pose-local 좌표로 변환하는 함수
Args:
    pose_x: float
    pose_y: float
    pose_theta: float
    world_x: float
    world_y: float
Returns:
    np.ndarray
"""
# L2 Regularization for embedding
def normalize_embedding(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None: 
        return None
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm


def cosine_similarity_np(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Return cosine similarity between two vectors."""
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return None
    return float(np.dot(a, b) / denom)


def world_to_local_2d(pose_x: float, pose_y: float, pose_theta: float,
                      world_x: float, world_y: float) -> np.ndarray:
    # Translate to pose origin
    dx = world_x - pose_x
    dy = world_y - pose_y
    
    # Rotate by -theta (inverse rotation)
    cos_theta = np.cos(pose_theta)
    sin_theta = np.sin(pose_theta)
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta
    
    return np.array([local_x, local_y, 0.0])
############## utility functions ###############

############## node&edge 타입 정의 ###############
"""
NodeKind
- pose: 움직이는 로봇의 위치
- object: 객체
- frontier: 탐색 경로
- region: 영역

EdgeKind
- pose_pose: 움직이는 로봇의 위치와 위치 사이의 관계
- pose_object: 움직이는 로봇의 위치와 객체 사이의 관계
- pose_frontier: 움직이는 로봇의 위치와 탐색 경로 사이의 관계
- pose_region: 움직이는 로봇의 위치와 영역 사이의 관계

PoseEdgeType
"""
NodeKind = Literal["pose", "object", "frontier", "region"]
EdgeKind = Literal["pose_pose", "pose_object", "pose_frontier", "pose_region"]

# PoseEdgeType = EdgeKind # 현재는 사용하지 않음
#################################################


############## node class 정의 ###################
"""
BaseNode
- id: 노드 고유 ID
- kind: 노드 타입

PoseNode
- x: x 좌표 in world frame
- y: y 좌표 in world frame
- theta: 방향 in world frame (radians)
- step: 시간 index / 프레임 index

ObjectNode
- label: 관측 라벨
- position: (3,) world 좌표
- confidence: 신뢰도
- embedding: global CLIP embedding 등 (optional)
- num_observations: 관측 횟수
- last_seen_step: 마지막 관측 시간 index
- position_covariance: 칼만 필터 공분산 저장

FrontierNode
- position: (3,) world 좌표 (frontier 위치)
- coarse_embedding: local visual context (feature 저장 용도)
- semantic_hint: optional LLM hint (e.g. "kitchen-like area") (optional)
- is_explored: 탐색 완료 여부 

RegionNode
- name: 영역 이름
- center: (3,) region 중심 world 좌표
- embedding: region-level aggregated embedding (feature 저장 용도) (optional)
- member_object_ids: 영역 내 객체 ID 목록
- member_frontier_ids: 영역 내 Frontier ID 목록
"""

@dataclass
class BaseNode:
    id: str
    kind: NodeKind


@dataclass
class PoseNode(BaseNode):
    x: float          # x coordinate in world frame
    y: float          # y coordinate in world frame
    theta: float      # orientation in world frame (radians)
    step: int         # time index / frame index
    
    @property
    def node_id(self) -> int:
        """호환성을 위한 node_id (id에서 추출)"""
        try:
            # id가 "pose_0", "pose_1" 형식이면 숫자 추출
            return int(self.id.split('_')[-1])
        except:
            return hash(self.id) % (2**31)

@dataclass
class ObjectNode(BaseNode):
    label: str                  # detector label / text label
    position: np.ndarray        # (3,) world 좌표
    confidence: float
    embedding: Optional[np.ndarray] = None  # global CLIP embedding 등
    best_view_score: float = float("-inf")  # 가장 좋은 뷰 스코어(높을수록 좋음)
    num_observations: int = 1
    last_seen_step: int = 0
    sim_indoor: Optional[float] = None
    indoor_weight: Optional[float] = None
    confidence_weighted: Optional[float] = None
    room_type: Optional[str] = None
    room_score: Optional[float] = None
    room_margin: Optional[float] = None
    room_top2_score: Optional[float] = None
    is_outdoor: Optional[bool] = None
    label_scores: Dict[str, float] = field(default_factory=dict)
    label_final: Optional[str] = None
    label_second: Optional[str] = None
    label_margin: float = 0.0
    best_label_prev: Optional[str] = None
    consecutive_best_label: int = 0
    # Kalman filter state (for improved matching)
    position_covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.1)  # (2,2) covariance for x,y
    # CLIP 검증 점수: 탐지된 라벨이 실제로 맞는지 검증한 점수들 (누적)
    clip_scores: List[float] = field(default_factory=list)
    
    @property
    def avg_clip_score(self) -> float:
        """평균 CLIP 검증 점수 반환"""
        return float(np.mean(self.clip_scores)) if self.clip_scores else 0.0
    
    @property
    def clip_verified(self) -> bool:
        """CLIP 검증 통과 여부 (임계값 0.3 이상이면 신뢰)"""
        return self.avg_clip_score >= 0.05


@dataclass
class FrontierNode(BaseNode):
    position: np.ndarray                 # (3,) world 좌표 (frontier 위치)
    coarse_embedding: Optional[np.ndarray] = None  # local visual context 요약
    semantic_hint: Optional[str] = None            # optional LLM hint (e.g. "kitchen-like area")
    is_explored: bool = False                      # 탐색 완료 여부


@dataclass
class RegionNode(BaseNode):
    name: str                                # e.g. "kitchen", "living room"
    center: np.ndarray                       # (3,) region 중심 world 좌표
    embedding: Optional[np.ndarray] = None   # region-level aggregated embedding
    member_object_ids: List[str] = field(default_factory=list)
    member_frontier_ids: List[str] = field(default_factory=list)

#################################################

############## edge class 정의 ###################
"""
Edge
- id: 엣지 고유 ID
- kind: 엣지 타입
- src: 소스 노드 ID
- dst: 목적지 노드 ID
- rel_pos: 소스 기준 목적지 상대 위치 (3,) - 2D 평면에서 사용
"""
@dataclass
class Edge:
    id: str
    kind: EdgeKind
    src: str   # node id
    dst: str   # node id
    rel_pos: Optional[np.ndarray] = None # src 기준 dst 상대 위치 (3,) - 2D 평면에서 사용
    
    # 호환성을 위한 속성
    @property
    def source(self) -> int:
        try:
            return int(self.src.split('_')[-1])
        except:
            return hash(self.src) % (2**31)
    
    @property
    def target(self) -> int:
        try:
            return int(self.dst.split('_')[-1])
        except:
            return hash(self.dst) % (2**31)
    
    @property
    def edge_type(self) -> EdgeKind:
        return self.kind
#################################################

############## Pose graph 정의 ###################
"""
PoseGraph
Rerun 로깅 기능과 데이터베이스 저장 기능을 포함한 최소 형태의 포즈 그래프 컨테이너

1. 노드 추가 메서드
add_pose(x,y,theta): 움직이는 로봇의 위치 노드 추가
add_object_node: 객체 노드 추가
add_object_nodes_batch : 다중 객체가 탐지되었을 때, add_object_node를 여러번 호출하는 대신 한 번에 처리하는 함수.
add_frontier_node: 탐색 경로 노드 추가
add_region_node: 영역 노드 추가 (구현 예정)

2. 객체 매칭 및 업데이트
find_nearby_objects: Mahalanobis 거리를 통해서 같은 라벨을 가진 근처 객체 검색
update_object_node: Kalman 필터 기반 객체 업데이트, 객체 중심점 보정 수행, 다중 객체 confidence 평균 계산
_predict_object_position: Kalman 필터 기반 객체 위치 예측
_calculate_mahalanobis_distance: 공분산 & Mahalanobis 거리 계산

3. 노드 제거
_remove_frontier_node: 탐색 경로 노드 제거, navigator.py에서 방문시 제거
_remove_object_node: 객체 노드 제거
_remove_low_confidence_objects: 신뢰도 낮은 객체 제거 (현재 미사용)

4. 시각화
log_to_rerun: 전체 그래프를 Rerun으로 로깅
_log_edges: 엣지 로깅
_log_objects: 객체 로깅
_log_frontiers: 프런티어 로깅
_log_regions: 영역 로깅 (추후 구현 예정)

5. 유틸리티 및 통계
get_statistics: 그래프 통계 정보 반환
get_trajectory_length: 포즈 간 엣지로 총 이동 거리 계산 (미사용)
export_to_file: JSON으로 내보내기
load_from_database : DB에서 로드 (구현 예정)

6. 내부 헬퍼 메서드
_new_id: UUID 기반 고유 ID 생성
_get_grid_cell: 공간 인덱싱용 그리드 셀 계산
_get_nearby_grid_cells: 반경 내 그리드 셀 검색
_add_node(),_add_edge(): 내부 노드/엣지 추가
_angle_diff: 각도 차이 계산(정규화)
"""
class PoseGraph:
    def __init__(self,
                 min_translation: float = 0.02,
                 min_rotation: float = np.deg2rad(1.0),
                 db_path: Optional[str] = None,
                 session_name: Optional[str] = None) -> None:
        self.min_translation = min_translation
        self.min_rotation = min_rotation
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, Edge] = {}
        self._dirty = False
        
        # 편의를 위한 인덱스
        self.pose_ids: List[str] = []
        self.object_ids: List[str] = []
        self.frontier_ids: List[str] = []
        self.region_ids: List[str] = []
        
        # 공간 인덱싱을 위한 그리드 (성능 개선)
        # label -> grid_cell -> object_ids
        self._object_spatial_index: Dict[str, Dict[tuple, List[str]]] = defaultdict(lambda: defaultdict(list))
        self._grid_cell_size = 1.0  # 1m x 1m 그리드 셀
        
        # Database integration
        self.db = None
        self.session_id = None
        self.proto_index = None
        if db_path:
            from .pose_graph_db import PoseGraphDB
            self.db = PoseGraphDB(db_path)
            if session_name:
                self.session_id = self.db.create_session(
                    session_name, 
                    f"Mapping session started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
        
        self._step_counter = 0

    def clear(self) -> None:
        """
        Clear all nodes and edges from the pose graph (for episode reset).
        Database connection is preserved, but in-memory data is cleared.
        """
        self.nodes.clear()
        self.edges.clear()
        self.pose_ids.clear()
        self.object_ids.clear()
        self.frontier_ids.clear()
        self.region_ids.clear()
        self._object_spatial_index.clear()
        self._step_counter = 0
        self._dirty = False

    def _new_id(self, prefix: str) -> str: # UUID 기반 고유 ID 생성 
        """Generate new unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_grid_cell(self, x: float, y: float) -> tuple: # 공간 인덱싱 용, 1m x 1m 그리드 셀 기준 좌표 계산
        """Get grid cell coordinates for spatial indexing."""
        cell_x = int(x / self._grid_cell_size)
        cell_y = int(y / self._grid_cell_size)
        return (cell_x, cell_y)
    
    def _get_nearby_grid_cells(self, x: float, y: float, radius: float) -> List[tuple]: # x,y를 중심으로 radius 반경 내 그리드 셀 검색
        """Get nearby grid cells within radius."""
        center_cell = self._get_grid_cell(x, y)
        cells = []
        radius_cells = int(np.ceil(radius / self._grid_cell_size))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cells.append((center_cell[0] + dx, center_cell[1] + dy))
        return cells
    
    def _add_node(self, node: BaseNode): # 내부 노드 추가
        """Add node to graph."""
        self.nodes[node.id] = node
        if node.kind == "pose":
            self.pose_ids.append(node.id)
        elif node.kind == "object":
            self.object_ids.append(node.id)
            # Add to spatial index
            obj_node = node
            assert isinstance(obj_node, ObjectNode)
            cell = self._get_grid_cell(obj_node.position[0], obj_node.position[1])
            self._object_spatial_index[obj_node.label][cell].append(node.id)
        elif node.kind == "frontier":
            self.frontier_ids.append(node.id)
        elif node.kind == "region":
            self.region_ids.append(node.id)

    def _add_edge(self, edge: Edge): # 내부 엣지 추가
        """Add edge to graph."""
        self.edges[edge.id] = edge
        self._dirty = True
        
        # Save to database if available
        if self.db:
            try:
                self.db.add_edge(edge, transform_matrix=None, covariance=None, confidence=1.0)
            except Exception as e:
                # Silently fail if DB save fails (e.g., during development)
                pass

    def set_semantic_prototypes(self, proto_index) -> None:
        self.proto_index = proto_index

    def _apply_semantic_decay(self, obj: ObjectNode) -> None:
        if self.proto_index is None:
            return
        label_used = obj.label_final or obj.label
        decision = self.proto_index.outdoor_decision(label_used)
        is_outdoor = decision.get("is_outdoor")
        obj.is_outdoor = is_outdoor
        if is_outdoor:
            obj.confidence *= float(self.proto_index.config.outdoor_decay_alpha)
            obj.confidence = float(np.clip(obj.confidence, 0.0, 1.0))
            obj.confidence_weighted = obj.confidence
        else:
            obj.confidence_weighted = obj.confidence
        obj.sim_indoor = decision.get("sim_indoor")
        obj.sim_outdoor = decision.get("sim_outdoor")
        obj.sim_margin = decision.get("sim_margin")

    def _update_label_belief(
        self,
        obj: ObjectNode,
        obs_label: str,
        obs_conf: float,
        clip_score: Optional[float] = None,
        q_view: float = 1.0,
    ) -> str:
        CLIP_GAIN = 0.0
        inc = q_view * obs_conf
        if clip_score is not None:
            inc *= (1.0 + CLIP_GAIN * clip_score)

        obj.label_scores[obs_label] = obj.label_scores.get(obs_label, 0.0) + float(inc)
        sorted_scores = sorted(obj.label_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = sorted_scores[0]
        second_label = sorted_scores[1][0] if len(sorted_scores) > 1 else None
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        obj.label_second = second_label
        obj.label_margin = float(best_score - second_score)
        return best_label

    def _apply_label_hysteresis(self, obj: ObjectNode, best_label: str) -> None:
        LABEL_SWITCH_MARGIN = 0.15
        LABEL_SWITCH_CONSEC = 3

        if best_label == obj.best_label_prev:
            obj.consecutive_best_label += 1
        else:
            obj.consecutive_best_label = 1
        obj.best_label_prev = best_label

        if obj.label_final is None:
            obj.label_final = best_label
            if obj.label != best_label:
                self._update_object_label_index(obj, obj.label, best_label)
                obj.label = best_label
            return

        if (obj.label_margin >= LABEL_SWITCH_MARGIN and
                obj.consecutive_best_label >= LABEL_SWITCH_CONSEC):
            if obj.label_final != best_label:
                self._update_object_label_index(obj, obj.label, best_label)
                obj.label_final = best_label
                obj.label = best_label

    def _update_embedding_ema(
        self,
        obj: ObjectNode,
        new_embedding: Optional[np.ndarray],
        alpha: float = 0.2,
    ) -> None:
        if new_embedding is None:
            return
        new_embedding = normalize_embedding(new_embedding)
        if new_embedding is None:
            return
        if obj.embedding is None:
            obj.embedding = new_embedding
            return
        merged = (1.0 - alpha) * np.array(obj.embedding) + alpha * np.array(new_embedding)
        obj.embedding = normalize_embedding(merged)

    def _update_object_label_index(self, obj: ObjectNode, old_label: str, new_label: str) -> None:
        old_cell = self._get_grid_cell(obj.position[0], obj.position[1])
        if old_label in self._object_spatial_index:
            if old_cell in self._object_spatial_index[old_label]:
                if obj.id in self._object_spatial_index[old_label][old_cell]:
                    self._object_spatial_index[old_label][old_cell].remove(obj.id)
                if not self._object_spatial_index[old_label][old_cell]:
                    del self._object_spatial_index[old_label][old_cell]
        self._object_spatial_index[new_label][old_cell].append(obj.id)

    def _associate_observations_hungarian(
        self,
        observations: List[dict],
        candidate_obj_ids: Optional[List[str]] = None,
        distance_threshold: float = 3.0,
        mahalanobis_threshold: float = 3.0,
        sim_threshold: float = 0.0,
        w_geo: float = 1.0,
        w_app: float = 1.0,
        max_candidates_per_obs: int = 50,
    ):
        if not observations or not self.object_ids:
            return [], list(range(len(observations))), list(self.object_ids)

        def safe_norm(v):
            n = np.linalg.norm(v)
            if n < 1e-8:
                return v
            return v / n

        def get_candidates(obs_pos):
            if candidate_obj_ids is not None:
                return list(candidate_obj_ids)
            radius = distance_threshold
            cells = self._get_nearby_grid_cells(obs_pos[0], obs_pos[1], radius)
            candidates = set()
            for label, cell_map in self._object_spatial_index.items():
                for cell in cells:
                    if cell in cell_map:
                        candidates.update(cell_map[cell])
            return list(candidates)

        obs_candidates = []
        for obs in observations:
            pos = np.array(obs["position_w"][:2])
            candidates = get_candidates(pos)
            scored = []
            for obj_id in candidates:
                obj = self.nodes[obj_id]
                if not isinstance(obj, ObjectNode):
                    continue
                obj_pos = obj.position[:2]
                euclid = float(np.linalg.norm(pos - obj_pos))

                mahal = None
                if obj.position_covariance is not None:
                    try:
                        diff = pos - obj_pos
                        inv_cov = np.linalg.inv(obj.position_covariance)
                        mahal = float(np.sqrt(diff.T @ inv_cov @ diff))
                    except np.linalg.LinAlgError:
                        mahal = None

                gate_ok = False
                if mahal is not None and mahal < mahalanobis_threshold:
                    gate_ok = True
                if euclid < distance_threshold:
                    gate_ok = True
                if not gate_ok:
                    continue

                if obs.get("embedding") is not None and obj.embedding is not None:
                    obs_emb = safe_norm(np.array(obs["embedding"]))
                    obj_emb = safe_norm(np.array(obj.embedding))
                    cos_sim = float(np.dot(obs_emb, obj_emb))
                    if cos_sim < sim_threshold:
                        continue
                else:
                    cos_sim = None

                scored.append((obj_id, euclid, mahal, cos_sim))

            scored.sort(key=lambda x: x[1])
            if max_candidates_per_obs and len(scored) > max_candidates_per_obs:
                scored = scored[:max_candidates_per_obs]
            obs_candidates.append(scored)

        candidate_ids = []
        for scored in obs_candidates:
            for obj_id, _, _, _ in scored:
                if obj_id not in candidate_ids:
                    candidate_ids.append(obj_id)

        if not candidate_ids:
            return [], list(range(len(observations))), list(self.object_ids)

        cost = np.full((len(observations), len(candidate_ids)), np.inf, dtype=np.float32)
        id_to_col = {obj_id: idx for idx, obj_id in enumerate(candidate_ids)}
        eps = 1e-6
        for i, scored in enumerate(obs_candidates):
            for obj_id, euclid, mahal, cos_sim in scored:
                j = id_to_col[obj_id]
                d_geo = mahal if mahal is not None else euclid / max(distance_threshold, eps)
                if cos_sim is not None:
                    d_app = 1.0 - cos_sim
                else:
                    d_app = 1.0
                cost[i, j] = w_geo * d_geo + w_app * d_app

        matches = []
        assigned_obs = set()
        assigned_objs = set()
        if linear_sum_assignment is None:
            print("[WARN] scipy.optimize.linear_sum_assignment not available, using greedy matching.")
            for i in range(cost.shape[0]):
                j = int(np.argmin(cost[i]))
                if np.isfinite(cost[i, j]) and j not in assigned_objs:
                    matches.append((i, candidate_ids[j]))
                    assigned_obs.add(i)
                    assigned_objs.add(j)
        else:
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if np.isfinite(cost[i, j]):
                    matches.append((int(i), candidate_ids[int(j)]))
                    assigned_obs.add(int(i))
                    assigned_objs.add(int(j))

        unmatched_obs = [i for i in range(len(observations)) if i not in assigned_obs]
        unmatched_obj_ids = [obj_id for k, obj_id in enumerate(candidate_ids) if k not in assigned_objs]
        return matches, unmatched_obs, unmatched_obj_ids

    @staticmethod
    def _angle_diff(rad_a: float, rad_b: float) -> float: # 각도 차이 계산(정규화)
        diff = rad_a - rad_b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def add_pose(self, x: float, y: float, theta: float) -> int: # 움직이는 로봇의 위치 노드 추가
        """Add a pose node if it is sufficiently different from the latest one."""
        # 기존 pose 노드와 비교
        if self.pose_ids:
            last_id = self.pose_ids[-1]
            last_node = self.nodes[last_id]
            assert isinstance(last_node, PoseNode)
            
            pos_delta = np.hypot(last_node.x - x, last_node.y - y)
            rot_delta = abs(self._angle_diff(theta, last_node.theta))
            if pos_delta < self.min_translation and rot_delta < self.min_rotation:
                return last_node.node_id

        # 새 pose 노드 생성
        node_id = self._new_id("pose")
        pose_node = PoseNode(
            id=node_id,
            kind="pose",
            x=x,
            y=y,
            theta=theta,
            step=self._step_counter
        )
        self._add_node(pose_node)
        self._step_counter += 1
        
        # Save to database if available
        if self.db:
            try:
                self.db.add_node(pose_node, session_id=self.session_id, timestamp=None, metadata=None)
            except Exception as e:
                # Silently fail if DB save fails (e.g., during development)
                pass
        
        # 이전 pose와의 edge 생성
        if len(self.pose_ids) > 1:
            prev_id = self.pose_ids[-2]
            prev_node = self.nodes[prev_id]
            assert isinstance(prev_node, PoseNode)
            
            # 이전 pose 기준 현재 pose의 상대 위치 계산
            rel_pos = world_to_local_2d(prev_node.x, prev_node.y, prev_node.theta, x, y)
            
            edge_id = self._new_id("e_pp")
            edge = Edge(
                id=edge_id,
                kind="pose_pose",
                src=prev_id,
                dst=node_id,
                rel_pos=rel_pos
            )
            self._add_edge(edge)
        
        self._dirty = True
        return pose_node.node_id

    def _predict_object_position(self, obj_node: ObjectNode, steps_since_last_seen: int) -> np.ndarray:
        """
        객체 위치 예측 상태 추정값 : 정적 객체이므로, 위치만 반환
        """
        # Constant position model (static objects)
        predicted_pos = obj_node.position[:2]
        return predicted_pos
    
    def _calculate_mahalanobis_distance(
        self,
        observed_pos: np.ndarray,
        predicted_pos: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        관측된 값이랑 예측된 값 사이의 mahalanobis 거리 계산

        Args:
            observed_pos: (2,) observed position
            predicted_pos: (2,) predicted position
            covariance: (2,2) position covariance matrix
        
        Returns:
            Mahalanobis distance
        """
        diff = observed_pos - predicted_pos
        try:
            inv_cov = np.linalg.inv(covariance + np.eye(2) * 1e-6)  # Add small value for numerical stability
            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
            return mahal_dist
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance is singular
            return np.linalg.norm(diff)
    
    def find_nearby_objects(
        self,
        label: str,
        position_w: np.ndarray,
        distance_threshold: float = 3.0,
        use_kalman: bool = True,
        mahalanobis_threshold: float = 3.0,  # 3-sigma threshold
    ) -> Optional[ObjectNode]:
        """
        칼만 필터 매칭을 사용해서 객체 근처의 같은 라벨을 가지는 객체가 있는 조사
        mahalanobis threshold를 설정해서, 근처에 같은 라벨을 가지는 객체 있는지 조사.
        Args:
            label: Object label to match
            position_w: World position (3,)
            distance_threshold: Maximum Euclidean distance threshold (meters)
            use_kalman: Whether to use Kalman filter prediction and Mahalanobis distance
            mahalanobis_threshold: Maximum Mahalanobis distance threshold (sigma)
        
        Returns:
            Nearest matching ObjectNode if found, None otherwise
        """
        best_match = None
        best_score = float('inf')
        observed_pos_2d = position_w[:2]
        
        # Use spatial indexing to only check nearby grid cells
        nearby_cells = self._get_nearby_grid_cells(position_w[0], position_w[1], distance_threshold)
        candidate_ids = set()
        
        for cell in nearby_cells:
            if label in self._object_spatial_index and cell in self._object_spatial_index[label]:
                candidate_ids.update(self._object_spatial_index[label][cell])
        
        # Check candidates with Kalman filter-based matching
        for obj_id in candidate_ids:
            if obj_id not in self.nodes:
                continue
            obj_node = self.nodes[obj_id]
            assert isinstance(obj_node, ObjectNode)
            
            if use_kalman and obj_node.num_observations > 1:
                # Use Kalman filter prediction
                steps_since_last_seen = self._step_counter - obj_node.last_seen_step
                predicted_pos = self._predict_object_position(obj_node, steps_since_last_seen)
                
                # Calculate Mahalanobis distance
                mahal_dist = self._calculate_mahalanobis_distance(
                    observed_pos_2d,
                    predicted_pos,
                    obj_node.position_covariance
                )
                
                # Also check Euclidean distance as fallback
                euclidean_dist = np.linalg.norm(observed_pos_2d - obj_node.position[:2])
                
                # Match if within Mahalanobis threshold OR within Euclidean threshold
                if mahal_dist < mahalanobis_threshold or euclidean_dist < distance_threshold:
                    # Use Mahalanobis distance as score (lower is better)
                    score = mahal_dist
                    if score < best_score:
                        best_score = score
                        best_match = obj_node
            else:
                # Fallback to simple Euclidean distance for new objects
                euclidean_dist = np.linalg.norm(observed_pos_2d - obj_node.position[:2])
                if euclidean_dist < distance_threshold:
                    score = euclidean_dist
                    if score < best_score:
                        best_score = score
                        best_match = obj_node
        
        return best_match
    
    def find_nearby_objects_by_embedding(
        self,
        position_w: np.ndarray,
        embedding: Optional[np.ndarray],
        distance_threshold: float = 3.0,
        use_kalman: bool = True,
        mahalanobis_threshold: float = 3.0,
        sim_threshold: float = 0.3,
        max_candidates: int = 50,
    ) -> Tuple[Optional[ObjectNode], Optional[float]]:
        """
        라벨 없이 임베딩+거리 기반으로 근처 객체를 탐색.

        Returns:
            (best_match, best_sim) 혹은 (None, None)
        """
        if embedding is None:
            return None, None

        observed_pos_2d = position_w[:2]
        nearby_cells = self._get_nearby_grid_cells(position_w[0], position_w[1], distance_threshold)

        candidate_ids: set = set()
        for cell in nearby_cells:
            for label_cells in self._object_spatial_index.values():
                if cell in label_cells:
                    candidate_ids.update(label_cells[cell])
            if len(candidate_ids) >= max_candidates:
                break

        best_match: Optional[ObjectNode] = None
        best_sim: float = -1.0

        for obj_id in list(candidate_ids)[:max_candidates]:
            if obj_id not in self.nodes:
                continue
            obj_node = self.nodes[obj_id]
            if not isinstance(obj_node, ObjectNode):
                continue
            if obj_node.embedding is None:
                continue

            euclidean_dist = np.linalg.norm(observed_pos_2d - obj_node.position[:2])

            # Kalman/Mahalanobis gating if available
            if use_kalman and obj_node.num_observations > 1:
                steps_since_last_seen = self._step_counter - obj_node.last_seen_step
                predicted_pos = self._predict_object_position(obj_node, steps_since_last_seen)
                mahal_dist = self._calculate_mahalanobis_distance(
                    observed_pos_2d,
                    predicted_pos,
                    obj_node.position_covariance
                )
                if not (mahal_dist < mahalanobis_threshold or euclidean_dist < distance_threshold):
                    continue
            else:
                if euclidean_dist >= distance_threshold:
                    continue

            sim = cosine_similarity_np(embedding, obj_node.embedding)
            if sim is None:
                continue
            if sim >= sim_threshold and sim > best_sim:
                best_sim = sim
                best_match = obj_node

        if best_sim < sim_threshold:
            return None, None

        return best_match, best_sim
    
    def update_object_node(
        self,
        obj_node: ObjectNode,
        new_position_w: np.ndarray,
        new_confidence: float,
        new_embedding: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        observation_noise: float = 0.1,  # Observation noise covariance
    ) -> ObjectNode:
        """
        칼만 필터 업데이트 단계 : 예측된 위치와 관측된 위치 사이의 차이를 줄이기 위해 칼만 필터 업데이트 단계를 수행하는 함수.
        
        Args:
            obj_node: Existing object node to update
            new_position_w: New observed position (3,)
            new_confidence: New observation confidence
            new_embedding: New embedding (optional)
            step: Current step number
            observation_noise: Observation noise standard deviation (meters)
        
        Returns:
            Updated ObjectNode
        """
        old_pos_2d = np.array(obj_node.position[:2])
        new_pos_2d = np.array(new_position_w[:2])
        old_cell = self._get_grid_cell(old_pos_2d[0], old_pos_2d[1])
        
        # Kalman filter update step
        steps_since_last_seen = (step or self._step_counter) - obj_node.last_seen_step
        
        # Predict step (constant position model for static objects)
        predicted_pos = old_pos_2d  # Position remains constant
        # Predict covariance (process noise increases uncertainty over time)
        process_noise = np.eye(2) * 0.01  # Small process noise
        predicted_cov = obj_node.position_covariance + process_noise
        
        # Update step (Kalman filter)
        observation_cov = np.eye(2) * (observation_noise ** 2)
        innovation = new_pos_2d - predicted_pos
        innovation_cov = predicted_cov + observation_cov
        
        try:
            kalman_gain = predicted_cov @ np.linalg.inv(innovation_cov)
            updated_pos_2d = predicted_pos + kalman_gain @ innovation
            updated_cov = (np.eye(2) - kalman_gain) @ predicted_cov
        except np.linalg.LinAlgError:
            # Fallback to weighted average if Kalman update fails
            old_weight = obj_node.num_observations
            new_weight = 1.0
            total_weight = old_weight + new_weight
            updated_pos_2d = (old_pos_2d * old_weight + new_pos_2d * new_weight) / total_weight
            updated_cov = obj_node.position_covariance  # Keep old covariance
        
        # Update 3D position (z remains from observation)
        obj_node.position = np.array([updated_pos_2d[0], updated_pos_2d[1], new_position_w[2]])
        obj_node.position_covariance = updated_cov
        
        # Update spatial index if position changed significantly
        new_cell = self._get_grid_cell(obj_node.position[0], obj_node.position[1])
        if old_cell != new_cell:
            # Remove from old cell
            if obj_node.label in self._object_spatial_index:
                if old_cell in self._object_spatial_index[obj_node.label]:
                    if obj_node.id in self._object_spatial_index[obj_node.label][old_cell]:
                        self._object_spatial_index[obj_node.label][old_cell].remove(obj_node.id)
                    if not self._object_spatial_index[obj_node.label][old_cell]:
                        del self._object_spatial_index[obj_node.label][old_cell]
            # Add to new cell
            self._object_spatial_index[obj_node.label][new_cell].append(obj_node.id)
        
        # Update confidence using weighted average
        old_weight = obj_node.num_observations
        new_weight = 1.0
        total_weight = old_weight + new_weight
        obj_node.confidence = (
            obj_node.confidence * old_weight + new_confidence * new_weight
        ) / total_weight
        
        # Update embedding if provided (EMA)
        if new_embedding is not None:
            self._update_embedding_ema(obj_node, new_embedding, alpha=0.2)
        
        # Update observation count and last seen step
        obj_node.num_observations += 1
        obj_node.last_seen_step = step or self._step_counter
        
        return obj_node
    
    def add_object_node(
        self,
        pose_id: str,
        label: str,
        position_w: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        distance_threshold: float = 3.0,
        clip_score: Optional[float] = None,  # CLIP 검증 점수 (탐지 라벨이 맞는지 검증)
    ) -> ObjectNode:
        """
        객체 노드를 추가하거나, 근처에 발견된 객체의 중심점을 이용해 칼만필터 업데이트
        find_nearby_objects 함수를 통해 근처에 같은 라벨이 가지는 노드가 있으면 하나로 통합. 통합하는 과정에서도 칼만필터 기반 중심점 보정 수행
        근처에 발견된 객체가 없다면 새로운 객체 노드에 추가.
        Add object node or update existing one if nearby object found.
        
        Args:
            pose_id: ID of pose that observed this object
            label: Object label
            position_w: World position (3,)
            confidence: Detection confidence
            embedding: Optional embedding
            step: Current step number
            distance_threshold: Distance threshold for matching (meters)
            clip_score: CLIP verification score (how confident CLIP is that the label is correct)
        
        Returns:
            ObjectNode (new or updated)
        """
        assert pose_id in self.nodes and self.nodes[pose_id].kind == "pose"
        
        # Try to find nearby object with same label
        existing_obj = self.find_nearby_objects(label, position_w, distance_threshold)
        
        if existing_obj is not None:
            # Update existing object (no new edge created - edge only on first observation)
            updated_obj = self.update_object_node(
                existing_obj,
                position_w,
                confidence,
                embedding,
                step or self._step_counter,
            )
            
            # CLIP 점수 누적
            if clip_score is not None:
                updated_obj.clip_scores.append(clip_score)
            
            # Remove object if confidence is too low after multiple observations
            min_observations_for_check = 2
            if updated_obj.confidence_weighted is not None:
                min_confidence_threshold = 0.8 # 다중 관측
                conf_value = updated_obj.confidence_weighted
            else:
                min_confidence_threshold = 0.5
                conf_value = updated_obj.confidence
            if (updated_obj.num_observations >= min_observations_for_check and
                conf_value < min_confidence_threshold):
                self._remove_object_node(updated_obj.id)
                return None  # Object was removed
            
            # Update database if available
            if self.db:
                try:
                    self.db.add_object_node(updated_obj, session_id=self.session_id)
                except Exception as e:
                    # Silently fail if DB save fails
                    pass
            
            return updated_obj
        else:
            # Create new object node
            node_id = self._new_id("obj")
            obj_node = ObjectNode(
                id=node_id,
                kind="object",
                label=label,
                position=position_w.copy(),
                confidence=confidence,
                embedding=normalize_embedding(embedding),
                num_observations=1,
                last_seen_step=step or self._step_counter,
                position_covariance=np.eye(2) * 0.1,  # Initial covariance
                clip_scores=[clip_score] if clip_score is not None else [],  # CLIP 검증 점수 초기화
            )
            self._add_node(obj_node)

            # pose 기준 상대 위치 edge 생성
            pose_node: PoseNode = self.nodes[pose_id]  # type: ignore
            rel_pos = world_to_local_2d(
                pose_node.x, pose_node.y, pose_node.theta,
                position_w[0], position_w[1]
            )

            edge_id = self._new_id("e_po")
            edge = Edge(
                id=edge_id,
                kind="pose_object",
                src=pose_id,
                dst=node_id,
                rel_pos=rel_pos,
            )
            self._add_edge(edge)

            # Save to database if available
            if self.db:
                try:
                    self.db.add_object_node(obj_node, session_id=self.session_id)
                except Exception as e:
                    # Silently fail if DB save fails
                    pass

            return obj_node

    def add_object_nodes_batch(
        self,
        pose_id: str,
        observations: List[dict],
        step: Optional[int] = None,
        distance_threshold: float = 3.0,
        use_kalman: bool = True,
        mahalanobis_threshold: float = 3.0,
        sim_threshold: float = 0.0,
        w_geo: float = 1.0,
        w_app: float = 1.0,
        max_candidates_per_obs: int = 50,
    ) -> List[Optional[ObjectNode]]:
        """
        navigator.py에서 직접적으로 사용하는 함수 : 한 프레임에서 여러 객체를 처리하는 용도.
        Add multiple object nodes using Hungarian assignment with Mahalanobis gating.
        
        Args:
            pose_id: ID of pose that observed these objects
            observations: List of observation dicts, each with keys:
                - label: str
                - position_w: np.ndarray (3,)
                - confidence: float
                - embedding: Optional[np.ndarray]
                - clip_score: Optional[float] (CLIP 검증 점수)
            step: Current step number
            distance_threshold: Maximum Euclidean distance threshold (meters)
            use_kalman: Whether to use Kalman filter prediction
            mahalanobis_threshold: Maximum Mahalanobis distance threshold (sigma)
        
        Returns:
            List of ObjectNode (updated or newly created), None for removed objects
        """
        assert pose_id in self.nodes and self.nodes[pose_id].kind == "pose"
        
        if not observations:
            return []
        
        matches, unmatched_obs_idx, _ = self._associate_observations_hungarian(
            observations=observations,
            candidate_obj_ids=None,
            distance_threshold=distance_threshold,
            mahalanobis_threshold=mahalanobis_threshold,
            sim_threshold=sim_threshold,
            w_geo=w_geo,
            w_app=w_app,
            max_candidates_per_obs=max_candidates_per_obs,
        )

        results: List[Optional[ObjectNode]] = [None] * len(observations)

        for obs_idx, obj_id in matches:
            obs = observations[obs_idx]
            obj_node = self.nodes[obj_id]
            assert isinstance(obj_node, ObjectNode)

            updated_obj = self.update_object_node(
                obj_node,
                obs["position_w"],
                obs["confidence"],
                obs.get("embedding"),
                step or self._step_counter,
            )

            best_label = self._update_label_belief(
                updated_obj,
                obs["label"],
                obs["confidence"],
                obs.get("clip_score"),
                q_view=1.0,
            )
            self._apply_label_hysteresis(updated_obj, best_label)
            self._apply_semantic_decay(updated_obj)

            min_observations_for_check = 2
            if updated_obj.confidence_weighted is not None:
                prune_conf = updated_obj.confidence_weighted
                prune_threshold = 0.8
            else:
                prune_conf = updated_obj.confidence
                prune_threshold = 0.5
            if (updated_obj.num_observations >= min_observations_for_check and
                    prune_conf < prune_threshold):
                self._remove_object_node(updated_obj.id)
                results[obs_idx] = None
            else:
                if self.db:
                    try:
                        self.db.add_object_node(updated_obj, session_id=self.session_id)
                    except Exception:
                        pass
                results[obs_idx] = updated_obj

        for obs_idx in unmatched_obs_idx:
            obs = observations[obs_idx]
            node_id = self._new_id("obj")
            obj_node = ObjectNode(
                id=node_id,
                kind="object",
                label=obs["label"],
                position=obs["position_w"].copy(),
                confidence=obs["confidence"],
                embedding=normalize_embedding(obs.get("embedding")),
                num_observations=1,
                last_seen_step=step or self._step_counter,
                position_covariance=np.eye(2) * 0.1,
            )
            obj_node.label_scores = {obs["label"]: float(obs["confidence"])}
            obj_node.label_final = obs["label"]
            obj_node.label_second = None
            obj_node.label_margin = 0.0
            obj_node.best_label_prev = obs["label"]
            obj_node.consecutive_best_label = 1
            self._add_node(obj_node)

            pose_node: PoseNode = self.nodes[pose_id]  # type: ignore
            rel_pos = world_to_local_2d(
                pose_node.x, pose_node.y, pose_node.theta,
                obs["position_w"][0], obs["position_w"][1]
            )
            edge_id = self._new_id("e_po")
            edge = Edge(
                id=edge_id,
                kind="pose_object",
                src=pose_id,
                dst=node_id,
                rel_pos=rel_pos,
            )
            self._add_edge(edge)

            self._apply_semantic_decay(obj_node)
            if self.db:
                try:
                    self.db.add_object_node(obj_node, session_id=self.session_id)
                except Exception:
                    pass
            results[obs_idx] = obj_node

        return results

    def add_frontier_node( # 프런티어 노드를 추가하는 함수
        self,
        pose_id: str,
        position_w: np.ndarray,
        coarse_embedding: Optional[np.ndarray] = None,
        semantic_hint: Optional[str] = None,
    ) -> FrontierNode:
        """Add frontier node and connect to pose."""
        assert pose_id in self.nodes and self.nodes[pose_id].kind == "pose"
        node_id = self._new_id("fr")
        fr_node = FrontierNode(
            id=node_id,
            kind="frontier",
            position=position_w.copy(),
            coarse_embedding=normalize_embedding(coarse_embedding),
            semantic_hint=semantic_hint,
            is_explored=False,
        )
        self._add_node(fr_node)

        pose_node: PoseNode = self.nodes[pose_id]  # type: ignore
        # pose 기준 frontier 상대 좌표 (2D 변환)
        rel_pos = world_to_local_2d(
            pose_node.x, pose_node.y, pose_node.theta,
            position_w[0], position_w[1]
        )

        edge_id = self._new_id("e_pf")
        edge = Edge(
            id=edge_id,
            kind="pose_frontier",
            src=pose_id,
            dst=node_id,
            rel_pos=rel_pos,
        )
        self._add_edge(edge)
        
        # Save to database if available
        if self.db:
            try:
                self.db.add_frontier_node(fr_node, session_id=self.session_id)
            except Exception as e:
                # Silently fail if DB save fails
                pass

        return fr_node
    
    def _remove_frontier_node(self, frontier_id: str) -> None: # 프런티어를 제거하는 내부 함수
        """
        Remove a frontier node and all its connected edges.
        
        Args:
            frontier_id: ID of frontier node to remove
        """
        if frontier_id not in self.nodes:
            return
        
        # Remove edges connected to this frontier
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.dst == frontier_id or edge.src == frontier_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        # Remove from frontier_ids list
        if frontier_id in self.frontier_ids:
            self.frontier_ids.remove(frontier_id)
        
        # Remove node
        del self.nodes[frontier_id]
        self._dirty = True
        
        # Remove from database if available
        if self.db:
            try:
                self.db.remove_frontier_node(frontier_id)
            except Exception as e:
                # Silently fail if DB removal fails
                pass

    def add_region_node( # 영역 노드를 추가하는 함수 (수정 필요)
        self,
        name: str,
        center_w: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        member_object_ids: Optional[List[str]] = None,
        member_frontier_ids: Optional[List[str]] = None,
    ) -> RegionNode:
        """Add region node."""
        node_id = self._new_id("rg")
        region_node = RegionNode(
            id=node_id,
            kind="region",
            name=name,
            center=center_w.copy(),
            embedding=normalize_embedding(embedding),
            member_object_ids=member_object_ids or [],
            member_frontier_ids=member_frontier_ids or [],
        )
        self._add_node(region_node)
        return region_node

## frontier to region 
    def promote_frontier_to_region( # frontier을 region 노드로 승격시키는 함수 (수정 필요)
        self,
        frontier_id: str,
        name: str,
        region_embedding: Optional[np.ndarray] = None,
    ) -> RegionNode:
        """
        1) frontier를 explored로 marking
        2) frontier 위치를 중심으로 region node 생성
        3) frontier_id를 region의 member_frontier_ids에 넣고,
           필요하다면 frontier를 그래프에서 제거하는 정책도 추가 가능
        """
        assert frontier_id in self.nodes
        frontier = self.nodes[frontier_id]
        assert isinstance(frontier, FrontierNode)

        frontier.is_explored = True

        region_node = self.add_region_node(
            name=name,
            center_w=frontier.position,
            embedding=region_embedding,
            member_object_ids=[],
            member_frontier_ids=[frontier_id]
        )
        return region_node

    def log_to_rerun(self, one_map) -> None: # rerun 로깅
        """Log the current pose graph into rerun aligned with the map space."""
        if not self._dirty or not self.pose_ids:
            return

        # Pose nodes만 픽셀 좌표로 변환
        node_pixels = []
        for pose_id in self.pose_ids:
            pose_node = self.nodes[pose_id]
            assert isinstance(pose_node, PoseNode)
            px, py = one_map.metric_to_px(pose_node.x, pose_node.y)
            node_pixels.append([py, px])
        
        if node_pixels:
            node_pixels_array = np.array(node_pixels, dtype=np.float32)
            colors = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (node_pixels_array.shape[0], 1))
            rr.log("map/pose_graph/nodes", rr.Points2D(node_pixels_array, colors=colors, radii=[0.5] * len(node_pixels_array)))

        self._log_edges(one_map, node_pixels_array if node_pixels else np.array([], dtype=np.float32))
        
        # Log object nodes
        self._log_objects(one_map)
        
        # Log frontier nodes
        self._log_frontiers(one_map)

        self._dirty = False

    def _log_edges(self, one_map, node_pixels: np.ndarray) -> None: # 엣지 로깅
        """Log edges to rerun."""
        if node_pixels.size == 0:
            return
        
        # pose_pose edges만 로깅 (기존 odometry)
        pose_pose_edges: List[Sequence[Sequence[float]]] = []
        
        for edge in self.edges.values():
            if edge.kind == "pose_pose":
                # src와 dst가 pose_ids에 있는지 확인
                if edge.src in self.pose_ids and edge.dst in self.pose_ids:
                    src_idx = self.pose_ids.index(edge.src)
                    dst_idx = self.pose_ids.index(edge.dst)
                    if src_idx < len(node_pixels) and dst_idx < len(node_pixels):
                        segment = [node_pixels[src_idx], node_pixels[dst_idx]]
                        pose_pose_edges.append(segment)

        if pose_pose_edges:
            rr.log("map/pose_graph/edges/odometry",
                   rr.LineStrips2D(np.array(pose_pose_edges, dtype=np.float32),
                                   colors=[[51, 153, 255]] * len(pose_pose_edges)))
     
    def _log_objects(self, one_map) -> None: # 객체 로깅
        if not self.object_ids:
            return
        
        # Object nodes를 픽셀 좌표로 변환
        object_pixels = []
        object_labels = []
        for obj_id in self.object_ids:
            obj_node = self.nodes[obj_id]
            assert isinstance(obj_node, ObjectNode)
            px, py = one_map.metric_to_px(obj_node.position[0], obj_node.position[1])
            object_pixels.append([py, px])
            object_labels.append(f"{obj_node.label} ({obj_node.confidence:.2f})")
        
        if object_pixels:
            object_pixels_array = np.array(object_pixels, dtype=np.float32)
            # 객체는 빨간색 별 모양으로 표시
            colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (object_pixels_array.shape[0], 1))
            rr.log("map/pose_graph/objects", 
                   rr.Points2D(object_pixels_array, colors=colors, radii=[1.0] * len(object_pixels_array)))
            # Explored map overlay
            rr.log("map/explored_objects",
                   rr.Points2D(object_pixels_array, colors=colors, radii=[1.0] * len(object_pixels_array)))
            
            # 객체 레이블 표시 (텍스트는 별도로 로깅)
            for pixel, label in zip(object_pixels_array, object_labels):
                # 텍스트 엔트리로 레이블 표시
                rr.log("map/pose_graph/object_labels",
                       rr.TextLog(label, color=[255, 0, 0]))
            
            # Pose-object edges 시각화
            pose_object_edges: List[Sequence[Sequence[float]]] = []
            for edge in self.edges.values():
                if edge.kind == "pose_object":
                    if edge.src in self.pose_ids and edge.dst in self.object_ids:
                        src_idx = self.pose_ids.index(edge.src)
                        dst_idx = self.object_ids.index(edge.dst)
                        pose_node = self.nodes[edge.src]
                        assert isinstance(pose_node, PoseNode)
                        pose_px, pose_py = one_map.metric_to_px(pose_node.x, pose_node.y)
                        obj_pixel = object_pixels_array[dst_idx]
                        segment = [[pose_py, pose_px], obj_pixel]
                        pose_object_edges.append(segment)
            
            if pose_object_edges:
                rr.log("map/pose_graph/edges/pose_object",
                       rr.LineStrips2D(np.array(pose_object_edges, dtype=np.float32),
                                       colors=[[255, 0, 255]] * len(pose_object_edges)))
                # Also log pose-object edges on explored map
                rr.log("map/explored_edges/pose_object",
                       rr.LineStrips2D(np.array(pose_object_edges, dtype=np.float32),
                                       colors=[[255, 0, 255]] * len(pose_object_edges)))

    def _log_frontiers(self, one_map) -> None: # 프런티어 로깅
        """Log frontier nodes to rerun (same logic as habitat_test.py)."""
        if not self.frontier_ids:
            return
        
        # Convert frontier node positions to pixel coordinates (same as object nodes)
        frontier_coords = []
        for fr_id in self.frontier_ids:
            if fr_id not in self.nodes:
                continue
            fr_node = self.nodes[fr_id]
            assert isinstance(fr_node, FrontierNode)
            px, py = one_map.metric_to_px(fr_node.position[0], fr_node.position[1])
            # Convert from [y, x] to [x, y] for rerun visualization (same as path and object nodes)
            frontier_coords.append([py, px])
        
        if len(frontier_coords) > 0:
            # Create small circles for each frontier (same as habitat_test.py)
            radius = 1.5  # Smaller radius
            num_points = 32  # Number of points to approximate circle
            
            # Generate circle points
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            circle_x = np.cos(angles) * radius
            circle_y = np.sin(angles) * radius
            
            # Create LineStrips2D for each frontier
            circle_strips = []
            green_color = [0, 255, 0]  # Green color
            
            for center in frontier_coords:
                circle = np.array([
                    [center[0] + x, center[1] + y] 
                    for x, y in zip(circle_x, circle_y)
                ], dtype=np.float32)
                # Close the circle
                circle = np.vstack([circle, circle[0:1]])
                circle_strips.append(circle)
            
            # Log all circles (same path as habitat_test.py)
            rr.log("map/frontiers_only", 
                   rr.LineStrips2D(circle_strips, 
                                   colors=[green_color] * len(circle_strips)))
            
            #### frontier-pose edge visualization option ####
            # # Also log pose-frontier edges (same pattern as pose-object edges)
            # pose_frontier_edges: List[Sequence[Sequence[float]]] = []
            # for edge in self.edges.values():
            #     if edge.kind == "pose_frontier":
            #         if edge.src in self.pose_ids and edge.dst in self.frontier_ids:
            #             # Get pose pixel coordinates (recalculate like object edges)
            #             pose_node = self.nodes[edge.src]
            #             assert isinstance(pose_node, PoseNode)
            #             pose_px, pose_py = one_map.metric_to_px(pose_node.x, pose_node.y)
                        
            #             # Get frontier pixel coordinates
            #             fr_node = self.nodes[edge.dst]
            #             assert isinstance(fr_node, FrontierNode)
            #             fr_px, fr_py = one_map.metric_to_px(fr_node.position[0], fr_node.position[1])
            #             fr_pixel = [fr_py, fr_px]
                        
            #             segment = [[pose_py, pose_px], fr_pixel]
            #             pose_frontier_edges.append(segment)
            
            # if pose_frontier_edges:
            #     # Green color for pose-frontier edges (same as frontier circles)
            #     pose_frontier_edges_array = np.array(pose_frontier_edges, dtype=np.float32)
            #     rr.log("map/pose_graph/edges/pose_frontier",
            #            rr.LineStrips2D(pose_frontier_edges_array,
            #                            colors=[[0, 255, 0]] * len(pose_frontier_edges)))
            #     # Also log pose-frontier edges on explored map
            #     rr.log("map/explored_edges/pose_frontier",
            #            rr.LineStrips2D(pose_frontier_edges_array,
            #                            colors=[[0, 255, 0]] * len(pose_frontier_edges)))
            ###################################################
    def load_from_database(self, session_id: Optional[int] = None) -> None: # 데이터베이스에서 로드가 되는 기능 (구현 예정)
        """Load pose graph from database."""
        if not self.db:
            return
        
        # TODO: 데이터베이스 로딩 로직 업데이트 필요
        # 기존 형식과 새 형식 간 변환 필요
        self._dirty = True

    def get_trajectory_length(self) -> float: # 로봇의 총 이동 거리 계산 (디버깅, 구체적으로 쓰이진 않음)
        """Calculate total trajectory length from pose_pose edges."""
        total_length = 0.0
        for edge in self.edges.values():
            if edge.kind == "pose_pose":
                if edge.src in self.nodes and edge.dst in self.nodes:
                    src_node = self.nodes[edge.src]
                    dst_node = self.nodes[edge.dst]
                    if isinstance(src_node, PoseNode) and isinstance(dst_node, PoseNode):
                        length = np.hypot(dst_node.x - src_node.x, dst_node.y - src_node.y)
                        total_length += length
        return total_length

    def _remove_object_node(self, obj_id: str) -> None: # object 노드 삭제 함수
        """
        Remove an object node and all its connected edges.
        
        Args:
            obj_id: ID of object node to remove
        """
        if obj_id not in self.nodes:
            return
        
        obj_node = self.nodes[obj_id]
        if isinstance(obj_node, ObjectNode):
            # Remove from spatial index
            cell = self._get_grid_cell(obj_node.position[0], obj_node.position[1])
            if obj_node.label in self._object_spatial_index:
                if cell in self._object_spatial_index[obj_node.label]:
                    if obj_id in self._object_spatial_index[obj_node.label][cell]:
                        self._object_spatial_index[obj_node.label][cell].remove(obj_id)
                    # Clean up empty cells
                    if not self._object_spatial_index[obj_node.label][cell]:
                        del self._object_spatial_index[obj_node.label][cell]
        
        # Remove edges connected to this object
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.dst == obj_id or edge.src == obj_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        # Remove from object_ids list
        if obj_id in self.object_ids:
            self.object_ids.remove(obj_id)
        
        # Remove node
        del self.nodes[obj_id]
        self._dirty = True
    
    def find_objects_by_label(
        self,
        target_label: str,
        min_confidence: float = 0.5,
        min_observations: int = 2,
    ) -> List[ObjectNode]:
        """
        특정 라벨을 가진 모든 ObjectNode 반환
        
        Args:
            target_label: 찾고자 하는 객체 라벨
            min_confidence: 최소 신뢰도 임계값 (기본값 0.5, 제거 임계값 0.8보다 낮음)
            min_observations: 최소 관측 횟수 (노이즈 필터링)
        
        Returns:
            조건을 만족하는 ObjectNode 리스트
        """
        results = []
        
        # 공간 인덱스에서 해당 라벨의 모든 객체 검색
        if target_label not in self._object_spatial_index:
            return results
        
        for cell_objects in self._object_spatial_index[target_label].values():
            for obj_id in cell_objects:
                if obj_id not in self.nodes:
                    continue
                obj_node = self.nodes[obj_id]
                if not isinstance(obj_node, ObjectNode):
                    continue
                
                # 신뢰도 및 관측 횟수 필터링
                if (obj_node.confidence >= min_confidence and 
                    obj_node.num_observations >= min_observations):
                    results.append(obj_node)
        
        return results

    def find_nearest_object_by_label(
        self,
        target_label: str,
        robot_position: np.ndarray,
        min_confidence: float = 0.5,
        min_observations: int = 2,
    ) -> Optional[ObjectNode]:
        """
        로봇 위치 기준 가장 가까운 목표 객체 반환
        
        Args:
            target_label: 찾고자 하는 객체 라벨
            robot_position: 로봇의 현재 위치 (2,) 또는 (3,)
            min_confidence: 최소 신뢰도 임계값
            min_observations: 최소 관측 횟수
        
        Returns:
            가장 가까운 ObjectNode, 없으면 None
        """
        candidates = self.find_objects_by_label(target_label, min_confidence, min_observations)
        
        if not candidates:
            return None
        
        robot_pos_2d = robot_position[:2]
        
        # 거리 기준 정렬하여 가장 가까운 객체 반환
        candidates_with_dist = [
            (obj, np.linalg.norm(obj.position[:2] - robot_pos_2d))
            for obj in candidates
        ]
        candidates_with_dist.sort(key=lambda x: x[1])
        
        return candidates_with_dist[0][0]

    def find_all_objects_sorted_by_distance(
        self,
        target_label: str,
        robot_position: np.ndarray,
        min_confidence: float = 0.5,
        min_observations: int = 2,
    ) -> List[Tuple[ObjectNode, float]]:
        """
        로봇 위치 기준 거리순으로 정렬된 모든 목표 객체 반환
        (경로 계획 실패 시 다음 가까운 객체를 시도하기 위함)
        
        Args:
            target_label: 찾고자 하는 객체 라벨
            robot_position: 로봇의 현재 위치 (2,) 또는 (3,)
            min_confidence: 최소 신뢰도 임계값
            min_observations: 최소 관측 횟수
        
        Returns:
            (ObjectNode, 거리) 튜플 리스트, 거리순 정렬
        """
        candidates = self.find_objects_by_label(target_label, min_confidence, min_observations)
        
        if not candidates:
            return []
        
        robot_pos_2d = robot_position[:2]
        
        # 거리 기준 정렬
        candidates_with_dist = [
            (obj, np.linalg.norm(obj.position[:2] - robot_pos_2d))
            for obj in candidates
        ]
        candidates_with_dist.sort(key=lambda x: x[1])
        
        return candidates_with_dist

    def get_statistics(self) -> dict: # 그래프 통계 정보 반환
        """Get pose graph statistics."""
        stats = {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'pose_nodes': len(self.pose_ids),
            'object_nodes': len(self.object_ids),
            'frontier_nodes': len(self.frontier_ids),
            'region_nodes': len(self.region_ids),
            'pose_pose_edges': len([e for e in self.edges.values() if e.kind == "pose_pose"]),
            'pose_object_edges': len([e for e in self.edges.values() if e.kind == "pose_object"]),
            'pose_frontier_edges': len([e for e in self.edges.values() if e.kind == "pose_frontier"]),
            'pose_region_edges': len([e for e in self.edges.values() if e.kind == "pose_region"]),
            'trajectory_length': self.get_trajectory_length(),
        }
        
        if self.db:
            db_stats = self.db.get_graph_statistics()
            stats.update(db_stats)
        
        return stats

    def export_to_file(self, filepath: str) -> None: # 그래프를 JSON 파일로 내보내는 함수
        """Export pose graph to JSON file."""
        import json
        
        data = {
            'nodes': [],
            'edges': [],
            'statistics': self.get_statistics()
        }
        
        # 노드 내보내기
        for node in self.nodes.values():
            node_data = {'id': node.id, 'kind': node.kind}
            if isinstance(node, PoseNode):
                node_data.update({'x': node.x, 'y': node.y, 'theta': node.theta, 'step': node.step})
            elif isinstance(node, ObjectNode):
                node_data.update({
                    'label': node.label,
                    'position': node.position.tolist(),
                    'confidence': node.confidence,
                    'num_observations': node.num_observations,
                    'last_seen_step': node.last_seen_step
                })
            elif isinstance(node, FrontierNode):
                node_data.update({
                    'position': node.position.tolist(),
                    'semantic_hint': node.semantic_hint,
                    'is_explored': node.is_explored
                })
            elif isinstance(node, RegionNode):
                node_data.update({
                    'name': node.name,
                    'center': node.center.tolist(),
                    'member_object_ids': node.member_object_ids,
                    'member_frontier_ids': node.member_frontier_ids
                })
            data['nodes'].append(node_data)
        
        # 엣지 내보내기
        for edge in self.edges.values():
            edge_data = {
                'id': edge.id,
                'kind': edge.kind,
                'src': edge.src,
                'dst': edge.dst
            }
            if edge.rel_pos is not None:
                edge_data['rel_pos'] = edge.rel_pos.tolist()
            data['edges'].append(edge_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def close(self) -> None: # 데이터베이스 연결 닫기 (구현 예정)
        """Close database connection."""
        if self.db:
            self.db.close()
