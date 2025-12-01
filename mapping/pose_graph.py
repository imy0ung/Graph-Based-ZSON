from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence
import time
import uuid
from collections import defaultdict

import numpy as np
import rerun as rr

# L2 Regularization for embedding
def normalize_embedding(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None:
        return None
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm


def world_to_local_2d(pose_x: float, pose_y: float, pose_theta: float,
                      world_x: float, world_y: float) -> np.ndarray:
    """
    Convert world coordinates to pose-local coordinates (2D).
    
    Args:
        pose_x, pose_y, pose_theta: Pose position and orientation in world frame
        world_x, world_y: Point position in world frame
    
    Returns:
        (3,) array: [local_x, local_y, 0] in pose-local frame
    """
    # Translate to pose origin
    dx = world_x - pose_x
    dy = world_y - pose_y
    
    # Rotate by -theta (inverse rotation)
    cos_theta = np.cos(pose_theta)
    sin_theta = np.sin(pose_theta)
    local_x = dx * cos_theta + dy * sin_theta
    local_y = -dx * sin_theta + dy * cos_theta
    
    return np.array([local_x, local_y, 0.0])


# 노드 타입 정의
NodeKind = Literal["pose", "object", "frontier", "region"]
EdgeKind = Literal["pose_pose", "pose_object", "pose_frontier", "pose_region"]

# 호환성을 위한 타입 별칭
PoseEdgeType = EdgeKind

############## node 정의 ###############
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
    num_observations: int = 1
    last_seen_step: int = 0
    # Kalman filter state (for improved matching)
    position_covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.1)  # (2,2) covariance for x,y


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

############## node 정의 ###############

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
        """호환성을 위한 source (node_id 추출)"""
        try:
            return int(self.src.split('_')[-1])
        except:
            return hash(self.src) % (2**31)
    
    @property
    def target(self) -> int:
        """호환성을 위한 target (node_id 추출)"""
        try:
            return int(self.dst.split('_')[-1])
        except:
            return hash(self.dst) % (2**31)
    
    @property
    def edge_type(self) -> EdgeKind:
        """호환성을 위한 edge_type"""
        return self.kind


class PoseGraph:
    """Minimal pose-graph container with rerun logging helpers and database storage."""

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
        if db_path:
            from .pose_graph_db import PoseGraphDB
            self.db = PoseGraphDB(db_path)
            if session_name:
                self.session_id = self.db.create_session(
                    session_name, 
                    f"Mapping session started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
        
        self._step_counter = 0

    def _new_id(self, prefix: str) -> str:
        """Generate new unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_grid_cell(self, x: float, y: float) -> tuple:
        """Get grid cell coordinates for spatial indexing."""
        cell_x = int(x / self._grid_cell_size)
        cell_y = int(y / self._grid_cell_size)
        return (cell_x, cell_y)
    
    def _get_nearby_grid_cells(self, x: float, y: float, radius: float) -> List[tuple]:
        """Get nearby grid cells within radius."""
        center_cell = self._get_grid_cell(x, y)
        cells = []
        radius_cells = int(np.ceil(radius / self._grid_cell_size))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cells.append((center_cell[0] + dx, center_cell[1] + dy))
        return cells
    
    def _add_node(self, node: BaseNode):
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

    def _add_edge(self, edge: Edge):
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

    @staticmethod
    def _angle_diff(rad_a: float, rad_b: float) -> float:
        diff = rad_a - rad_b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def add_pose(self, x: float, y: float, theta: float) -> int:
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
        Predict object position using Kalman filter prediction step.
        For static objects, position remains constant (only uncertainty increases).
        
        Args:
            obj_node: Object node with Kalman filter state
            steps_since_last_seen: Number of steps since last observation
        
        Returns:
            Predicted position (2,) for x, y (same as current position for static objects)
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
        Calculate Mahalanobis distance between observed and predicted position.
        
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
        Find nearby object with the same label using Kalman filter-based matching.
        
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
        Update existing object node with new observation using Kalman filter update.
        
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
        
        # Update embedding if provided (weighted average if both exist)
        if new_embedding is not None:
            new_embedding = normalize_embedding(new_embedding)
            if obj_node.embedding is not None:
                # Weighted average of embeddings
                obj_node.embedding = (
                    np.array(obj_node.embedding) * old_weight + np.array(new_embedding) * new_weight
                ) / total_weight
                obj_node.embedding = normalize_embedding(obj_node.embedding)
            else:
                obj_node.embedding = new_embedding
        
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
    ) -> ObjectNode:
        """
        Add object node or update existing one if nearby object found.
        
        Args:
            pose_id: ID of pose that observed this object
            label: Object label
            position_w: World position (3,)
            confidence: Detection confidence
            embedding: Optional embedding
            step: Current step number
            distance_threshold: Distance threshold for matching (meters)
        
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
            
            # Remove object if confidence is too low after multiple observations
            min_observations_for_check = 2
            min_confidence_threshold = 0.8
            if (updated_obj.num_observations >= min_observations_for_check and
                updated_obj.confidence < min_confidence_threshold):
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
    ) -> List[Optional[ObjectNode]]:
        """
        Add multiple object nodes using greedy matching (sequential processing).
        
        Args:
            pose_id: ID of pose that observed these objects
            observations: List of observation dicts, each with keys:
                - label: str
                - position_w: np.ndarray (3,)
                - confidence: float
                - embedding: Optional[np.ndarray]
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
        
        results: List[Optional[ObjectNode]] = []
        
        # Process each observation sequentially (greedy matching)
        for obs in observations:
            obj_node = self.add_object_node(
                pose_id=pose_id,
                label=obs["label"],
                position_w=obs["position_w"],
                confidence=obs["confidence"],
                embedding=obs.get("embedding"),
                step=step or self._step_counter,
                distance_threshold=distance_threshold,
            )
            results.append(obj_node)
        
        return results

    def add_frontier_node(
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
    
    def _remove_frontier_node(self, frontier_id: str) -> None:
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

    def add_region_node(
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
    def promote_frontier_to_region(
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

    def log_to_rerun(self, one_map) -> None:
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

    def _log_edges(self, one_map, node_pixels: np.ndarray) -> None:
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
    
    def _log_objects(self, one_map) -> None:
        """Log object nodes to rerun."""
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

    def _log_frontiers(self, one_map) -> None:
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
            
            # Also log pose-frontier edges (same pattern as pose-object edges)
            pose_frontier_edges: List[Sequence[Sequence[float]]] = []
            for edge in self.edges.values():
                if edge.kind == "pose_frontier":
                    if edge.src in self.pose_ids and edge.dst in self.frontier_ids:
                        # Get pose pixel coordinates (recalculate like object edges)
                        pose_node = self.nodes[edge.src]
                        assert isinstance(pose_node, PoseNode)
                        pose_px, pose_py = one_map.metric_to_px(pose_node.x, pose_node.y)
                        
                        # Get frontier pixel coordinates
                        fr_node = self.nodes[edge.dst]
                        assert isinstance(fr_node, FrontierNode)
                        fr_px, fr_py = one_map.metric_to_px(fr_node.position[0], fr_node.position[1])
                        fr_pixel = [fr_py, fr_px]
                        
                        segment = [[pose_py, pose_px], fr_pixel]
                        pose_frontier_edges.append(segment)
            
            if pose_frontier_edges:
                # Green color for pose-frontier edges (same as frontier circles)
                pose_frontier_edges_array = np.array(pose_frontier_edges, dtype=np.float32)
                rr.log("map/pose_graph/edges/pose_frontier",
                       rr.LineStrips2D(pose_frontier_edges_array,
                                       colors=[[0, 255, 0]] * len(pose_frontier_edges)))
                # Also log pose-frontier edges on explored map
                rr.log("map/explored_edges/pose_frontier",
                       rr.LineStrips2D(pose_frontier_edges_array,
                                       colors=[[0, 255, 0]] * len(pose_frontier_edges)))

    def load_from_database(self, session_id: Optional[int] = None) -> None:
        """Load pose graph from database."""
        if not self.db:
            return
        
        # TODO: 데이터베이스 로딩 로직 업데이트 필요
        # 기존 형식과 새 형식 간 변환 필요
        self._dirty = True

    def get_trajectory_length(self) -> float:
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

    def get_trusted_objects(
        self,
        min_observations: int = 2,
        min_confidence: float = 0.3,
    ) -> List[ObjectNode]:
        """
        Get objects that meet minimum observation and confidence criteria.
        
        Args:
            min_observations: Minimum number of observations required
            min_confidence: Minimum average confidence required
        
        Returns:
            List of trusted ObjectNode objects
        """
        trusted = []
        for obj_id in self.object_ids:
            obj_node = self.nodes[obj_id]
            assert isinstance(obj_node, ObjectNode)
            if (obj_node.num_observations >= min_observations and
                obj_node.confidence >= min_confidence):
                trusted.append(obj_node)
        return trusted
    
    def _remove_object_node(self, obj_id: str) -> None:
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
    
    def remove_low_confidence_objects(
        self,
        min_observations: int = 2,
        min_confidence: float = 0.3,
    ) -> int:
        """
        Remove objects that don't meet minimum criteria (false positives).
        
        Args:
            min_observations: Minimum number of observations required
            min_confidence: Minimum average confidence required
        
        Returns:
            Number of objects removed
        """
        to_remove = []
        for obj_id in self.object_ids:
            obj_node = self.nodes[obj_id]
            assert isinstance(obj_node, ObjectNode)
            if (obj_node.num_observations < min_observations or
                obj_node.confidence < min_confidence):
                to_remove.append(obj_id)
        
        # Remove objects using helper method
        for obj_id in to_remove:
            self._remove_object_node(obj_id)
        
        return len(to_remove)
    
    def get_statistics(self) -> dict:
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

    def export_to_file(self, filepath: str) -> None:
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

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
