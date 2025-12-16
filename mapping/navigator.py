"""
This module contains the Navigator class, which is responsible for the main functionality. It updates Onemap and uses it
for navigation and exploration.
"""
import time

from mapping import (OneMap, PoseGraph, PoseNode, FrontierNode, detect_frontiers, get_frontier_midpoint,
                     cluster_high_similarity_regions, find_local_maxima,
                     watershed_clustering, gradient_based_clustering, cluster_thermal_image,
                     Cluster, NavGoal, Frontier)

from planning import Planning
from vision_models.base_model import BaseModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from onemap_utils import monochannel_to_inferno_rgb, log_map_rerun
from config import Conf, load_config
from config import SpotControllerConf
from mobile_sam import sam_model_registry, SamPredictor

# numpy
import numpy as np

# typing
from typing import List, Optional, Set, Any, Union, Tuple

# torch
import torch

# warnings
import warnings

# rerun
import rerun as rr

# cv2
import cv2


def closest_point_within_threshold(nav_goals: List[NavGoal], target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        nav_goals (List[NavGoal]): An array of potential nav points, where each point is retrieved by nav_goal.get_descr_point
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    points_array = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


class HistoricDetectData:
    def __init__(self, position: np.ndarray, action: str, other: Any = None):
        self.position = position
        self.other = other
        self.action = action

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.action}_{self.other}"
        return hash(string_repr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistoricDetectData):
            return NotImplemented
        return (np.array_equal(self.position, other.position) and
                self.action == other.action and
                self.other == other.other)


class CyclicDetectChecker:
    history: Set[HistoricDetectData] = set()

    def check_cyclic(self, position: np.ndarray, action: str, other: Any = None) -> bool:
        state_action = HistoricDetectData(position, action, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: str, other: Any = None) -> None:
        state_action = HistoricDetectData(position, action, other)
        self.history.add(state_action)


class HistoricData:
    def __init__(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None):
        self.position = position
        self.frontier_pt = frontier_pt
        self.other = other

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.frontier_pt}_{self.other}"
        return hash(string_repr)


class CyclicChecker:
    history: Set[HistoricData] = set()

    def check_cyclic(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> bool:
        state_action = HistoricData(position, frontier_pt, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> None:
        state_action = HistoricData(position, frontier_pt, other)
        self.history.add(state_action)


class Navigator:
    query_text: List[str]  # the query texts as a list. The first element will be used for planning and frontier score
    # computation
    query_text_features: torch.Tensor
    blacklisted_nav_goals: List[np.ndarray]
    nav_goals: List[NavGoal]
    last_nav_goal: Union[NavGoal, None]

    def __init__(self,
                 model: BaseModel,
                 detector: YOLOWorldDetector,
                 config: Conf
                 ) -> None:

        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.config = config

        # Models
        self.model = model
        self.detector = detector  # Single YOLO detector for both target and COCO detection
        sam_model_t = "vit_t"
        sam_checkpoint = "weights/mobile_sam.pt"
        self.sam = sam_model_registry[sam_model_t](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")
        self.sam.eval()
        self.sam_predictor = SamPredictor(self.sam)

        self.one_map = OneMap(self.model.feature_dim, config.mapping, map_device="cpu")
        
        # Initialize pose graph with database support
        # DB 저장 비활성화 (평가 시 인메모리만 사용)
        self.pose_graph = PoseGraph(db_path=None, session_name=None)

        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)

        # Current frame image features for frontier embedding extraction
        self.current_image_features: Optional[torch.Tensor] = None
        self.current_image: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None
        self.current_odometry: Optional[np.ndarray] = None

        # Frontier navigation goals
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []

        self.last_nav_goal = None
        self.last_frontier_node = None  # Track last frontier node for stuck detection
        self.last_pose = None
        self.saw_left = False
        self.saw_right = False
        
        # Track frontier nodes in pose graph: frontier_midpoint (tuple) -> frontier_node_id
        self.frontier_node_map: dict = {}

        self.first_obs = True
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.is_goal_path = False
        self.navigation_scores = np.zeros_like(self.one_map.navigable_map, dtype=np.float32)
        self.path = None
        self.is_spot = type(config.controller) == SpotControllerConf
        self.initializing = True
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0

        self.percentile_exploitation = config.planner.percentile_exploitation
        self.frontier_depth = int(config.planner.frontier_depth / self.one_map.cell_size)
        self.no_nav_radius = int(config.planner.no_nav_radius / self.one_map.cell_size)
        self.max_detect_distance = int(config.planner.max_detect_distance / self.one_map.cell_size)
        self.obstcl_kernel_size = int(config.planner.obstcl_kernel_size / self.one_map.cell_size)
        self.min_goal_dist = int(config.planner.min_goal_dist / self.one_map.cell_size)

        self.path_id = 0
        self.filter_detections_depth = config.planner.filter_detections_depth
        self.consensus_filtering = config.planner.consensus_filtering

        self.log = config.log_rerun
        self.allow_replan = config.planner.allow_replan
        self.use_frontiers = config.planner.use_frontiers
        self.allow_far_plan = config.planner.allow_far_plan

        # For the closed-vocabulary object detector, not needed for OneMap
        # 쿼리 텍스트를 COCO 라벨로 매핑 (동의어 처리)
        self.class_map = {
            # 기존 매핑
            "chair": "chair",
            "tv_monitor": "tv",
            "tv": "tv",
            "television": "tv",
            "monitor": "tv",
            "plant": "potted plant",
            "potted plant": "potted plant",
            "sofa": "couch",
            "couch": "couch",
            "bed": "bed",
            "toilet": "toilet",
            # COCO 라벨 추가 (YOLOv8 기준)
            "refrigerator": "refrigerator",
            "fridge": "refrigerator",
            "microwave": "microwave",
            "oven": "oven",
            "sink": "sink",
            "dining table": "dining table",
            "table": "dining table",
            "laptop": "laptop",
            "cell phone": "cell phone",
            "phone": "cell phone",
            "book": "book",
            "clock": "clock",
            "vase": "vase",
            "bottle": "bottle",
            "cup": "cup",
            "bowl": "bowl",
        }
        
        # 그래프 기반 목표 객체 추적
        self.target_object_node = None  # 현재 목표로 설정된 ObjectNode

    def reset(self):
        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.target_object_node = None  # Reset target object node tracking
        self.last_pose = None
        self.last_frontier_node = None  # Reset frontier node tracking
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self.is_goal_path = False
        self.navigation_scores = np.zeros_like(self.one_map.navigable_map, dtype=np.float32)
        self.path = None
        self.path_id = 0
        self.initializing = True
        self.one_map.reset()
        self.first_obs = True
        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []
        self.frontier_node_map = {}  # Reset frontier node mapping
        # Clear pose graph for fair evaluation (each episode starts fresh)
        self.pose_graph.clear()

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        self.one_map.set_camera_matrix(camera_matrix)

    def set_query(self,
                  txt: List[str]
                  ) -> None:
        """
        Sets the query text
        :param txt: List of strings
        :return:
        """
        for t in txt:
            if t in self.class_map:
                txt[txt.index(t)] = self.class_map[t]
        if txt != self.query_text:
            print(f"Setting query to {txt}")
            self.query_text = txt
            self.query_text_features = self.model.get_text_features(["a " + self.query_text[0]]).to(
                self.one_map.map_device)
            self.one_map.reset_checked_map()
            self.detector.set_classes(self.query_text)
            # Reset object detection state when query changes (for multi-object navigation)
            self.object_detected = False
            self.chosen_detection = None
            self.target_object_node = None  # Reset target object node
            self.path = None  # Clear path to allow new exploration

    def find_target_object_in_graph(
        self,
        robot_x: float,
        robot_y: float,
        min_confidence: float = 0.5,
        min_observations: int = 2,
    ) -> Optional[Tuple[np.ndarray, Any]]:
        """
        PoseGraph에서 쿼리 텍스트에 매칭되는 가장 가까운 객체 검색.
        경로 계획이 가능한 객체만 반환.
        
        Args:
            robot_x: 로봇 x 좌표 (metric)
            robot_y: 로봇 y 좌표 (metric)
            min_confidence: 최소 신뢰도 임계값 (기본값 0.5, 제거 임계값 0.8보다 낮게 설정)
            min_observations: 최소 관측 횟수 (노이즈 필터링)
        
        Returns:
            (목표 픽셀 좌표 [px, py], ObjectNode) 또는 None
        """
        # class_map을 통해 쿼리 텍스트를 정규화된 라벨로 변환
        query = self.query_text[0].lower()
        normalized_label = self.class_map.get(query, query)
        
        # 그래프에서 거리순으로 정렬된 모든 객체 검색
        robot_pos = np.array([robot_x, robot_y])
        candidates = self.pose_graph.find_all_objects_sorted_by_distance(
            target_label=normalized_label,
            robot_position=robot_pos,
            min_confidence=min_confidence,
            min_observations=min_observations,
        )
        
        if not candidates:
            return None
        
        # 각 객체에 대해 경로 계획 가능 여부 확인 (거리 가까운 순서대로)
        start_px, start_py = self.one_map.metric_to_px(robot_x, robot_y)
        start = np.array([start_px, start_py])
        explored_mask = self.one_map.explored_area.astype(np.uint8)
        
        for target_obj, distance in candidates:
            # 월드 좌표를 픽셀 좌표로 변환
            px, py = self.one_map.metric_to_px(
                target_obj.position[0], 
                target_obj.position[1]
            )
            target_px = np.array([px, py])
            
            # 경로 계획 시도
            test_path = Planning.compute_to_goal(
                start,
                self.one_map.navigable_map,
                explored_mask,
                target_px,
                self.obstcl_kernel_size,
                self.min_goal_dist
            )
            
            if test_path is not None and len(test_path) > 0:
                # 경로 계획 성공 → 이 객체 반환
                return target_px, target_obj
        
        # 모든 객체에 대해 경로 계획 실패
        return None

    def get_path(self
                 ) -> Union[np.ndarray, str]:
        if not self.path:
            return None
        if not self.object_detected:
            if self.saw_left:
                return "L"
            if self.saw_right:
                return "R"
        return self.path[min(self.path_id, len(self.path)):]

    def compute_best_path(self,
                          start: np.ndarray) -> None:
        """
        Computes the best path from the start point to a frontier node with highest similarity.
        Uses PoseGraph FrontierNodes and computes similarity between coarse_embedding and query text.
        :param start: start point as [X, Y]
        :return:
        """
        self.path_id = 0
        if not self.object_detected:
            # We are exploring
            # First, ensure frontiers are computed
            if len(self.nav_goals) == 0:
                if not self.initializing:
                    self.one_map.reset_checked_map()  # We need new points of interest
                    self.compute_frontiers_and_POIs(start[0], start[1])
            
            # Get all FrontierNodes from PoseGraph
            frontier_nodes = []
            for frontier_id in self.pose_graph.frontier_ids:
                node = self.pose_graph.nodes[frontier_id]
                if isinstance(node, FrontierNode) and not node.is_explored:
                    frontier_nodes.append(node)
            
            # If no frontier nodes, fall back to nav_goals
            if len(frontier_nodes) == 0:
                if len(self.nav_goals) == 0:
                    return
                # Fall back to old logic if no FrontierNodes available
                self._compute_best_path_legacy(start)
                return
            
            self.initializing = False
            self.is_goal_path = False
            self.path = None
            
            # Compute similarity for each FrontierNode (excluding blacklisted ones)
            best_frontier_node = None
            best_similarity = -float('inf')
            
            for frontier_node in frontier_nodes:
                if frontier_node.coarse_embedding is None:
                    continue  # Skip nodes without embedding
                
                # Check if this frontier is blacklisted
                goal_world = frontier_node.position[:2]  # [x, y]
                goal_px, goal_py = self.one_map.metric_to_px(goal_world[0], goal_world[1])
                goal_point_px = np.array([goal_px, goal_py])
                
                # Skip if blacklisted
                if len(self.blacklisted_nav_goals) > 0 and np.any(
                        np.all(goal_point_px == self.blacklisted_nav_goals, axis=1)):
                    continue
                
                # Convert embedding to torch tensor
                embedding_tensor = torch.from_numpy(frontier_node.coarse_embedding).float()
                # Reshape for similarity computation: [F] -> [1, F, 1, 1] for dense model compatibility
                if len(embedding_tensor.shape) == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                else:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                
                # Compute similarity
                similarity = self.model.compute_similarity(
                    embedding_tensor.to(self.query_text_features.device),
                    self.query_text_features
                )
                
                # Extract scalar similarity value
                if similarity.numel() == 1:
                    sim_value = similarity.item()
                else:
                    sim_value = similarity.mean().item()
                
                if sim_value > best_similarity:
                    best_similarity = sim_value
                    best_frontier_node = frontier_node
            
            # If we found a best frontier node, plan path to it
            if best_frontier_node is not None:
                # Convert world position to map pixel coordinates
                goal_world = best_frontier_node.position[:2]  # [x, y]
                goal_px, goal_py = self.one_map.metric_to_px(goal_world[0], goal_world[1])
                goal_point = np.array([goal_px, goal_py])
                
                # Plan path using Grid Map (navigable_map)
                # Use explored_area instead of confidence_map > 0
                explored_mask = self.one_map.explored_area.astype(np.uint8)
                self.path = Planning.compute_to_goal(
                    start,
                    self.one_map.navigable_map & explored_mask,
                    explored_mask,
                    goal_point,
                    self.obstcl_kernel_size,
                    2
                )
                
                if self.path:
                    if self.log:
                        rr.log("path_updates", rr.TextLog(
                            f"Selected frontier node with similarity {best_similarity:.3f} at ({goal_world[0]:.2f}, {goal_world[1]:.2f})"
                        ))
                        # 경로 좌표를 [y, x]에서 [x, y]로 스왑
                        path_swapped = np.array(self.path)[:, [1, 0]]
                        rr.log("map/path", rr.LineStrips2D(
                            path_swapped,
                            colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :], len(self.path), axis=0)
                        ))
                    
                    # Track stuck state for this frontier
                    # Check if we're stuck (not moving towards goal)
                    if self.last_pose is not None:
                        # Check if position hasn't changed much and path is short (stuck indicator)
                        if (len(self.path) < 5 and 
                            self.last_pose[0] == start[0] and 
                            self.last_pose[1] == start[1]):
                            # Only increment if we're still targeting the same frontier
                            if (self.last_frontier_node is not None and 
                                self.last_frontier_node.id == best_frontier_node.id):
                                self.stuck_at_nav_goal_counter += 1
                            else:
                                # New frontier, reset counter
                                self.stuck_at_nav_goal_counter = 1
                        else:
                            # Reset counter if we're making progress or targeting different frontier
                            if (self.last_frontier_node is None or 
                                self.last_frontier_node.id != best_frontier_node.id):
                                self.stuck_at_nav_goal_counter = 0
                    # Update last frontier node
                    self.last_frontier_node = best_frontier_node
                    
                    # If stuck for too long, blacklist this frontier
                    if self.stuck_at_nav_goal_counter > 10:
                        # Blacklist this frontier
                        self.blacklisted_nav_goals.append(goal_point)
                        # Remove from pose graph
                        frontier_key = tuple(goal_point)
                        if frontier_key in self.frontier_node_map:
                            frontier_node_id = self.frontier_node_map[frontier_key]
                            self.pose_graph._remove_frontier_node(frontier_node_id)
                            del self.frontier_node_map[frontier_key]
                        # Mark as explored to prevent re-selection
                        best_frontier_node.is_explored = True
                        if self.log:
                            rr.log("path_updates", rr.TextLog(
                                f"Frontier at ({goal_world[0]:.2f}, {goal_world[1]:.2f}) blacklisted due to stuck state."
                            ))
                        # Clear path and try again
                        self.path = None
                        self.stuck_at_nav_goal_counter = 0
                        # Recursively try to find another frontier
                        if len(frontier_nodes) > 1:
                            self.compute_best_path(start)
                            return
                else:
                    # No path found - DON'T immediately blacklist, just try another frontier
                    # Only blacklist after repeated failures (handled by stuck_at_nav_goal_counter)
                    if self.log:
                        rr.log("path_updates", rr.TextLog(
                            f"No path found to frontier at ({goal_world[0]:.2f}, {goal_world[1]:.2f}), trying another."
                        ))
                    # Try to find another frontier without blacklisting
                    if len(frontier_nodes) > 1:
                        # Temporarily mark as explored to skip in this iteration
                        best_frontier_node.is_explored = True
                        self.compute_best_path(start)
                        # Restore for future iterations
                        best_frontier_node.is_explored = False
                        return
            else:
                # No valid frontier nodes with embeddings, fall back to nav_goals
                if len(self.nav_goals) > 0:
                    self._compute_best_path_legacy(start)
                else:
                    if self.log:
                        rr.log("path_updates", rr.TextLog("No frontier nodes or nav goals available."))
        else:
            # Object detection path (unchanged)
            if np.linalg.norm(start - self.chosen_detection) < self.max_detect_distance:
                self.path = [start] * 5
                return
            # Use explored_area instead of confidence_map > 0
            explored_mask = self.one_map.explored_area.astype(np.uint8)
            self.path = Planning.compute_to_goal(
                start,
                self.one_map.navigable_map,
                explored_mask,
                self.chosen_detection,
                self.obstcl_kernel_size,
                self.min_goal_dist
            )
            self.is_goal_path = True
            if self.path and len(self.path) > 0:
                if self.log:
                    # 경로 좌표를 [y, x]에서 [x, y]로 스왑
                    path_swapped = np.array(self.path)[:, [1, 0]]
                    rr.log("map/path", rr.LineStrips2D(
                        path_swapped,
                        colors=np.repeat(np.array([0, 255, 0])[np.newaxis, :], len(self.path), axis=0)
                    ))
    
    def _compute_best_path_legacy(self, start: np.ndarray) -> None:
        """
        Legacy path computation using nav_goals (fallback when no FrontierNodes available).
        :param start: start point as [X, Y]
        :return:
        """
        # If we still have no nav goals, we can't plan anything
        if len(self.nav_goals) == 0:
            return
        self.initializing = False
        self.is_goal_path = False
        self.path = None
        best_nav_goal = None  # Initialize to avoid undefined variable error
        while self.path is None and len(self.nav_goals) > 0:
            best_idx = None
            if len(self.nav_goals) == 1:
                top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[0].get_score()))
            else:
                top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[1].get_score()))

            # We have a frontier and we need to consider following up on that
            # Filter out Cluster types - they should not be considered as navigation goals
            valid_nav_goals = [nav_goal for nav_goal in self.nav_goals if not isinstance(nav_goal, Cluster)]
            if len(valid_nav_goals) == 0:
                # No valid nav goals (only clusters), remove all and reset
                # Remove all frontier nodes from pose graph
                for nav_goal in self.nav_goals:
                    if isinstance(nav_goal, Frontier):
                        frontier_key = tuple(nav_goal.frontier_midpoint)
                        if frontier_key in self.frontier_node_map:
                            frontier_node_id = self.frontier_node_map[frontier_key]
                            self.pose_graph._remove_frontier_node(frontier_node_id)
                            del self.frontier_node_map[frontier_key]
                self.nav_goals = []
                break
            
            curr_index = None
            if self.last_nav_goal is not None and not isinstance(self.last_nav_goal, Cluster):
                last_pt = self.last_nav_goal.get_descr_point()
                for nav_id in range(len(self.nav_goals)):
                    nav_goal = self.nav_goals[nav_id]
                    if isinstance(nav_goal, Cluster):
                        continue
                    if np.array_equal(last_pt, nav_goal.get_descr_point()):
                        # frontier still exists!
                        curr_index = nav_id
                        break
                if curr_index is None:
                    closest_index = closest_point_within_threshold(valid_nav_goals, last_pt,
                                                                   0.5 / self.one_map.cell_size)
                    if closest_index != -1:
                        # Map back to original index
                        valid_idx = 0
                        for nav_id in range(len(self.nav_goals)):
                            if not isinstance(self.nav_goals[nav_id], Cluster):
                                if valid_idx == closest_index:
                                    curr_index = nav_id
                                    break
                                valid_idx += 1
                        # there is a close point to the previous frontier that we could consider instead
                if curr_index is not None:
                    curr_value = self.nav_goals[curr_index].get_score()
                    if curr_value + 0.01 > self.last_nav_goal.get_score():
                        best_idx = curr_index
            if best_idx is None:
                # Select the current best nav_goal, and check for cyclic (excluding clusters)
                for nav_id in range(len(self.nav_goals)):
                    nav_goal = self.nav_goals[nav_id]
                    if isinstance(nav_goal, Cluster):
                        continue
                    cyclic = self.cyclic_checker.check_cyclic(start, nav_goal.get_descr_point(), top_two_vals)
                    if cyclic:
                        continue
                    best_idx = nav_id
                    # rr.log("path_updates", rr.TextLog(f"Selected frontier or POI based on score {self.frontiers[best_idx, 2]}. Max score is {self.frontiers[0, 2]}"))

                    break
            
            # If no valid nav goal found, break the loop
            if best_idx is None:
                break
            
            # TODO We should check if the chosen waypoint is reachable via simple path planning!
            best_nav_goal = self.nav_goals[best_idx]
            # Double check that best_nav_goal is not a Cluster (should not happen, but safety check)
            if isinstance(best_nav_goal, Cluster):
                # Remove cluster and continue to next iteration
                self.nav_goals.pop(best_idx)
                continue
                
            self.cyclic_checker.add_state_action(start, best_nav_goal.get_descr_point(), top_two_vals)
            if isinstance(best_nav_goal, Frontier):
                # Use explored_area instead of confidence_map > 0
                explored_mask = self.one_map.explored_area.astype(np.uint8)
                self.path = Planning.compute_to_goal(start, self.one_map.navigable_map & explored_mask,
                                                     explored_mask,
                                                     best_nav_goal.get_descr_point(),
                                                     self.obstcl_kernel_size, 2)
            if self.path is None:
                # remove the nav goal from the list, we don't know how to reach it
                self.nav_goals.pop(best_idx)
        
        # Handle path planning results
        if self.path is None:
            if not self.initializing:
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Resetting checked map as no path found."))
                self.one_map.reset_checked_map()
            return  # No path found, exit early
        
        # Update last_nav_goal and stuck counter only if we have a valid path and nav goal
        if best_nav_goal is not None:
            if self.last_nav_goal is not None and not np.array_equal(self.last_nav_goal.get_descr_point(),
                                                                     best_nav_goal.get_descr_point()):
                self.stuck_at_nav_goal_counter = 0
            else:
                if self.last_pose is not None:
                    if self.path is not None:
                        if len(self.path) < 5 and self.last_pose[0] == start[0] and self.last_pose[1] == start[1]:
                            self.stuck_at_nav_goal_counter += 1
            if self.stuck_at_nav_goal_counter > 10:
                # We probably are trying to reach an unreachable goal, for instance a frontier to the void in habitat
                self.blacklisted_nav_goals.append(best_nav_goal.get_descr_point())
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Frontier at position {best_nav_goal.get_descr_point()[0]}"
                                                      f",{best_nav_goal.get_descr_point()[1]} invalid."))
            self.last_nav_goal = best_nav_goal

        if self.path:
            if self.log:
                rr.log("path_updates", rr.TextLog(f"Computed path of length {len(self.path)}"))
                # 경로 좌표를 [y, x]에서 [x, y]로 스왑
                path_swapped = np.array(self.path)[:, [1, 0]]
                rr.log("map/path", rr.LineStrips2D(path_swapped, colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                                                                               len(self.path), axis=0)))
        # Note: Object detection path handling is done in compute_best_path(), not here
        # This function is only for exploration mode (frontier navigation)

    def compute_frontiers_and_POIs(self, px, py):
        """
        Computes the frontiers (VLFM style: at the border from explored to unexplored),
        and points of interest (high similarity regions within the fully explored, but not checked map)
        :return:
        """
        # Store current frontier midpoints before clearing nav_goals
        current_frontier_midpoints = {tuple(nav_goal.frontier_midpoint) for nav_goal in self.nav_goals if isinstance(nav_goal, Frontier)}
        
        # Remove frontier nodes from pose graph for frontiers that are no longer in nav_goals
        for frontier_key in list(self.frontier_node_map.keys()):
            if frontier_key not in current_frontier_midpoints:
                frontier_node_id = self.frontier_node_map[frontier_key]
                self.pose_graph._remove_frontier_node(frontier_node_id)
                del self.frontier_node_map[frontier_key]
        
        self.nav_goals = []
        # Compute the frontiers using VLFM classical definition
        # explored_area vs unexplored (no longer using confidence_map)
        
        # Debug: Check map state before frontier detection
        navigable_sum = self.one_map.navigable_map.sum()
        explored_sum = self.one_map.explored_area.sum()
        blacklist_count = len(self.blacklisted_nav_goals)
        
        frontiers, unexplored_map, largest_contour = detect_frontiers(
            self.one_map.navigable_map.astype(np.uint8).copy(),  # Ensure copy to avoid modification
            self.one_map.explored_area.astype(np.uint8).copy(),  # Ensure copy to avoid modification
            None,  # known_th parameter is not used in VLFM style
            int(1.0 * ((
                               self.one_map.n_cells /
                               self.one_map.size) ** 2)))
        
        # Debug: Log frontier detection results
        print(f"[DEBUG] compute_frontiers_and_POIs: query={self.query_text}, navigable={navigable_sum}, explored={explored_sum}, frontiers_detected={len(frontiers)}, blacklist={blacklist_count}")

        # Note: Points of interest (POIs) are no longer computed.
        # Frontier selection is now based on pose-graph FrontierNode similarity scores.

        if self.log:
            log_map_rerun(unexplored_map, path="map/unexplored")

        frontiers = [f[:, :, ::-1] for f in frontiers]  # need to flip coords for some reason

        # set the score of the fully explored map to 0 for the frontiers
        valid_frontiers_mask = np.zeros((len(frontiers),), dtype=bool)

        for i_frontier, frontier in enumerate(frontiers):
                frontier_mp = get_frontier_midpoint(frontier).astype(np.uint32)
                # Use explored_area instead of confidence_map for reachable area score
                # Since we don't have similarity map anymore, use a simple score based on frontier size
                explored_mask = self.one_map.explored_area.astype(np.uint8)
                # Simple score: use frontier size as score (can be improved later)
                score = len(frontier) * 0.1  # Simple heuristic score
                n_els = len(frontier)
                best_reachable = frontier_mp
                reachable_area = explored_mask
                frontier_mp = np.round(frontier_mp)
                if len(self.blacklisted_nav_goals) == 0 or not np.any(
                        np.all(frontier_mp == self.blacklisted_nav_goals, axis=1)):
                    valid_frontiers_mask[i_frontier] = True
                    frontier_obj = Frontier(frontier_midpoint=frontier_mp, points=frontier, frontier_score=score)
                    self.nav_goals.append(frontier_obj)
                    
                    # Add frontier to pose graph if not already exists
                    frontier_key = tuple(frontier_mp)
                    if frontier_key not in self.frontier_node_map:
                        # Get current pose ID (the pose that first discovered this frontier)
                        current_pose_id = self.pose_graph.pose_ids[-1] if self.pose_graph.pose_ids else None
                        if current_pose_id:
                            # Convert pixel coordinates to world coordinates
                            x_world, y_world = self.one_map.px_to_metric(frontier_mp[0], frontier_mp[1])
                            position_w = np.array([x_world, y_world, 0.0])
                            
                            # Extract image feature at frontier location
                            coarse_embedding = None
                            if (self.current_image_features is not None and 
                                self.current_depth is not None and 
                                self.current_odometry is not None):
                                # Get current pose for yaw
                                current_pose = self.pose_graph.nodes[current_pose_id]
                                assert isinstance(current_pose, PoseNode)
                                yaw = current_pose.theta
                                
                                # Convert world position to camera pixel coordinates
                                pixel_coords = self._world_to_camera_pixel(
                                    position_w,
                                    self.current_depth,
                                    yaw,
                                    self.current_odometry
                                )
                                
                                if pixel_coords is not None:
                                    pixel_x, pixel_y = pixel_coords
                                    # Extract feature at this pixel location
                                    coarse_embedding = self._extract_image_feature_at_pixel(
                                        self.current_image_features,
                                        pixel_x,
                                        pixel_y
                                    )
                            
                            # Add frontier node to pose graph
                            fr_node = self.pose_graph.add_frontier_node(
                                pose_id=current_pose_id,
                                position_w=position_w,
                                coarse_embedding=coarse_embedding,
                                semantic_hint=None,
                            )
                            # Store mapping from frontier midpoint to node ID
                            self.frontier_node_map[frontier_key] = fr_node.id

        self.nav_goals = sorted(self.nav_goals, key=lambda x: x.get_score(), reverse=True)

    def add_data(self,
                 image: np.ndarray,
                 depth: np.ndarray,
                 odometry: np.ndarray,
                 ) -> bool:
        """
        Adds data to the navigator
        :param image: RGB image of dimension [C, H, W]
        :param depth: depth image of dimension [H, W]
        :param odometry: 4x4 transformation matrix from camera to world
        :return: boolean indicating if the episode is over
        """
        odometry = odometry.astype(np.float32)
        x = odometry[0, 3]
        y = odometry[1, 3]
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])
        self.pose_graph.add_pose(x, y, yaw)
        if self.log:
            self.pose_graph.log_to_rerun(self.one_map)

        px, py = self.one_map.metric_to_px(x, y)
        if self.last_pose:
            if np.linalg.norm(np.array([px, py, yaw]) - np.array(self.last_pose)) < 0.01:
                if self.path is not None:
                    self.stuck_at_cell_counter += 1
            else:
                self.stuck_at_cell_counter = 0
        if self.stuck_at_cell_counter > 5:
            # we are stuck we need to add an obstacle right in front of us!
            dx = np.cos(yaw)
            dy = np.sin(yaw)

            # Round to nearest integer to get facing direction
            facing_dx = round(dx)
            facing_dy = round(dy)

            # Calculate coordinates of facing cell
            facing_px = px + facing_dx
            facing_py = py + facing_dy
            self.artificial_obstacles.append((facing_px, facing_py))

        if not self.one_map.camera_initialized:
            warnings.warn("Camera matrix not set, please set camera matrix first")
            return

        # Prepare RGB image (H, W, C) format
        rgb_image = image.transpose(1, 2, 0)
        sam_image_set = False

        # Single YOLO detection for both target object and pose graph registration
        # detect_all() returns all COCO classes with class_names
        all_detections = self.detector.detect_all(rgb_image, confidence_threshold=0.8)
        
        # Map COCO class names to standardized names using class_map
        # e.g., "tv_monitor" -> "tv", "couch" -> "couch"
        if "class_names" in all_detections:
            mapped_class_names = []
            for class_name in all_detections["class_names"]:
                # Map COCO class to standardized name
                mapped_name = self.class_map.get(class_name.lower(), class_name.lower())
                mapped_class_names.append(mapped_name)
            all_detections["class_names"] = mapped_class_names
        
        # Filter for target object detection (for navigation)
        target_class = self.query_text[0].lower() if self.query_text else None
        detections = {"boxes": [], "scores": []}
        if target_class and "class_names" in all_detections:
            # Map target class to standardized name
            target_mapped = self.class_map.get(target_class, target_class)
            for box, score, class_name in zip(all_detections["boxes"], 
                                               all_detections["scores"], 
                                               all_detections["class_names"]):
                # Match if mapped class_name matches target (both are now standardized)
                if class_name.lower() == target_mapped or class_name.lower() == target_class:
                    detections["boxes"].append(box)
                    detections["scores"].append(score)
        
        # Use all detections for pose graph registration
        coco_detections = all_detections
        if True:  # Always run pose graph registration
            current_pose_id = self.pose_graph.pose_ids[-1] if self.pose_graph.pose_ids else None
            mask_overlay_ids = np.zeros(depth.shape, dtype=np.uint16) if self.log else None
            mask_annotation_infos: List[rr.AnnotationInfo] = []
            
            # Log RGB image and COCO detections to Rerun
            if self.log:
                # Log RGB image (convert from [C, H, W] to [H, W, C])
                rgb_for_logging = rgb_image
                # Ensure image is uint8 and in correct format
                if rgb_for_logging.dtype != np.uint8:
                    rgb_for_logging = (rgb_for_logging * 255).astype(np.uint8) if rgb_for_logging.max() <= 1.0 else rgb_for_logging.astype(np.uint8)
                
                # Log image first (under camera/ to match blueprint)
                rr.log("camera/rgb", rr.Image(rgb_for_logging).compress(jpeg_quality=85))
                
                # Log COCO detection boxes if any (same path as image for overlay)
                num_boxes = len(coco_detections.get("boxes", []))
                if num_boxes > 0:
                    boxes = np.array(coco_detections["boxes"], dtype=np.float32)
                    # Ensure boxes are in correct format [x1, y1, x2, y2]
                    if len(boxes.shape) == 2 and boxes.shape[1] == 4:
                        labels = [f"{name} {score:.2f}" for name, score in 
                                 zip(coco_detections["class_names"], coco_detections["scores"])]
                        # Log boxes to same path as image (Rerun will overlay them)
                        rr.log("camera/rgb", 
                               rr.Boxes2D(
                                   array_format=rr.Box2DFormat.XYXY,
                                   array=boxes,
                                   labels=labels
                               ))
                        # Debug: print detection info
                        if self.pose_graph._step_counter % 10 == 0:
                            print(f"[COCO] Step {self.pose_graph._step_counter}: {num_boxes} detections logged to camera/rgb")
            
            if current_pose_id and len(coco_detections.get("boxes", [])) > 0 and "class_names" in coco_detections:
                # Collect all valid observations first
                observations = []
                if not sam_image_set:
                    self.sam_predictor.set_image(rgb_image)
                    sam_image_set = True

                for det_idx, (box, score, class_name) in enumerate(zip(coco_detections["boxes"],
                                                                        coco_detections["scores"],
                                                                        coco_detections["class_names"])):
                    position_w = None
                    pixel_center = None
                    try:
                        masks, _, _ = self.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=np.array(box)[None, :],
                            multimask_output=False,
                        )
                        mask_bool = masks[0].astype(bool)
                        mask_result = self._compute_world_center_from_mask(mask_bool, depth, yaw, odometry)
                        if mask_result is not None:
                            position_w, pixel_center = mask_result
                            if self.log:
                                if mask_overlay_ids is not None:
                                    class_idx = len(mask_annotation_infos) + 1
                                    mask_overlay_ids[mask_bool] = class_idx
                                    mask_annotation_infos.append(
                                        rr.AnnotationInfo(
                                            id=class_idx,
                                            label=f"{class_name} ({score:.2f})",
                                            color=[0, 255, 0, 120],
                                        )
                                    )
                                rr.log("camera/rgb",
                                       rr.Points2D(
                                           np.array([[pixel_center[0], pixel_center[1]]], dtype=np.float32),
                                           colors=[[0, 255, 0]],
                                           radii=[3],
                                       ))
                    except Exception as e:
                        if self.pose_graph._step_counter % 100 == 0:
                            print(f"[COCO] SAM segmentation failed: {e}")

                    if position_w is None:
                        # Fallback to bounding-box center depth projection
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        
                        if 0 <= center_y < depth.shape[0] and 0 <= center_x < depth.shape[1]:
                            obj_depth = depth[center_y, center_x]
                            if 0 < obj_depth < 5.0:
                                y_world = -(center_x - self.one_map.cx) * obj_depth / self.one_map.fx
                                x_world = obj_depth
                                r = np.array([[np.cos(yaw), -np.sin(yaw)],
                                              [np.sin(yaw), np.cos(yaw)]])
                                x_rot, y_rot = np.dot(r, np.array([x_world, y_world]))
                                x_world_final = x_rot + odometry[0, 3]
                                y_world_final = y_rot + odometry[1, 3]
                                position_w = np.array([x_world_final, y_world_final, 0.0])

                    if position_w is not None:
                        observations.append({
                            "label": class_name,
                            "position_w": position_w,
                            "confidence": float(score),
                            "embedding": None,  # Can add CLIP embedding later if needed
                        })
                
                # Process all observations in batch
                if observations:
                    if self.log and mask_overlay_ids is not None and mask_overlay_ids.any():
                        if mask_annotation_infos:
                            rr.log("camera/rgb/mask_annotations", rr.AnnotationContext(mask_annotation_infos))
                        rr.log("camera/rgb/masks", rr.SegmentationImage(mask_overlay_ids))
                    
                    self.pose_graph.add_object_nodes_batch(
                        pose_id=current_pose_id,
                        observations=observations,
                        step=self.pose_graph._step_counter,
                        distance_threshold=1.0,
                        use_kalman=True,
                        mahalanobis_threshold=3.0,
                    )
        a = time.time()
        image_features = self.model.get_image_features(image[np.newaxis, ...]).squeeze(0)
        b = time.time()
        
        # Store current frame data for frontier embedding extraction
        self.current_image_features = image_features
        self.current_image = image
        self.current_depth = depth
        self.current_odometry = odometry
        
        self.one_map.update(image_features, depth, odometry, self.artificial_obstacles)
        detected_just_now = False
        start = np.array([px, py])
        old_path = self.path.copy() if self.path else self.path
        old_id = self.path_id
        if self.first_obs:
            # Mark initial area as explored and checked
            # Use explored_area and checked_conf_map directly
            self.one_map.explored_area[px - 10:px + 10, py - 10:py + 10] = True
            self.one_map.checked_conf_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.first_obs = False
        # if detections.class_id.shape[0] > 0:
        last_saw_left = self.saw_left
        last_saw_right = self.saw_right
        self.saw_left = False
        self.saw_right = False
        
        # ===== 그래프 기반 목표 객체 탐색 (YOLO 실시간 감지 전에 먼저 시도) =====
        if not self.object_detected:
            graph_result = self.find_target_object_in_graph(x, y)
            
            if graph_result is not None:
                target_px, target_obj = graph_result
                
                # 경로가 이미 계산됨 (find_target_object_in_graph에서 검증)
                self.object_detected = True
                self.chosen_detection = (target_px[0], target_px[1])
                self.target_object_node = target_obj
                
                # 경로 계획
                explored_mask = self.one_map.explored_area.astype(np.uint8)
                self.path = Planning.compute_to_goal(
                    start,
                    self.one_map.navigable_map,
                    explored_mask,
                    target_px,
                    self.obstcl_kernel_size,
                    self.min_goal_dist
                )
                self.is_goal_path = True
                
                if self.log:
                    rr.log("path_updates", rr.TextLog(
                        f"[Graph] Found '{target_obj.label}' in graph at "
                        f"({target_obj.position[0]:.2f}, {target_obj.position[1]:.2f}), "
                        f"confidence={target_obj.confidence:.2f}, observations={target_obj.num_observations}"
                    ))
                    # 목표 위치 좌표를 [x, y]에서 [y, x]로 스왑
                    goal_swapped = [self.chosen_detection[1], self.chosen_detection[0]]
                    rr.log("map/goal_pos",
                           rr.Points2D([goal_swapped], colors=[[0, 255, 0]], radii=[2]))
                    if self.path:
                        path_swapped = np.array(self.path)[:, [1, 0]]
                        rr.log("map/path", rr.LineStrips2D(
                            path_swapped,
                            colors=np.repeat(np.array([0, 255, 0])[np.newaxis, :], len(self.path), axis=0)
                        ))
        
        # ===== YOLO 실시간 감지 (그래프에서 목표를 찾지 못한 경우 폴백) =====
        if len(detections["boxes"]) > 0 and not self.object_detected:
            print(f"[DEBUG] add_data: Object detection triggered for query={self.query_text}, boxes={len(detections['boxes'])}")
            # wants rgb
            if not sam_image_set:
                self.sam_predictor.set_image(rgb_image)
                sam_image_set = True
            for area, confidence in zip(detections["boxes"], detections['scores']):
                if self.log:
                    rr.log("camera/detection", rr.Boxes2D(array_format=rr.Box2DFormat.XYXY, array=area))
                    rr.log("object_detections", rr.TextLog(f"Object {self.query_text[0]} detected"))

                # TODO Find free point in front of object
                chosen_detection = (
                    int((area[3] + area[1]) / 2), int((area[2] + area[0]) / 2))
                masks, _, _ = self.sam_predictor.predict(point_coords=None,
                                                         point_labels=None,
                                                         box=np.array(area)[None, :],
                                                         multimask_output=False, )
                # Project the points where the mask is one
                mask_ids = np.argwhere(masks[0] & (depth != 0))
                depth_detection = depth[chosen_detection[0], chosen_detection[1]]

                depths = depth[mask_ids[:, 0], mask_ids[:, 1]]

                if not self.filter_detections_depth or depth_detection < 2.5:
                    y_world = -(mask_ids[:, 1] - self.one_map.cx) * depths / self.one_map.fx
                    x_world = depths
                    r = np.array([[np.cos(yaw), -np.sin(yaw)],
                                  [np.sin(yaw), np.cos(yaw)]])
                    x_rot, y_rot = np.dot(r, np.stack((x_world, y_world)))
                    x_rot += odometry[0, 3]
                    y_rot += odometry[1, 3]

                    x_id = ((x_rot / self.one_map.cell_size)).astype(np.uint32) + \
                           self.one_map.map_center_cells[0].item()
                    y_id = ((y_rot / self.one_map.cell_size)).astype(np.uint32) + \
                           self.one_map.map_center_cells[1].item()

                    object_valid = True
                    if self.log:
                        rr.log("map/proj_detect",
                               rr.Points2D(np.stack((x_id, y_id)).T, colors=[[0, 0, 255]], radii=[1]))
                        # log the segmentation mask as rgba
                        rr.log("camera", rr.SegmentationImage(masks[0].astype(np.uint8))
                               )
                    # Consensus filtering is disabled since we don't have similarity map anymore
                    # Simply use the closest valid detection point
                    if object_valid:
                        best = np.argmin(depths)
                        if object_valid:
                            self.chosen_detection = (x_id[best], y_id[best])
                    if object_valid:
                        self.object_detected = True
                        self.compute_best_path(start)
                        if not self.path:
                            self.object_detected = False
                            self.path = old_path
                            self.path_id = old_id
                        else:
                            if self.log:
                                rr.log("path_updates",
                                       rr.TextLog(f"The object {self.query_text[0]} has been detected just now."))
                                # 목표 위치 좌표를 [x, y]에서 [y, x]로 스왑 (chosen_detection은 (x,y) 형식)
                                goal_swapped = [self.chosen_detection[1], self.chosen_detection[0]]
                                rr.log("map/goal_pos",
                                       rr.Points2D([goal_swapped], colors=[[0, 255, 0]], radii=[1]))
        elif not self.object_detected:
            self.chosen_detection = None
            self.object_detected = False
        if not self.object_detected:
            if self.saw_left:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "L")
            elif self.saw_right:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "R")
        self.compute_frontiers_and_POIs(*self.one_map.metric_to_px(odometry[0, 3], odometry[1, 3]))
        e = time.time()
        # Similarity map logging is disabled since we don't have feature_map/confidence_map anymore
        # Compute the new path
        # TODO Make the thresholds and distances to object a parameter
        if self.object_detected:
            if np.linalg.norm(start - self.chosen_detection) <= self.max_detect_distance:
                # Object reached: reset detection state and signal success
                if self.log and self.target_object_node:
                    rr.log("path_updates", rr.TextLog(
                        f"[Reached] Target '{self.target_object_node.label}' reached!"
                    ))
                self.object_detected = False
                self.chosen_detection = None
                self.target_object_node = None  # Reset target object node
                self.path = None  # Clear path to allow new exploration
                return True
            # Consensus filtering is disabled since we don't have similarity map anymore
        if self.allow_replan:
            self.compute_best_path(start)
        if self.object_detected and self.path is not None and len(self.path) < 3:
            # Path too short (likely reached object): reset detection state and signal success
            if self.log and self.target_object_node:
                rr.log("path_updates", rr.TextLog(
                    f"[Reached] Target '{self.target_object_node.label}' reached (short path)!"
                ))
            self.object_detected = False
            self.chosen_detection = None
            self.target_object_node = None  # Reset target object node
            self.path = None  # Clear path to allow new exploration
            return True
        self.last_pose = (px, py, yaw)

    def get_pose_graph_statistics(self) -> dict:
        """Get pose graph statistics for monitoring."""
        return self.pose_graph.get_statistics()

    def _world_to_camera_pixel(
        self,
        world_pos: np.ndarray,
        depth: np.ndarray,
        yaw: float,
        odometry: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates to camera pixel coordinates.
        Uses the same coordinate system as project_depth_camera:
        - Camera frame: x is depth (forward), y is horizontal (left), z is vertical (up)
        - Inverse of: pixel -> camera -> world transformation
        
        Args:
            world_pos: World position (3,) [x, y, z] (z is typically 0 for ground plane)
            depth: Depth image (H, W) - used for bounds checking
            yaw: Current yaw angle (rotation around z-axis)
            odometry: 4x4 transformation matrix from camera to world
            
        Returns:
            Tuple of (pixel_x, pixel_y) if valid, None otherwise
        """
        # Convert world position to camera-relative position
        world_x = world_pos[0] - odometry[0, 3]
        world_y = world_pos[1] - odometry[1, 3]
        world_z = world_pos[2] - odometry[2, 3] if len(world_pos) > 2 else 0.0
        
        # Rotate by -yaw (inverse rotation to get camera-local coordinates)
        # This matches the rotation in rotate_pcl function
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        r_inv = np.array([[cos_yaw, -sin_yaw],
                          [sin_yaw, cos_yaw]])
        
        # Apply rotation to get camera-local 2D coordinates
        cam_local_2d = np.dot(r_inv, np.array([world_x, world_y]))
        
        # Camera coordinate system (from project_depth_camera):
        # x_cam = depth (forward into image)
        # y_cam = horizontal (left)
        # z_cam = vertical (up)
        x_cam = np.sqrt(cam_local_2d[0]**2 + cam_local_2d[1]**2)  # depth
        y_cam = -cam_local_2d[1]  # horizontal (left is positive in camera frame)
        z_cam = -world_z  # vertical (up is positive in camera frame, but we use -world_z)
        
        # Check if depth is valid
        if x_cam <= 0 or x_cam > 10.0:
            return None
        
        # Project to pixel coordinates (inverse of project_depth_camera)
        # From project_depth_camera: x_world = (xx - cx) * zz / fx, y_world = (yy - cy) * zz / fy
        # Inverse: pixel_x = -y_cam * fx / x_cam + cx, pixel_y = -z_cam * fy / x_cam + cy
        pixel_x = int(-y_cam * self.one_map.fx / x_cam + self.one_map.cx)
        pixel_y = int(-z_cam * self.one_map.fy / x_cam + self.one_map.cy)
        
        # Check if within image bounds
        if 0 <= pixel_x < depth.shape[1] and 0 <= pixel_y < depth.shape[0]:
            return (pixel_x, pixel_y)
        return None
    
    def _extract_image_feature_at_pixel(
        self,
        image_features: torch.Tensor,
        pixel_x: int,
        pixel_y: int,
    ) -> Optional[np.ndarray]:
        """
        Extract image feature at specific pixel location from dense feature map.
        
        Args:
            image_features: Dense image features [F, Hf, Wf]
            pixel_x: Pixel x coordinate in original image
            pixel_y: Pixel y coordinate in original image
            
        Returns:
            Feature vector [F] if valid, None otherwise
        """
        # Get feature map dimensions
        feat_h, feat_w = image_features.shape[1], image_features.shape[2]
        
        # Scale pixel coordinates to feature map coordinates
        # Assuming image_features are from CLIP dense model
        # Need to know original image size - use current_image if available
        if self.current_image is not None:
            img_h, img_w = self.current_image.shape[1], self.current_image.shape[2]
            feat_x = int(pixel_x * feat_w / img_w)
            feat_y = int(pixel_y * feat_h / img_h)
        else:
            # Fallback: assume feature map is same size as depth
            if self.current_depth is not None:
                img_h, img_w = self.current_depth.shape[0], self.current_depth.shape[1]
                feat_x = int(pixel_x * feat_w / img_w)
                feat_y = int(pixel_y * feat_h / img_h)
            else:
                return None
        
        # Check bounds
        if 0 <= feat_x < feat_w and 0 <= feat_y < feat_h:
            feature = image_features[:, feat_y, feat_x].cpu().numpy()
            return feature
        return None

    def _compute_world_center_from_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        yaw: float,
        odometry: np.ndarray,
        max_depth: float = 5.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Project a SAM mask to world coordinates and return the centroid.
        
        Returns:
            Tuple of (world_center (3,), pixel_center (2,)) if successful, otherwise None.
        """
        valid_mask = mask & (depth > 0) & (depth < max_depth)
        if not np.any(valid_mask):
            return None
        
        indices = np.argwhere(valid_mask)
        depth_vals = depth[valid_mask].astype(np.float32)
        pixel_x = indices[:, 1].astype(np.float32)
        pixel_y = indices[:, 0].astype(np.float32)
        
        y_cam = -(pixel_x - self.one_map.cx) * depth_vals / self.one_map.fx
        x_cam = depth_vals
        
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
        x_rot, y_rot = np.dot(rot, np.vstack((x_cam, y_cam)))
        x_world = x_rot + odometry[0, 3]
        y_world = y_rot + odometry[1, 3]
        
        center_world = np.array([np.mean(x_world), np.mean(y_world), 0.0], dtype=np.float32)
        center_pixel = np.array([np.mean(pixel_x), np.mean(pixel_y)], dtype=np.float32)
        return center_world, center_pixel

    def export_pose_graph(self, filepath: str) -> None:
        """Export pose graph to file."""
        self.pose_graph.export_to_file(filepath)

    def check_loop_closure(self, current_node_id: int) -> Optional[int]:
        """Simple loop closure detection based on spatial proximity."""
        candidates = self.pose_graph.find_loop_closure_candidates(
            current_node_id, distance_threshold=0.5
        )
        
        if candidates:
            # For now, just return the closest candidate
            # In a real implementation, you'd use visual features for verification
            return candidates[0]
        return None

    def get_map(self,
                return_map=True
                ) -> Optional[np.ndarray]:
        """
        Legacy method - no longer computes pixel-level similarity map.
        Kept for compatibility but returns None.
        Similarity is now computed at FrontierNode level in compute_best_path().
        """
        # This method is deprecated - similarity is now computed at FrontierNode level
        # Return None to maintain compatibility
        return None

    def get_confidence_map(self,
                           ) -> np.ndarray:
        """
          Legacy method - confidence_map no longer exists.
          Returns explored_area as a replacement.
          :return: explored_area as numpy array
          """
        # Return explored_area as a replacement for confidence_map
        return self.one_map.explored_area.astype(np.float32)


if __name__ == "__main__":
    from vision_models.clip_dense import ClipModel
    # from vision_models.yolov7_model import YOLOv7Detector
    from vision_models.trt_yolo_world_detector import TRTYOLOWorldDetector
    import matplotlib.pyplot as plt
    import cv2

    # Yolo World
    # from ultralytics import YOLOWorld, YOLO

    # yw_detector = YOLO("yolov8s-worldv2.engine")

    print("Am I doing this?")

    camera_matrix = np.array([[384.41534423828125, 0.0, 328.7698059082031],
                              [0.0, 384.0389404296875, 245.87942504882812],
                              [0.0, 0.0, 1.0]])
    rgb = cv2.imread("/home/spot/Finn/MON/test_images/pairs/rgb_1.png")
    rgb = rgb[:, :, ::-1]
    rgb = cv2.resize(rgb, (640, 480)).transpose(2, 0, 1)
    depth = cv2.imread("/home/spot/Finn/MON/test_images/pairs/depth_1.png")
    depth = depth.astype(np.float32) / 255.0 * 3.0
    depth = cv2.resize(depth, (640, 480))[:, :, 0]
    odom = np.eye(4)
    # entire forward pass test
    base_conf = load_config()
    mapper = Navigator(ClipModel("", True), TRTYOLOWorldDetector(base_conf.Conf.planner.yolo_confidence), base_conf.Conf)
    mapper.set_camera_matrix(camera_matrix)
    mapper.set_query(["A fridge"])
    mapper.add_data(rgb, depth, odom)
    # test image, depth, odometry
    a = time.time()
    for i in range(10):
        mapper.add_data(rgb, depth, odom)
        # yw_detections = yw_detector(rgb.transpose(1, 2, 0))
        # yw_detections[0].show()

    print(f"Entire map update: {(time.time() - a) / 10}")
    # get_map() no longer returns similarity map (returns None)
    # Use explored_area for visualization instead
    explored_area = mapper.get_confidence_map()  # Returns explored_area
    plt.imshow(explored_area)
    plt.savefig("firstmap.png")
    plt.show()
