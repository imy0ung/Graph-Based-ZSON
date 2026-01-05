"""
This is the core mapping module, which contains the OneMap class.
"""
from transforms3d.derivations.angle_axes import point

from mapping import (precompute_gaussian_kernel_components,
                     precompute_gaussian_sum_els, gaussian_kernel_sum,
                     compute_gaussian_kernel_components,
                     detect_frontiers,
                     )
from config import MappingConf

from onemap_utils import ceildiv

import time

# enum
from enum import Enum

# NumPy
import numpy as np

# typing
from typing import Tuple, List, Optional

# rerun
import rerun as rr

# torch
import torch

# warnings
import warnings

# cv2
import cv2

# functools
from functools import wraps


def rotate_pcl(
        pointcloud: torch.Tensor,
        tf_camera_to_episodic: torch.Tensor,
) -> torch.Tensor:
    # TODO We might be interested in a complete 3d rotation if the camera is not perfectly horizontal
    rotation_matrix = tf_camera_to_episodic[:3, :3]

    yaw = torch.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # print(yaw)
    r = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]], dtype=torch.float32).to("cuda")
    pointcloud[:, :2] = (r @ pointcloud[:, :2].T).T
    return pointcloud

def print_memory_stats(label):
    print(f"\n--- Memory Stats for {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

class DenseProjectionType(Enum):
    INTERPOLATE = "interpolate"
    SUBSAMPLE = "subsample"


class FusionType(Enum):
    EMA = "EMA"
    SPATIAL = "Spatial"


class OneMap:
    obstacle_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # obstacle likelihood
    navigable_map: np.ndarray  # binary traversability map where first dimension is x direction, second dimension is y
    # navigable likelihood
    fully_explored_map: np.ndarray  # binary explored map where first dimension is x direction, second dimension is y
    explored_area: np.ndarray  # binary explored area map (VLFM style: any observation counts as explored)
    checked_map: np.ndarray  # binary checked map where first dimension is x direction, second dimension is y,
    # can be reset
    checked_conf_map: torch.Tensor

    def __init__(self,
                 feature_dim: int,
                 config: MappingConf,
                 dense_projection: DenseProjectionType = DenseProjectionType.INTERPOLATE,
                 fusion_type: FusionType = FusionType.EMA,
                 map_device: str = "cuda",
                 ) -> None:
        """

        :param feature_dim: The dimension of the feature space
        :param n_cells: The number of cells in the x and y direction respectively
        :param size: The size of the map in meters
        :param dense_projection: The type of dense projection to use, must be one of DenseProjectionType
        :param fusion_type: The type of fusion to use, must be one of FusionType
        """
        assert isinstance(dense_projection,
                          DenseProjectionType), "Invalid dense_projection. It should be one of DenseProjection."
        assert isinstance(fusion_type, FusionType), "Invalid fusion_type. It should be one of FusionType."

        self.dense_projection = dense_projection
        self.fusion_type = fusion_type
        self.map_device = map_device

        self.n_cells = config.n_points
        self.map_center_cells = self.map_center_cells = torch.tensor([self.n_cells // 2, self.n_cells // 2],
                                                                     dtype=torch.int32).to("cuda")
        self.size = config.size
        self.cell_size = self.size / self.n_cells
        self.feature_dim = feature_dim  # Keep for compatibility, but not used for storage

        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)
        # 누적 신뢰도를 별도로 유지해 가중 평균을 수행
        self.obstacle_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)
        self.agent_radius = config.agent_radius
        col_kernel_size = self.n_cells / self.size * self.agent_radius
        col_kernel_size = int(col_kernel_size) + (int(col_kernel_size) % 2 == 0)
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.navigable_kernel = np.ones((col_kernel_size, col_kernel_size), np.uint8)

        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.explored_area = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        # Depth-based frontier strategy: track unexplored area in FOV (1.8m~10m range)
        self.unexplored_fov_area = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        # Current frame's far observed area (1.8m~10m) for visualization
        self.current_far_observed_area = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.checked_conf_map = self.checked_conf_map.to(self.map_device)
        
        # D435 depth camera thresholds for frontier detection
        self.depth_explored_threshold = 4.5  # meters - explored range
        self.depth_unexplored_max = 10.0  # meters - unexplored range max

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_initialized = False
        self.agent_height_0 = None

        self.kernel_half = int(np.round(config.blur_kernel_size / self.cell_size))
        self.kernel_size = self.kernel_half * 2 + 1
        self.kernel_components_sum = precompute_gaussian_sum_els(self.kernel_size).to("cuda")
        self.kernel_components = precompute_gaussian_kernel_components(self.kernel_size).to("cuda")
        self.kernel_ids = torch.arange(-self.kernel_half, self.kernel_half + 1).to("cuda")
        self.kernel_ids_x, self.kernel_ids_y = torch.meshgrid(self.kernel_ids, self.kernel_ids)
        self.kernel_ids_x = self.kernel_ids_x.unsqueeze(0)
        self.kernel_ids_y = self.kernel_ids_y.unsqueeze(0)
        print("OneMap initialized. The map contains {} cells. Obstacle and navigable maps are maintained for navigation.".format(self.n_cells ** 2))

        self.obstacle_map_threshold = config.obstacle_map_threshold
        self.fully_explored_threshold = config.fully_explored_threshold
        self.checked_map_threshold = config.checked_map_threshold
        self.depth_factor = config.depth_factor
        self.gradient_factor = config.gradient_factor
        self.optimal_object_distance = config.optimal_object_distance
        self.optimal_object_factor = config.optimal_object_factor
        self.obstacle_min = config.obstacle_min
        self.obstacle_max = config.obstacle_max
        self.filter_stairs = config.filter_stairs
        self.floor_threshold = config.floor_threshold
        self.floor_level = config.floor_level

        self._iters = 0

    def reset(self):
        # Reset obstacle map
        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)
        self.obstacle_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset navigable map
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset fully explored map
        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset explored area (VLFM style)
        self.explored_area = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset checked map
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset checked confidence map
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset iteration counter
        self._iters = 0
        self.agent_height_0 = None


    def reset_checked_map(self):
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        """
        Sets the camera matrix for the map
        :param camera_matrix: 3x3 numpy array representing the camera matrix
        :return:
        """
        self.camera_initialized = True
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    def update(self,
               values: torch.Tensor,
               depth: np.ndarray,
               tf_camera_to_episodic: np.ndarray,
               artifical_obstacles: Optional[List[Tuple[float]]] = None
               ) -> None:
        """
        Updates the map with values by projecting them into the map from depth
        :param values: torch tensor of values. Either a 3D array of shape (feature_dim, hf, wf)
                        or a 1D array of shape (feature_dim)
        :param depth:  numpy array of depth values of shape (h, w)
        :param tf_camera_to_episodic: 4x4 numpy array representing the transformation from camera to episodic
        """
        assert values.shape[0] == self.feature_dim
        if not self.camera_initialized:
            warnings.warn("Camera matrix must be set before updating the map")
            return
        if self.agent_height_0 is None:
            self.agent_height_0 = tf_camera_to_episodic[2, 3] / tf_camera_to_episodic[3, 3]
        if len(values.shape) == 1 or (values.shape[-1] == 1 and values.shape[-2] == 1):
            # project_single is not implemented, skip for now
            raise NotImplementedError("project_single is not implemented")
        elif len(values.shape) == 3:
            values = values.permute(1, 2, 0)  # feature_dim last for convenience
            (observed_cell_indices,
             obstacle_mapped, obstcl_confidence_mapped, far_cell_indices) = self.project_dense(values, torch.Tensor(depth).to("cuda"),
                                                                             torch.tensor(tf_camera_to_episodic),
                                                                             self.fx, self.fy,
                                                                             self.cx, self.cy)
        else:
            raise Exception("Provided Value observation of unsupported format")
        self.fuse_maps(observed_cell_indices, obstacle_mapped, obstcl_confidence_mapped, far_cell_indices, artifical_obstacles)

    def fuse_maps(self,
                  observed_cell_indices: torch.Tensor,
                  obstacle_mapped: torch.Tensor,
                  obstcl_confidence_mapped: torch.Tensor,
                  far_cell_indices: torch.Tensor,
                  artifical_obstacles: Optional[List[Tuple[float]]] = None
                  ) -> None:
        """
        Fuses the mapped observations into the map.
        Updates obstacle map and explored area based on observed cells.
        :param observed_cell_indices: torch tensor of shape (2, N) containing (x, y) indices of observed cells
        :param obstacle_mapped: torch sparse COO tensor of obstacle values
        :param obstcl_confidence_mapped: torch sparse COO tensor of obstacle confidences
        :param artifical_obstacles: Optional list of artificial obstacle positions
        :return:
        """
        if self.fusion_type == FusionType.EMA:
            indices_obstacle = obstacle_mapped.indices()
            
            # Update explored_area directly from observed cells (VLFM style: any observation counts as explored)
            if observed_cell_indices.shape[1] > 0:
                obs_indices_np = observed_cell_indices.cpu().numpy()
                # Ensure indices are within bounds and convert to integer type
                valid_mask = (obs_indices_np[0] >= 0) & (obs_indices_np[0] < self.n_cells) & \
                             (obs_indices_np[1] >= 0) & (obs_indices_np[1] < self.n_cells)
                if valid_mask.any():
                    valid_indices = obs_indices_np[:, valid_mask].astype(np.int32)  # Convert to integer type
                    valid_indices_torch = torch.from_numpy(valid_indices).to(self.map_device)
                    # Mark as explored
                    self.explored_area[valid_indices[0], valid_indices[1]] = True
                    # Update checked_conf_map (accumulate confidence for checked_map calculation)
                    # 관측 신뢰도를 누적 (현재 관측값에 대한 명시적 신뢰도가 없으므로 1.0 가중을 합산)
                    self.checked_conf_map[valid_indices_torch[0].long(), valid_indices_torch[1].long()] += 1.0

            # Obstacle Map update (신뢰도 누적 기반 가중 평균 융합)
            if indices_obstacle.shape[1] > 0:
                # 새 관측의 장애물 신뢰도와 값 추출
                # confs_new: project_dense에서 계산된 신뢰도 (depth, 거리, gradient 기반)
                # obstacle_values: 장애물 여부 (0 또는 1)
                confs_new = obstcl_confidence_mapped.values().data.squeeze().to(self.map_device)
                obstacle_values = obstacle_mapped.values().data.squeeze().to(self.map_device)

                # 텐서 차원 보장 (1D로 변환)
                if len(confs_new.shape) > 1:
                    confs_new = confs_new.squeeze()
                if len(obstacle_values.shape) > 1:
                    obstacle_values = obstacle_values.squeeze()

                # 장애물 셀 인덱스 추출
                indices_0 = indices_obstacle[0].long().to(self.map_device)
                indices_1 = indices_obstacle[1].long().to(self.map_device)

                # 신뢰도 누적 기반 가중 평균 계산
                # 기존 누적 신뢰도 조회
                confs_old = self.obstacle_conf_map[indices_0, indices_1]
                # 총 신뢰도 = 기존 신뢰도 + 새 신뢰도
                conf_den = confs_old + confs_new
                # 가중치 계산: 기존 값의 비중 = 기존 신뢰도 / 총 신뢰도
                weight_old = torch.nan_to_num(confs_old / conf_den, nan=0.0)
                # 가중치 계산: 새 값의 비중 = 새 신뢰도 / 총 신뢰도
                weight_new = torch.nan_to_num(confs_new / conf_den, nan=0.0)

                # 장애물 맵 업데이트: 가중 평균으로 융합
                # 관측이 많아질수록 기존 값의 비중이 커져 안정화됨
                self.obstacle_map[indices_0, indices_1] = (
                    self.obstacle_map[indices_0, indices_1] * weight_old +
                    obstacle_values * weight_new
                )
                # 누적 신뢰도 업데이트 (다음 관측을 위한 저장)
                self.obstacle_conf_map[indices_0, indices_1] = conf_den

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    self.occluded_map[obs[0], obs[1]] = True
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)

            # Only mark navigable areas as explored
            self.explored_area = self.explored_area & self.navigable_map
            
            # Update unexplored_fov_area: mark 1.8m~10m range cells as observed in FOV
            # These cells are removed from unexplored when they enter the robot's FOV
            # Also track current frame's far observed area for visualization
            self.current_far_observed_area.fill(False)  # Reset current frame
            if far_cell_indices.shape[1] > 0:
                far_indices_np = far_cell_indices.cpu().numpy()
                valid_mask_far = (far_indices_np[0] >= 0) & (far_indices_np[0] < self.n_cells) & \
                                (far_indices_np[1] >= 0) & (far_indices_np[1] < self.n_cells)
                if valid_mask_far.any():
                    valid_far_indices = far_indices_np[:, valid_mask_far].astype(np.int32)
                    # Mark as observed in FOV (these areas are no longer unexplored)
                    # Remove from unexplored_fov_area when they enter FOV
                    self.unexplored_fov_area[valid_far_indices[0], valid_far_indices[1]] = False
                    # Store current frame's far observed area for visualization
                    self.current_far_observed_area[valid_far_indices[0], valid_far_indices[1]] = True

            # Update fully_explored_map based on checked_map threshold
            # 신뢰도 기반 판정: 1 / conf < threshold 형태로 복원
            checked_conf_np = self.checked_conf_map.cpu().numpy()
            self.checked_map = (np.nan_to_num(1.0 / checked_conf_np) < self.checked_map_threshold)
            self.fully_explored_map = self.checked_map.copy()

    @torch.no_grad()
    # @torch.compile
    def project_dense(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: (observed_cell_indices, obstacle_mapped, obstcl_confidence_mapped)
                 observed_cell_indices: torch tensor of shape (2, N) with (x, y) indices of observed cells
                 obstacle_mapped: sparse COO tensor of obstacle values
                 obstcl_confidence_mapped: sparse COO tensor of obstacle confidences
        """
        # check if values is on cuda
        if not values.is_cuda:
            print("Warning: Provided value array is not on cuda, which it should be as an output of a model. Moving to "
                  "Cuda, which will slow things down.")
            values = values.to("cuda")
        if not depth.is_cuda:
            print(
                "Warning: Provided depth array is not on cuda, which it could be if is an output of a model. Moving to "
                "Cuda, which will slow things down.")
            depth = depth.to("cuda")

        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            depth_aligned = depth
        else:
            # our values are to be considered "patch wise" where we need to project each patch, by averaging the
            # depth values within that patch
            if self.dense_projection == DenseProjectionType.SUBSAMPLE:
                nh = values.shape[0]
                nw = values.shape[1]
                h = depth.shape[0]
                w = depth.shape[1]
                # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
                patch_size_h = ceildiv(h, nh)
                patch_size_w = ceildiv(w, nw)

                pad_h = patch_size_h * nh - h
                pad_w = patch_size_w * nw - w
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before

                depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
                depth_aligned = depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))
            elif self.dense_projection == DenseProjectionType.INTERPOLATE:
                values = torch.nn.functional.interpolate(values.permute(2, 0, 1).unsqueeze(0),
                                                         size=depth.shape,
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1, 2, 0)
                depth_aligned = depth
            else:
                raise Exception("Unsupported Dense Projection Mode.")

        # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
        depth_image_smoothed = depth_aligned

        mask = depth_image_smoothed == float('inf')
        depth_image_smoothed[mask] = depth_image_smoothed[~mask].max()
        kernel_size = 11
        pad = kernel_size // 2

        depth_image_smoothed = -torch.nn.functional.max_pool2d(-depth_image_smoothed.unsqueeze(0), kernel_size,
                                                               padding=pad,
                                                               stride=1).squeeze(0)
        # 깊이 기반 가중치만 사용 (gradient 계산 제거)
        scores = torch.exp(-((self.optimal_object_distance - depth) / self.optimal_object_factor) ** 2 / 3.0)
        scores_aligned = scores.reshape(-1)

        projected_depth, hole_mask = self.project_depth_camera(depth_aligned, (depth.shape[0], depth.shape[1]), fx,
                                                    fy, cx, cy)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        values_aligned = values.reshape((-1, values.shape[-1]))

        pcl_grid_ids = torch.floor(rotated_pcl[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # Filter valid updates
        depth_flat = depth_aligned.flatten()
        mask = (depth_flat != float('inf')) & (depth_flat != 0) & (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & (
                pcl_grid_ids[:, 0] < self.n_cells - self.kernel_half - 1) & (
                       pcl_grid_ids[:, 1] >= self.kernel_half + 1) & (
                       pcl_grid_ids[:, 1] < self.n_cells - self.kernel_half - 1)  # for value map
        
        # Depth-based frontier strategy: separate explored (<=1.8m) and far observed (1.8m~10m) areas
        mask_explored = mask & (depth_flat <= self.depth_explored_threshold)  # 1.8m 이내: 탐색된 영역
        mask_far_observed = mask & (depth_flat > self.depth_explored_threshold) & (depth_flat <= self.depth_unexplored_max)  # 1.8m~10m: 미탐색 영역 (시야각 내)
        
        if hole_mask.nelement() == 0:
            mask_obstacle = mask_explored & (((rotated_pcl[:, 2]> self.obstacle_min) & (
                                         rotated_pcl[:, 2]  < self.obstacle_max)) )
        else:
            mask_obstacle = mask_explored & (((rotated_pcl[:, 2] > self.obstacle_min) & (
                    rotated_pcl[:, 2] < self.obstacle_max)) | hole_mask)
        mask_explored &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask_explored]
        scores_masked = scores_aligned[mask_explored]

        pcl_grid_ids_masked = pcl_grid_ids[mask_explored].T
        # Store far observed cell indices for unexplored_fov_area update
        pcl_grid_ids_far = pcl_grid_ids[mask_far_observed].T if mask_far_observed.any() else torch.empty((2, 0), dtype=torch.int32, device='cuda')
        values_to_add = values_aligned[mask_explored] * scores_masked.unsqueeze(1)

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1)),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)   
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        # 같은 셀에 여러 관측이 들어온 경우 합을 평균으로 변환
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        # 초기 신뢰도: depth gradient와 거리 기반으로 계산된 score
        obstcl_confidence_mapped = scores_mapped

        # 복잡한 가우시안 블러링 없이, 관측 셀에 한정해 신뢰도 사용
        # (explored_area/navigable_map 중심 사용 시 성능 부담을 줄이기 위함)
        all_ids = pcl_grid_ids_masked_unique
        coalesced_scores = scores_mapped

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, coalesced_scores, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        
        # Return observed cell indices for explored_area update (only <= 1.8m)
        observed_cell_indices = all_ids.cpu()
        
        # Return far observed cell indices (1.8m~10m) for unexplored_fov_area update
        if pcl_grid_ids_far.shape[1] > 0:
            far_cell_indices = pcl_grid_ids_far.unique(dim=1).cpu()
        else:
            far_cell_indices = torch.empty((2, 0), dtype=torch.int32)
        
        return observed_cell_indices, obstacle_mapped.cpu(), obstcl_confidence_mapped.cpu(), far_cell_indices

    def project_single(self,
                       values: torch.Tensor,
                       depth: np.ndarray,
                       tf_camera_to_episodic,
                       fx, fy, cx, cy
                       ) -> (torch.Tensor, torch.Tensor):
        """
        Projects a single value observation into the map using a heuristic, similar to VLFM
        :param values:
        :param depth:
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return:
        """
        projected_depth = self.project_depth_camera(depth, *(depth.shape[0:2]), fx, fy, cx, cy)
        # TODO needs to be implemented
        raise NotImplementedError

    def project_depth_camera(self,
                             depth: torch.Tensor,
                             camera_resolution: Tuple[int, int],
                             fx, fy, cx, cy
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the depth into 3D pointcloud. Camera resolution is passed if the depth is subsampled,
        to match value array resolution.
        :param depth: torch Tensor of shape (h, w), not necessarily the same as camera resolution
        :param camera_resolution: tuple of original camera resolution to correct depth if necessary (w, h)
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: a point cloud of shape (h * w, 3), where x is depth (points into the image),
                                                          y is horizontal (points left),
                                                          z is vertical (points up)
        """
        # TODO are the "-1" necessary?
        x = torch.arange(0, depth.shape[1], device="cuda") * (camera_resolution[1] - 1) / (depth.shape[1] - 1)
        y = torch.arange(0, depth.shape[0], device="cuda") * (camera_resolution[0] - 1) / (depth.shape[0] - 1)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_world = (xx - cx) * zz / fx
        y_world = (yy - cy) * zz / fy
        z_world = zz
        point_cloud = torch.vstack((z_world, -x_world, -y_world)).T
        if self.filter_stairs:
            hole_mask = -y_world < self.floor_threshold # todo threshold parameter
            if hole_mask.any():
                scale_factor = self.floor_level / -y_world[hole_mask]
                point_cloud[hole_mask] *= scale_factor.unsqueeze(-1)
                return point_cloud, hole_mask

        return point_cloud, torch.empty((0,))

    def metric_to_px(self, x, y):
        epsilon = 1e-9  # Small value to account for floating-point imprecision

        return (
            int(x / self.cell_size + self.map_center_cells[0].item() + epsilon),
            int(y / self.cell_size + self.map_center_cells[1].item() + epsilon))

    def px_to_metric(self, px, py):
        return ((px - self.map_center_cells[0].item()) * self.cell_size,
                (py - self.map_center_cells[1].item()) * self.cell_size)


if __name__ == "__main__":
    rr.init("rerun_example_points3d", spawn=False)
    rr.connect_grpch()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
    from detectron2.data.detection_utils import read_image

    map = OneMap(1)
    depth = read_image('test_images/depth2.png', format="BGR") * (-1) + 255
    depth2 = read_image('test_images/depth.png', format="BGR")

    fac = 10
    x = torch.arange(0, depth.shape[1] / fac, dtype=torch.float32)
    y = torch.arange(0, depth.shape[0], dtype=torch.float32)
    xx, yy, = torch.meshgrid(x, y)

    values = torch.sin(xx / (50.0 / fac)).T.unsqueeze(0)

    # values[:, :depth.shape[1]//2] = 1.0
    start = time.time()
    # map.update(torch.zeros((depth.shape[0], depth.shape[1], 3)), depth, np.eye(4))
    map.update(values, depth[:, :, 0], np.eye(4))
    map.update(-values, depth2[:, :, 0], np.eye(4))
    print(time.time() - start)
