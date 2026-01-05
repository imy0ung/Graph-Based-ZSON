"""
Mapping Rerun Logger. Sets up the experiment blueprint and logs the map and robot position.
"""
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mapping import Navigator
from onemap_utils import log_map_rerun


def log_pos(x, y):
    # x, y를 스왑하여 rerun 좌표계에 맞춤
    rr.log("map/position", rr.Points2D([[y, x]], colors=[[255, 0, 0]], radii=[1]))


def setup_blueprint_debug():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection",
                                            "$origin/rgb/masks"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Vertical(
                    rrb.TextLogView(origin="object_detections"),
                    rrb.TextLogView(origin="path_updates"),
                ),
                rrb.Spatial2DView(origin="map",
                                  name="Traversable",
                                  contents=["$origin/traversable",
                                            "$origin/position",
                                            "$origin/pose_graph/nodes",
                                            "$origin/pose_graph/edges/odometry",
                                            "$origin/pose_graph/edges/loop_closure",
                                            "$origin/pose_graph/objects",
                                            "$origin/pose_graph/edges/pose_object",
                                            "$origin/pose_graph/edges/pose_frontier"], ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/proj_detect",
                                         "$origin/position",
                                         "$origin/pose_graph/nodes",
                                         "$origin/pose_graph/edges/odometry",
                                         "$origin/pose_graph/edges/loop_closure",
                                         "$origin/pose_graph/objects",
                                         "$origin/pose_graph/edges/pose_object",
                                         "$origin/pose_graph/edges/pose_frontier",
                                        ]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholded",
                                        contents=
                                        ["$origin/similarity_th/",
                                         "$origin/proj_detect",
                                         "$origin/position",
                                         "$origin/pose_graph/nodes",
                                         "$origin/pose_graph/edges/odometry",
                                         "$origin/pose_graph/edges/loop_closure",
                                         "$origin/pose_graph/objects",
                                         "$origin/pose_graph/edges/pose_object",
                                         "$origin/pose_graph/edges/pose_frontier",
                                        ]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                "$origin/largest_contour",
                                                "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                "$origin/ground_truth",
                                                "$origin/pose_graph/nodes",
                                                "$origin/pose_graph/edges/odometry",
                                                "$origin/pose_graph/edges/loop_closure",
                                                "$origin/pose_graph/objects",
                                                "$origin/pose_graph/edges/pose_object",
                                                "$origin/pose_graph/edges/pose_frontier",
                                                "$origin/explored_objects",
                                                "$origin/explored_edges/pose_object",
                                                "$origin/explored_edges/pose_frontier",
                                                ]),
                    rrb.Spatial2DView(origin="map",
                                      name="Scores",
                                      contents=["$origin/scores",
                                                "$origin/position",
                                                "$origin/goal_pos",
                                                "$origin/path",
                                                "$origin/pose_graph/nodes",
                                                "$origin/pose_graph/edges/odometry",
                                                "$origin/pose_graph/edges/loop_closure",
                                                "$origin/pose_graph/objects",
                                                "$origin/pose_graph/edges/pose_object",
                                                "$origin/pose_graph/edges/pose_frontier",
                                               ]),
                    rrb.Spatial2DView(origin="map",
                                      name="Unexplored",
                                      contents=["$origin/largest_contour",
                                                "$origin/position",
                                                "$origin/unexplored",
                                                "$origin/pose_graph/nodes",
                                                "$origin/pose_graph/edges/odometry",
                                                "$origin/pose_graph/edges/loop_closure",
                                                "$origin/pose_graph/edges/pose_frontier",
                                               ]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored_objects", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                  rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                                angle=rr.datatypes.Angle(
                                                                                    rad=-np.pi / 2))))
    rr.log("map/explored_edges/pose_object", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                                          angle=rr.datatypes.Angle(
                                                                                              rad=-np.pi / 2))))
    rr.log("map/pose_graph/edges/pose_frontier", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                                rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                                              angle=rr.datatypes.Angle(
                                                                                                  rad=-np.pi / 2))))
    rr.log("map/explored_edges/pose_frontier", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                              rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                                            angle=rr.datatypes.Angle(
                                                                                                rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_only", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))
    for pose_path in [
        "map/pose_graph/nodes",
        "map/pose_graph/edges/odometry",
        "map/pose_graph/edges/loop_closure",
        "map/pose_graph/objects",
        "map/pose_graph/edges/pose_object",
        "map/pose_graph/edges/pose_frontier",
        "map/explored_objects",
        "map/explored_edges/pose_frontier",
    ]:
        rr.log(pose_path,
               rr.Transform3D(translation=np.array([0, 600, 0]),
                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                            angle=rr.datatypes.Angle(rad=-np.pi))))

def setup_blueprint():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection",
                                            "$origin/rgb/masks"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/position",
                                         "$origin/pose_graph/nodes",
                                         "$origin/pose_graph/edges/odometry",
                                         "$origin/pose_graph/edges/loop_closure",
                                         "$origin/pose_graph/objects",
                                         "$origin/pose_graph/edges/pose_object",
                                         "$origin/pose_graph/edges/pose_frontier",
                                        ]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                # "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                # "$origin/largest_contour",
                                                # "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                "$origin/pose_graph/nodes",
                                                "$origin/pose_graph/edges/odometry",
                                                "$origin/pose_graph/edges/loop_closure",
                                                # "$origin/ground_truth",
                                                "$origin/frontiers_only",
                                                "$origin/unexplored_fov",
                                                "$origin/explored_objects",
                                                "$origin/explored_edges/pose_object",
                                                "$origin/pose_graph/edges/pose_frontier",
                                                "$origin/explored_edges/pose_frontier",
                                                ]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Scores",
                    #                   contents=["$origin/scores",
                    #                             "$origin/position",
                    #                             "$origin/goal_pos",
                    #                             "$origin/path"]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Unexplored",
                    #                   contents=["$origin/frontiers",
                    #                             "$origin/frontiers_far",
                    #                             "$origin/largest_contour",
                    #                             "$origin/position",
                    #                             "$origin/unexplored"]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored_objects", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                  rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                                angle=rr.datatypes.Angle(
                                                                                    rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_only", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))
    for pose_path in [
        "map/pose_graph/nodes",
        "map/pose_graph/edges/odometry",
        "map/pose_graph/edges/loop_closure",
        "map/pose_graph/objects",
        "map/pose_graph/edges/pose_object",
        "map/pose_graph/edges/pose_frontier",
        "map/explored_edges/pose_object",
        "map/explored_edges/pose_frontier",
        "map/explored_objects",
    ]:
        rr.log(pose_path,
               rr.Transform3D(translation=np.array([0, 600, 0]),
                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                            angle=rr.datatypes.Angle(rad=-np.pi))))


class RerunLogger:
    def __init__(self, mapper: Navigator, to_file: bool, save_path: str, debug: bool = True):
        self.debug_log = debug
        self.to_file = to_file
        self.mapper = mapper
        rr.init("MON", spawn=False)
        if self.to_file:
            rr.save(save_path)
        else:
            rr.connect_grpc()
        if self.debug_log:
            setup_blueprint_debug()
        else:
            setup_blueprint()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_map(self):
        # Use explored_area instead of confidence_map
        explored_area = self.mapper.get_confidence_map()  # This now returns explored_area
        # similarities is no longer available (get_map returns None)
        # Use explored_area directly for visualization
        similarities = None

        # Combined visualization in explored map:
        # - Explored area (1.8m 이내): 0.5 (탐험됨)
        # - Current far observed (1.8m~10m, 현재 프레임): 0.7 (현재 시야각 내 미탐사)
        # - Non-navigable: 0
        # - 나머지 (미탐사 영역): 0 (표시 안 함)
        explored = np.zeros_like(self.mapper.one_map.navigable_map, dtype=np.float32)
        
        # Explored area (1.8m 이내, 탐험된 영역)
        explored[explored_area > 0] = 0.5
        
        # Current far observed area (1.8m~10m, 현재 프레임의 시야각 내)
        if hasattr(self.mapper.one_map, 'current_far_observed_area'):
            explored[self.mapper.one_map.current_far_observed_area & (self.mapper.one_map.navigable_map == 1)] = 0.7
        
        # Non-navigable areas remain 0

        # Removed fully_explored_map visualization to avoid confidence_map-style red regions
        # explored[self.mapper.one_map.fully_explored_map] = 1.0

        # frontiers = np.zeros((confidences.shape[0], confidences.shape[1]), dtype=np.float32)
        # for i, f in enumerate(self.mapper.frontiers):
        #     frontiers[f[:, 0, 0], f[:, 0, 1]] = 1
        # if (frontiers != 0).sum():
        #     log_map_rerun(frontiers, path="map/frontiers")

        # log_map_rerun(self.mapper.value_mapper.navigable_map, path="map/traversable")
        log_map_rerun(explored, path="map/explored")
        # Similarities are no longer available (feature_map removed)
        # log_map_rerun(similarities[0], path="map/similarity")
        # log_map_rerun(confidences, path="map/confidence")

    def log_pos(self, x, y):
        px, py = self.mapper.one_map.metric_to_px(x, y)
        log_pos(px, py)
