"""
This script is used to test the habitat-sim library together with OneMap
"""
import time
from pathlib import Path

# habitat
import habitat_sim
from habitat_sim.utils import common as utils

# numpy
import numpy as np

# rerun
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim import ActionSpec, ActuationSpec
from numpy.lib.function_base import angle

# keyboard input
# from pynput import keyboard

# scipy
from scipy.spatial.transform import Rotation as R

# MON
from mapping import Navigator
from mapping import Frontier
from mapping import ObjectNode
from semantic_prototypes import PrototypeConfig, SemanticPrototypeIndex
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from vision_models.yolov7_model import YOLOv7Detector

# from onemap_utils import log_map_rerun
from planning import Planning, Controllers
from config import *
from mapping import rerun_logger

# Global variables
running = True

if __name__ == "__main__":
    config = load_config().Conf
    if type(config.controller) == HabitatControllerConf:
        pass
    else:
        raise NotImplementedError("Spot controller not suited for habitat sim")

    # Initialize database - reset for each run
    db_path = "pose_graph.db"
    if Path(db_path).exists():
        print(f"Removing existing database: {db_path}")
        Path(db_path).unlink()
    
    model = ClipModel("weights/clip.pth")
    # Target object detector (for navigation)
    #detector = YOLOWorldDetector(0.8)
    detector = YOLOv7Detector(0.8)
    mapper = Navigator(model, detector, config)
    proto_config = PrototypeConfig()
    proto_index = SemanticPrototypeIndex(model, config=proto_config, auto_build=False)
    proto_index.build_or_load(ignore_cache=True)
    mapper.pose_graph.set_semantic_prototypes(proto_index)
    logger = rerun_logger.RerunLogger(mapper, False, "", debug=False) if config.log_rerun else None

    mapper.debug_observation_distance = True
    mapper.pose_graph.debug_map_logging = True


    
    # Multi-object navigation: list of objects to find
    #qs = ["A fridge", "A TV", "A toilet", "A Couch", "A bed"]
    qs =["toilet", "bed", "couch"]
    #qs =["bed"]
    mapper.set_query([qs[0]])  # Start with first object
    hm3d_path = "datasets/scene_datasets/hm3d"

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    #backend_cfg.scene_id = hm3d_path + "/val/00809-Qpor2mEya8F/Qpor2mEya8F.basis.glb"
    backend_cfg.scene_dataset_config_file = hm3d_path + "/hm3d_annotated_basis.scene_dataset_config.json"

    hfov = 90
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = hfov
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    res_x = 640
    res_y = 640
    rgb.resolution = [res_y, res_x]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.hfov = hfov
    depth.position = np.array([0, 0.88, 0])
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [res_y, res_x]

    hfov = np.deg2rad(hfov)
    focal_length = (res_x / 2) / np.tan(hfov / 2)
    principal_point_x = res_x / 2
    principal_point_y = res_y / 2
    K = np.array([
        [focal_length, 0, principal_point_x],
        [0, focal_length, principal_point_y],
        [0, 0, 1]
    ])

    agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
        move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
        turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
    ))
    agent_cfg.sensor_specifications = [rgb, depth]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    objects = sim.semantic_scene.objects
    categories = [ob.category.name() for ob in objects]
    scene_categories = sim.semantic_scene.categories
    scene_categories = [cat.name() for cat in scene_categories]
    for cat in categories:
        if cat not in scene_categories:
            print("Object category not in scene categories:", cat)

    for cat in scene_categories:
        if cat not in categories:
            print("Scene category not in object categories:", cat)
    print(len(categories), len(scene_categories))
    print("Unique categories:", len(set(categories)))
    print(set(categories))

    # Global flag to control the simulation loop
    running = True

    action_mapping = {
        "w": "move_forward",
        "a": "turn_left",
        "d": "turn_right",
        "q": "enter_query",
        "o": "autonomous",
    }

    # Main simulation loop
    initial_sequence = ["turn_left"] * 28 * 2  # + ["move_forward"]*10
    # initial_sequence = ["turn_left"]*5 + ["move_forward"]*5
    running = True
    autonomous = True
    controller = Controllers.HabitatController(sim, config.controller)
    while running:
        action = None
        if len(initial_sequence):
            action = initial_sequence[0]
            initial_sequence.pop(0)
            if not len(initial_sequence):
                mapper.one_map.reset_checked_map()
        elif autonomous:
            action = None
            # print("Goal pos: ", goal_pos)

            state = sim.get_agent(0).get_state()
            orientation = state.rotation
            q0 = orientation.x
            q1 = orientation.y
            q2 = orientation.z
            q3 = orientation.w

            r = R.from_quat([q0, q1, q2, q3])
            # r to euler
            pitch, yaw, roll = r.as_euler("yxz")
            # pitch is actually around z
            # orientation is pitch!
            yaw = pitch
            current_pos = np.array([[-state.position[2]], [-state.position[0]], [state.position[1]]])
            path = mapper.get_path()
            # rr.log("map/path", rr.LineStrips2D(path))
            if path and len(path) > 1:
                path = Planning.simplify_path(np.array(path))
                path = path.astype(np.float32)
                for i in range(path.shape[0]):
                    path[i, :] = mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
                controller.control(current_pos, yaw, path)
                observations = sim.get_sensor_observations()
        if action and action != "enter_query":
            observations = sim.step(action)
        elif not autonomous:
            continue

        state = sim.get_agent(0).get_state()
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
        # print(pos)
        mapper.set_camera_matrix(K)
        orientation = state.rotation
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w

        r = R.from_quat([q0, q1, q2, q3])
        # r to euler
        pitch, yaw, roll = r.as_euler("yxz")
        # pitch is actually around z
        r = R.from_euler("xyz", [0, 0, pitch])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        t = time.time()
        obj_found = mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1), observations["depth"].astype(np.float32),
                        transformation_matrix)

        # print("Time taken to add data: ", time.time() - t)
        
        # Print nav_goals coordinates for debugging (every 10 steps)
        if mapper.pose_graph._step_counter % 10 == 0:
            print(f"\n[Step {mapper.pose_graph._step_counter}] Nav Goals ({len(mapper.nav_goals)} total):")
            for i, nav_goal in enumerate(mapper.nav_goals): # cluster 삭제 필요 (코드 수정 필요)
                coord = nav_goal.get_descr_point()
                score = nav_goal.get_score()
                goal_type = type(nav_goal).__name__
                if goal_type == "Frontier":
                    print(f"  [{i}] {goal_type}: coord=({coord[0]:.2f}, {coord[1]:.2f}), score={score:.4f}")
        
        # Print graph statistics periodically
        if mapper.pose_graph._step_counter % 10 == 0:
            stats = mapper.pose_graph.get_statistics()
            print(f"\n[Step {mapper.pose_graph._step_counter}] Graph Stats:")
            print(f"  Poses: {stats['pose_nodes']}, Objects: {stats['object_nodes']}, "
                  f"Edges: {stats['edge_count']} (pose_pose: {stats['pose_pose_edges']}, "
                  f"pose_object: {stats['pose_object_edges']})")

            if len(mapper.pose_graph.object_ids) > 0:
                print(f"  Registered Objects:")
                for obj_id in mapper.pose_graph.object_ids:
                    obj_node = mapper.pose_graph.nodes[obj_id]
                    clip_info = f", clip={obj_node.avg_clip_score:.3f}" if obj_node.clip_scores else ", clip=N/A"
                    verified_mark = "✓" if obj_node.clip_verified else "✗" if obj_node.clip_scores else "?"
                    print(f"    - {obj_node.label} [{verified_mark}]: pos=({obj_node.position[0]:.2f}, {obj_node.position[1]:.2f}), "
                          f"conf={obj_node.confidence:.2f}, obs={obj_node.num_observations}{clip_info}")

                max_print = 10
                obj_ids = mapper.pose_graph.object_ids[-max_print:]
                print(f"\n[Step {mapper.pose_graph._step_counter}] Objects (showing {len(obj_ids)}/{len(mapper.pose_graph.object_ids)}):")
                for obj_id in obj_ids:
                    obj_node = mapper.pose_graph.nodes[obj_id]
                    label = getattr(obj_node, "label_final", None) or obj_node.label
                    conf = float(getattr(obj_node, "confidence", 0.0))
                    conf_w = getattr(obj_node, "confidence_weighted", None)
                    sim_in = getattr(obj_node, "sim_indoor", None)
                    sim_out = getattr(obj_node, "sim_outdoor", None)
                    sim_margin = getattr(obj_node, "sim_margin", None)
                    is_outdoor = getattr(obj_node, "is_outdoor", None)
                    conf_w_str = f"{conf_w:.3f}" if conf_w is not None else "None"
                    sim_in_str = f"{sim_in:.3f}" if sim_in is not None else "None"
                    sim_out_str = f"{sim_out:.3f}" if sim_out is not None else "None"
                    sim_margin_str = f"{sim_margin:.3f}" if sim_margin is not None else "None"
                    print(
                        f"  - {label}: conf={conf:.3f} conf_w={conf_w_str} "
                        f"sim_in={sim_in_str} sim_out={sim_out_str} "
                        f"margin={sim_margin_str} outdoor={is_outdoor}"
                    )

        # Print target object info if navigating to graph-based target
        if mapper.target_object_node is not None:
            print(f"[Target] Navigating to '{mapper.target_object_node.label}' "
                  f"at ({mapper.target_object_node.position[0]:.2f}, {mapper.target_object_node.position[1]:.2f})")

        cam_x = pos[0, 0]
        cam_y = pos[1, 0]
        if logger:
            rr.log("camera/rgb", rr.Image(observations["rgb"]))
            rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                    observations["depth"].max() - observations["depth"].min())))
            logger.log_map()
            logger.log_pos(cam_x, cam_y)
            
            # Extract and visualize Frontier coordinates in rerun (every 10 steps)
            if mapper.pose_graph._step_counter % 10 == 0:
                frontier_coords = []
                for nav_goal in mapper.nav_goals:
                    if isinstance(nav_goal, Frontier):
                        coord = nav_goal.get_descr_point()
                        # Convert from [y, x] to [x, y] for rerun visualization (same as path)
                        frontier_coords.append([coord[1], coord[0]])
                
                if len(frontier_coords) > 0:
                    # Create small circles for each frontier
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
                    
                    # Log all circles
                    rr.log("map/frontiers_only", 
                           rr.LineStrips2D(circle_strips, 
                                         colors=[green_color] * len(circle_strips)))
        if obj_found:
            print(f"Object '{mapper.query_text[0]}' found! Moving to next object...")
            if len(qs) > 0:
                next_obj = qs[0]
                qs.pop(0)
                mapper.set_query([next_obj])
                print(f"Now searching for: {next_obj}")
            else:
                # All objects found, exit
                print("All target objects found. Exiting...")
                running = False
                break
            # Continue to next iteration to start searching for new object
            continue
        
        # Exit if no more queries and no path
        if len(qs) == 0 and (mapper.get_path() is None or len(mapper.get_path()) == 0):
            print("No more queries and no path. Exiting...")
            running = False
            break
