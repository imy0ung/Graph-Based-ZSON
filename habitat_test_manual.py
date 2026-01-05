"""
수동 조종 전용 스크립트 - 키보드로 로봇을 직접 조종할 수 있습니다.
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

# scipy
from scipy.spatial.transform import Rotation as R

# MON
from mapping import Navigator
from mapping import Frontier
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from vision_models.yolov7_model import YOLOv7Detector

from planning import Planning, Controllers
from config import *
from mapping import rerun_logger

if __name__ == "__main__":
    import sys
    import os
    # Config 파일 경로 명시적으로 지정 (habitat_test.py와 동일하게)
    # 기본 config 파일 사용
    if "--config" not in sys.argv:
        # 현재 스크립트의 디렉토리 기준으로 config 파일 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config", "mon", "base_conf_sim.yaml")
        if not os.path.exists(config_path):
            # 상대 경로로도 시도
            config_path = "config/mon/base_conf_sim.yaml"
        sys.argv.extend(["--config", config_path])
    
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
    detector = YOLOv7Detector(0.8)
    mapper = Navigator(model, detector, config)
    # 수동 조종 모드에서는 항상 rerun 활성화 (config가 frozen이므로 logger는 항상 생성)
    logger = rerun_logger.RerunLogger(mapper, False, "", debug=False)

    mapper.debug_observation_distance = True
    mapper.pose_graph.debug_map_logging = True
    
    # Query 설정
    qs = ["tv"]
    mapper.set_query([qs[0]])
    hm3d_path = "datasets/scene_datasets/hm3d"

    backend_cfg = habitat_sim.SimulatorConfiguration()
    #backend_cfg.scene_id = hm3d_path + "/val/00878-XB4GS9ShBRE/XB4GS9ShBRE.basis.glb"
    #backend_cfg.scene_id = hm3d_path + "/val/00820-mL8ThkuaVTM/mL8ThkuaVTM.basis.glb"
    backend_cfg.scene_id = hm3d_path + "/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb" # (sr 23.23) + stair
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

    # 액션 매핑
    action_mapping = {
        "w": "move_forward",
        "a": "turn_left",
        "d": "turn_right",
        "s": "stop",  # 정지 (액션 없음)
        "q": "quit",  # 종료
    }

    print("\n" + "="*60)
    print("수동 조종 모드")
    print("="*60)
    print("조작법:")
    print("  w: 앞으로 이동")
    print("  a: 왼쪽으로 회전")
    print("  d: 오른쪽으로 회전")
    print("  s: 정지 (액션 없음)")
    print("  q: 종료")
    print("="*60)
    print("명령을 입력하고 Enter를 누르세요.\n")

    running = True
    observations = None

    # 초기 관측 획득
    state = sim.get_agent(0).get_state()
    pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
    mapper.set_camera_matrix(K)
    orientation = state.rotation
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    r = R.from_quat([q0, q1, q2, q3])
    pitch, yaw, roll = r.as_euler("yxz")
    yaw = pitch
    r = R.from_euler("xyz", [0, 0, pitch])
    r = r.as_matrix()
    transformation_matrix = np.hstack((r, pos))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
    
    observations = sim.get_sensor_observations()

    while running:
        # 사용자 입력 받기
        try:
            user_input = input(f"[Step {mapper.pose_graph._step_counter}] 명령 입력 (w/a/d/s/q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        action = None
        
        if user_input in action_mapping:
            action_str = action_mapping[user_input]
            
            if action_str == "quit":
                print("종료합니다.")
                running = False
                break
            elif action_str == "stop":
                action = None
                print("정지")
            else:
                action = action_str
                print(f"액션 실행: {action_str}")
        else:
            print(f"알 수 없는 명령: {user_input}")
            continue

        # 액션 실행
        if action:
            observations = sim.step(action)
        else:
            # 액션이 없어도 현재 상태 업데이트
            observations = sim.get_sensor_observations()

        # 현재 상태 가져오기
        state = sim.get_agent(0).get_state()
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
        mapper.set_camera_matrix(K)
        orientation = state.rotation
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w

        r = R.from_quat([q0, q1, q2, q3])
        pitch, yaw, roll = r.as_euler("yxz")
        yaw = pitch
        r = R.from_euler("xyz", [0, 0, pitch])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

        # 데이터 추가
        obj_found = mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1), 
            observations["depth"].astype(np.float32),
            transformation_matrix
        )

        # 그래프 통계 출력 (매 5 스텝마다)
        if mapper.pose_graph._step_counter % 5 == 0:
            stats = mapper.pose_graph.get_statistics()
            print(f"\n[Graph Stats] Poses: {stats['pose_nodes']}, Objects: {stats['object_nodes']}, "
                  f"Edges: {stats['edge_count']}")
            
            # 등록된 객체 출력
            if len(mapper.pose_graph.object_ids) > 0:
                print(f"  Registered Objects:")
                for obj_id in mapper.pose_graph.object_ids:
                    obj_node = mapper.pose_graph.nodes[obj_id]
                    clip_info = f", clip={obj_node.avg_clip_score:.3f}" if obj_node.clip_scores else ", clip=N/A"
                    verified_mark = "✓" if obj_node.clip_verified else "✗" if obj_node.clip_scores else "?"
                    print(f"    - {obj_node.label} [{verified_mark}]: pos=({obj_node.position[0]:.2f}, {obj_node.position[1]:.2f}), "
                          f"conf={obj_node.confidence:.2f}, obs={obj_node.num_observations}{clip_info}")

        # Rerun 로깅
        cam_x = pos[0, 0]
        cam_y = pos[1, 0]
        if logger:
            rr.log("camera/rgb", rr.Image(observations["rgb"]))
            rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                    observations["depth"].max() - observations["depth"].min())))
            logger.log_map()
            logger.log_pos(cam_x, cam_y)

        if obj_found:
            print(f"\n*** Object '{mapper.query_text[0]}' found! ***\n")

    print("시뮬레이션 종료")

