from eval.habitat_evaluator import HabitatEvaluator
from config import load_eval_config
from eval.actor import MONActor
import habitat_sim
import os


class HabitatEvaluator500(HabitatEvaluator):
    """
    Modified HabitatEvaluator for testing 500 episodes evenly distributed across environments.
    Uses results2/ directory with subdirectories: trajectories/, similarities2/, state2/
    """
    
    def __init__(self, config, actor):
        super().__init__(config, actor)
        # Override results_path to use results2 instead of results
        self.results_path = "/home/finn/active/MON/results2_gibson" if self.is_gibson else "results2/"
        self._ensure_results_dirs()
        
        # Select 500 episodes evenly distributed from the full dataset
        # Take every 4th episode to ensure even distribution (assuming ~2000 episodes)
        total_episodes = len(self.episodes)
        step_size = max(1, total_episodes // 500)  # Calculate step size dynamically
        selected_indices = list(range(0, total_episodes, step_size))[:500]  # Take first 500
        
        # Filter episodes to only keep selected ones
        self.episodes = [self.episodes[i] for i in selected_indices]
        
        print(f"Selected {len(self.episodes)} episodes from {total_episodes} total episodes")
        print(f"Using step size: {step_size} (every {step_size}th episode)")
    
    def _ensure_results_dirs(self):
        """Override to create results2 directory structure with similarities2 and state2"""
        base = self.results_path
        for subdir in ("", "trajectories", "similarities2", "state2"):
            os.makedirs(os.path.join(base, subdir), exist_ok=True)
    
    def evaluate(self):
        """
        Override evaluate to use state2 and similarities2 subdirectories.
        This is a modified version that saves to the correct subdirectories.
        """
        # Import dependencies needed for evaluate
        from eval import get_closest_dist
        from scipy.spatial.transform import Rotation as R
        from eval.habitat_evaluator import Result
        import numpy as np
        import cv2
        from onemap_utils import monochannel_to_inferno_rgb
        
        success = 0
        n_eps = 0
        success_per_obj = {}
        obj_count = {}
        results = []
        
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            results.append(Result.FAILURE_OOT)
            steps = 0
            if n_ep in self.exclude_ids:
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            
            self.sim.initialize_agent(0, habitat_sim.AgentState(
                episode.start_position, episode.start_rotation))
            self.actor.reset()
            current_obj_id = 0
            current_obj = episode.obj_sequence[current_obj_id]
            if current_obj not in success_per_obj:
                success_per_obj[current_obj] = 0
                obj_count[current_obj] = 1
            else:
                obj_count[current_obj] += 1
            self.actor.set_query(current_obj)
            
            if self.log_rerun:
                import rerun as rr
                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            while steps < self.max_steps and current_obj_id < len(episode.obj_sequence):
                observations = self.sim.get_sensor_observations()
                observations['state'] = self.sim.get_agent(0).get_state()
                pose = np.zeros((4, ))
                pose[0] = -observations['state'].position[2]
                pose[1] = -observations['state'].position[0]
                pose[2] = observations['state'].position[1]
                # yaw
                orientation = observations['state'].rotation
                q0 = orientation.x
                q1 = orientation.y
                q2 = orientation.z
                q3 = orientation.w
                r = R.from_quat([q0, q1, q2, q3])
                # r to euler
                yaw, _, _1 = r.as_euler("yxz")
                pose[3] = yaw

                poses.append(pose)
                if self.log_rerun:
                    import rerun as rr
                    cam_x = -self.sim.get_agent(0).get_state().position[2]
                    cam_y = -self.sim.get_agent(0).get_state().position[0]
                    rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                    self.logger.log_pos(cam_x, cam_y)
                action, called_found = self.actor.act(observations)
                self.execute_action(action)
                if self.log_rerun:
                    self.logger.log_map()

                if called_found:
                    # We will now compute the closest distance to the bounding box of the object
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    if dist < self.max_dist:
                        results[n_ep] = Result.SUCCESS
                        success += 1
                        print("Object found!")
                        success_per_obj[current_obj] += 1
                    else:
                        # chosen_detection may be None if object was reached (reset in add_data)
                        pos = self.actor.mapper.chosen_detection
                        if pos is not None:
                            pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                            dist_detect = get_closest_dist([-pos_metric[1], -pos_metric[0]],
                                                self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                            if dist_detect < self.max_dist:
                                results[n_ep] = Result.FAILURE_NOT_REACHED
                            else:
                                results[n_ep] = Result.FAILURE_MISDETECT
                            print(f"Object not found! Dist {dist}, detect dist: {dist_detect}.")
                        else:
                            # If chosen_detection is None but called_found is True,
                            # object was reached but agent is too far from ground truth
                            results[n_ep] = Result.FAILURE_MISDETECT
                            print(f"Object not found! Dist {dist}, chosen_detection was reset.")
                    current_obj_id += 1

                if steps % 100 == 0:
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    print(f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}")
                steps += 1
            poses = np.array(poses)
            # If the last 10 poses didn't change much and we have OOT, assume stuck
            if results[n_ep] == Result.FAILURE_OOT and np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                results[n_ep] = Result.FAILURE_STUCK

            num_frontiers = len(self.actor.mapper.nav_goals)
            np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv", poses, delimiter=",")
            # save final exploration map to image file - save to similarities2
            try:
                explored_map = self.actor.mapper.get_confidence_map()  # Returns explored_area
                final_sim = explored_map.astype(np.float32)
                final_sim = final_sim.transpose((1, 0))
                final_sim = np.flip(final_sim, axis=0)
                final_sim = monochannel_to_inferno_rgb(final_sim)
                cv2.imwrite(f"{self.results_path}/similarities2/final_sim_{episode.episode_id}.png", final_sim)
            except Exception as e:
                print(f"Warning: Could not save similarity image: {e}")
            if (results[n_ep] == Result.FAILURE_STUCK or results[n_ep] == Result.FAILURE_OOT) and num_frontiers == 0:
                results[n_ep] = Result.FAILURE_ALL_EXPLORED
            print(f"Overall success: {success / (n_eps)}, per object: ")
            for obj in success_per_obj.keys():
                print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            print(
                f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            # Write result to file - save to state2
            with open(f"{self.results_path}/state2/state_{episode.episode_id}.txt", 'w') as f:
                f.write(str(results[n_ep].value))


def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator500 object instead of HabitatEvaluator
    evaluator = HabitatEvaluator500(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    evaluator.evaluate()

if __name__ == "__main__":
    main()
