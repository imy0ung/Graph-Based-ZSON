from eval.habitat_evaluator import HabitatEvaluator
from config import load_eval_config
from eval.habitat_evaluator import Result
import os
import numpy as np
import pandas as pd
from tabulate import tabulate


class HabitatEvaluator500(HabitatEvaluator):
    """
    Modified HabitatEvaluator for reading 500-episode evaluation results.
    Uses results2/ directory with subdirectories: state2/ and trajectories/
    """
    
    def __init__(self, config, actor):
        super().__init__(config, actor)
        # Override results_path to use results2 instead of results
        self.results_path = "/home/finn/active/MON/results2_gibson" if self.is_gibson else "results2/"
    
    def read_results(self, path, sort_by):
        """
        Override read_results to read from state2/ subdirectory instead of state/.
        This is identical to the parent class except for the path to state directory.
        """
        state_dir = os.path.join(path, 'state2')  # Changed from 'state' to 'state2'
        state_results = {}
        object_query = {}
        scene_name = {}
        spl = {}

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        for filename in os.listdir(state_dir):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    # Extract the experiment number from the filename
                    experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                    
                    # Read the content of the file
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    # Convert the content to a number (assuming it's a float)
                    state_value = int(content)
                    # Store the result in the dictionary
                    state_results[experiment_num] = state_value
                    object_query[experiment_num] = self.episodes[experiment_num].obj_sequence[0]
                    scene_name[experiment_num] = self.episodes[experiment_num].scene_id
                    poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + ".csv"), delimiter=",")
                    deltas = poses[1:, :2] - poses[:-1, :2]
                    distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                    if state_value == 1:
                        spl[experiment_num] = self.episodes[experiment_num].best_dist / max(self.episodes[experiment_num].best_dist, distance_traveled)
                    else:
                        spl[experiment_num] = 0
                    if self.episodes[experiment_num].episode_id != experiment_num:
                        print(f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        dict_res = {"state": state_results, "obj" : object_query, "scene" : scene_name, "spl" : spl}
        data = pd.DataFrame.from_dict(dict_res)

        states = data["state"].unique()

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})

            # Calculate average SPL and multiply by 100
            avg_spl = group['spl'].mean()
            result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('obj').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'obj': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        return data


def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    # Create the HabitatEvaluator500 object
    evaluator = HabitatEvaluator500(eval_config.EvalConf, None)
    data = evaluator.read_results('results2/', Result.SUCCESS.name)
    # print(data[data['state'] == 2].index.tolist())
    # print(data[data['scene'] == 'hm3d/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb'][data['state'] == 5].index.tolist())
    # print(data[data['state'] == 5].index.tolist())
    # print(len(data[data['obj'] == 'bed'].index.tolist()))


if __name__ == "__main__":
    main()
