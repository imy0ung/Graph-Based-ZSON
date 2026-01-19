import os
import pandas as pd
from tabulate import tabulate
from eval.dataset_utils.hm3d_dataset import load_hm3d_episodes
from eval.dataset_utils import Episode
from eval.habitat_evaluator import Result
import enum

# Define Result enum if not importing from habitat_evaluator to avoid dependencies
# But since we are in the same repo, we can import it.
# from eval.habitat_evaluator import Result

def monitor_results(results_path="results/", object_nav_path="datasets/objectnav_hm3d_v1/val/content/", minutes=None):
    print(f"Loading episodes from {object_nav_path}...")
    episodes, _ = load_hm3d_episodes([], {}, object_nav_path)
    
    # Create a map from episode index to episode object
    
    state_dir = os.path.join(results_path, 'state')
    if not os.path.isdir(state_dir):
        print(f"Error: {state_dir} is not a valid directory")
        return

    state_results = {}
    object_query = {}
    scene_name = {}
    
    if minutes is not None:
        print(f"Reading results from {state_dir} (modified in last {minutes} minutes)...")
        cutoff_time = time.time() - (minutes * 60)
    else:
        print(f"Reading results from {state_dir} (all files)...")
        cutoff_time = 0
    
    # Iterate through all files to find the latest modified file index
    latest_mtime = 0
    latest_idx = -1
    
    # First pass: find the latest modified file and its index
    all_files = []
    for filename in os.listdir(state_dir):
        if filename.startswith('state_') and filename.endswith('.txt'):
            filepath = os.path.join(state_dir, filename)
            try:
                mtime = os.path.getmtime(filepath)
                idx = int(filename[6:-4])
                all_files.append((filename, idx, mtime))
                
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_idx = idx
            except:
                pass

    if latest_idx != -1:
        print(f"Latest modified result is episode {latest_idx} (from {time.ctime(latest_mtime)})")
        print(f"Filtering results to range [0, {latest_idx}]...")
    
    # Second pass: process files
    file_count = 0
    for filename, experiment_num, mtime in all_files:
        # Filter by minutes if specified
        if minutes is not None:
            if mtime < cutoff_time:
                continue
        
        # Filter by latest index (cutoff future episodes from old runs)
        if latest_idx != -1 and experiment_num > latest_idx:
            continue

        try:
            # Read the content of the file
            with open(os.path.join(state_dir, filename), 'r') as file:
                content = file.read().strip()

            # Convert the content to a number
            state_value = int(content)
            
            # Store the result
            state_results[experiment_num] = state_value
            
            # Map to object and scene using the loaded episodes
            if experiment_num < len(episodes):
                episode = episodes[experiment_num]
                object_query[experiment_num] = episode.obj_sequence[0]
                scene_name[experiment_num] = episode.scene_id
            else:
                # Fallback if episode index out of range
                object_query[experiment_num] = "unknown"
                scene_name[experiment_num] = "unknown"
            
            file_count += 1
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")

    if file_count == 0:
        print("No result files found yet.")
        return

    print(f"Found {file_count} results.\n")

    dict_res = {"state": state_results, "obj" : object_query, "scene" : scene_name}
    data = pd.DataFrame.from_dict(dict_res)

    # Calculate statistics
    states = data["state"].unique()
    
    def calculate_percentages(group):
        total = len(group)
        # Map integer state to Result enum name if possible
        try:
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
        except:
            result = pd.Series({str(state): (group['state'] == state).sum() / total for state in states})
        
        result['Count'] = total
        return result

    # Overall results
    overall_percentages = calculate_percentages(data)
    print("="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total Episodes: {int(overall_percentages['Count'])}")
    
    # Format and print overall stats
    for key, val in overall_percentages.items():
        if key != 'Count':
            print(f"{key:20s}: {val:.2%}")
    print("\n")

    # Per-object results
    object_results = data.groupby('obj').apply(calculate_percentages).reset_index()
    object_results = object_results.rename(columns={'obj': 'Object'})
    
    # Sort by SUCCESS if available
    sort_col = 'SUCCESS' if 'SUCCESS' in object_results.columns else object_results.columns[1]
    object_results = object_results.sort_values(by=sort_col, ascending=False)

    def format_percentages(val):
        if isinstance(val, float):
            return f"{val:.2%}"
        return val

    # Format columns except Count
    formatted_df = object_results.copy()
    for col in formatted_df.columns:
        if col != 'Object' and col != 'Count':
            formatted_df[col] = formatted_df[col].apply(format_percentages)
            
    # Add Total row
    total_row = {'Object': 'Total'}
    for col in formatted_df.columns:
        if col in overall_percentages:
            val = overall_percentages[col]
            if col != 'Count':
                total_row[col] = format_percentages(val)
            else:
                total_row[col] = val
    
    formatted_df = pd.concat([formatted_df, pd.DataFrame([total_row])], ignore_index=True)

    print("="*60)
    print("RESULTS BY OBJECT")
    print("="*60)
    print(tabulate(formatted_df, headers='keys', tablefmt='pretty', showindex=False))

import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor evaluation results.")
    parser.add_argument("--minutes", type=float, default=None, help="Only include results modified in the last N minutes.")
    args = parser.parse_args()
    
    monitor_results(minutes=args.minutes)
