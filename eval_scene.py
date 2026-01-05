#!/usr/bin/env python3
"""
Evaluate a specific scene (or scenes) and analyze SR and SPL metrics.

Usage:
    python eval_scene.py --scene_id "00878-XB4GS9ShBRE"
    python eval_scene.py --scene_id "00878-XB4GS9ShBRE" --episode_ids 1683 1690 1692
    python eval_scene.py --scene_id "00878-XB4GS9ShBRE" --max_episodes 10
"""
import sys
import os
import argparse
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from eval.habitat_evaluator import HabitatEvaluator, Result
from config import load_eval_config
from eval.actor import MONActor
from eval.dataset_utils.hm3d_dataset import load_hm3d_episodes
from tabulate import tabulate
import pandas as pd
import numpy as np


class SceneEvaluator(HabitatEvaluator):
    """Extended evaluator that filters episodes by scene ID"""
    
    def __init__(self, config, actor, scene_id: str, episode_ids: Optional[List[int]] = None, max_episodes: Optional[int] = None):
        super().__init__(config, actor)
        
        # Filter episodes by scene_id
        original_episodes = self.episodes.copy()
        self.episodes = [ep for ep in original_episodes if scene_id in ep.scene_id]
        
        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found for scene_id: {scene_id}")
        
        print(f"Found {len(self.episodes)} episodes for scene: {scene_id}")
        
        # Further filter by episode_ids if provided
        if episode_ids is not None:
            episode_id_set = set(episode_ids)
            self.episodes = [ep for ep in self.episodes if ep.episode_id in episode_id_set]
            print(f"Filtered to {len(self.episodes)} episodes by episode_ids")
        
        # Limit number of episodes if max_episodes is specified
        if max_episodes is not None and max_episodes > 0:
            self.episodes = self.episodes[:max_episodes]
            print(f"Limited to {len(self.episodes)} episodes")
        
        # Store scene_id for analysis
        self.target_scene_id = scene_id
        
        # Print episode info
        print(f"\nEpisodes to evaluate:")
        for ep in self.episodes:
            print(f"  Episode {ep.episode_id}: target={ep.obj_sequence[0]}, start_height={ep.start_position[1]:.2f}m")
        print()
    
    def evaluate(self):
        """Override evaluate to add scene-specific analysis"""
        # Call parent evaluate
        super().evaluate()
        
        # Analyze results for this scene
        print("\n" + "="*80)
        print(f"SCENE-SPECIFIC ANALYSIS: {self.target_scene_id}")
        print("="*80)
        
        # Read results
        data = self.read_results(self.results_path, sort_by='SUCCESS')
        
        # Filter data for this scene
        scene_data = data[data['scene'].str.contains(self.target_scene_id, na=False)]
        
        if len(scene_data) == 0:
            print("No results found for this scene.")
            return
        
        # Calculate metrics
        total_episodes = len(scene_data)
        success_count = (scene_data['state'] == Result.SUCCESS.value).sum()
        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
        avg_spl = scene_data['spl'].mean()
        
        print(f"\nOverall Metrics for Scene:")
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Success Rate (SR): {success_rate:.2%}")
        print(f"  Average SPL: {avg_spl:.4f}")
        
        # Per-object breakdown
        print(f"\nPer-Object Metrics:")
        obj_grouped = scene_data.groupby('obj')
        obj_results = pd.DataFrame({
            'Count': obj_grouped.size(),
            'Success': obj_grouped['state'].apply(lambda x: (x == Result.SUCCESS.value).sum()),
            'SR': obj_grouped['state'].apply(lambda x: (x == Result.SUCCESS.value).sum() / len(x)),
            'Avg SPL': obj_grouped['spl'].mean()
        })
        
        # Reorder columns
        obj_results = obj_results[['Count', 'Success', 'SR', 'Avg SPL']]
        obj_results['SR'] = obj_results['SR'].apply(lambda x: f"{x:.2%}")
        obj_results['Avg SPL'] = obj_results['Avg SPL'].apply(lambda x: f"{x:.4f}")
        
        print(tabulate(obj_results, headers='keys', tablefmt='pretty'))
        
        # Failure breakdown
        print(f"\nFailure Breakdown:")
        failure_counts = scene_data['state'].value_counts()
        failure_names = {r.value: r.name for r in Result}
        for state_value, count in failure_counts.items():
            if state_value != Result.SUCCESS.value:
                print(f"  {failure_names.get(state_value, f'Unknown({state_value})')}: {count} ({count/total_episodes:.2%})")
        
        # Episode-by-episode results
        print(f"\nEpisode-by-Episode Results:")
        episode_results = scene_data[['state', 'obj', 'spl']].copy()
        episode_results['Result'] = episode_results['state'].map(lambda x: failure_names.get(x, f'Unknown({x})'))
        episode_results['SR'] = episode_results['state'].apply(lambda x: '✓' if x == Result.SUCCESS.value else '✗')
        episode_results = episode_results[['obj', 'SR', 'Result', 'spl']]
        episode_results.columns = ['Object', 'Success', 'Result', 'SPL']
        episode_results['SPL'] = episode_results['SPL'].apply(lambda x: f"{x:.4f}" if x > 0 else "0.0000")
        
        print(tabulate(episode_results, headers='keys', tablefmt='pretty', showindex=True))
        
        return {
            'scene_id': self.target_scene_id,
            'total_episodes': total_episodes,
            'success_rate': success_rate,
            'avg_spl': avg_spl,
            'per_object': obj_results.to_dict(),
            'episode_results': episode_results
        }


def main():
    # Parse our custom arguments first, before Spock loads
    # We need to extract our args from sys.argv before Spock processes it
    custom_args = []
    scene_id = None
    episode_ids = None
    max_episodes = None
    config_path = None
    
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--scene_id' and i + 1 < len(sys.argv):
            scene_id = sys.argv[i + 1]
            i += 2
        elif arg == '--episode_ids':
            episode_ids = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                try:
                    episode_ids.append(int(sys.argv[i]))
                    i += 1
                except ValueError:
                    break
        elif arg == '--max_episodes' and i + 1 < len(sys.argv):
            max_episodes = int(sys.argv[i + 1])
            i += 2
        elif arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        else:
            custom_args.append(arg)
            i += 1
    
    # Remove our custom args from sys.argv so Spock doesn't see them
    sys.argv = custom_args
    
    if scene_id is None:
        print("Error: --scene_id is required")
        print("Usage: python eval_scene.py --scene_id <scene_id> [--episode_ids <id1> <id2> ...] [--max_episodes <N>] [--config <path>]")
        sys.exit(1)
    
    # Load config (Spock will process remaining sys.argv)
    # If no config specified, use default config path
    if config_path:
        sys.argv.extend(['--config', config_path])
    else:
        # Use default config if available (same as eval_habitat.py behavior)
        # Check common config locations
        default_configs = [
            "config/mon/eval_conf.yaml",
            "config/eval_conf.yaml",
        ]
        config_found = False
        for default_config in default_configs:
            if os.path.exists(default_config):
                sys.argv.extend(['--config', default_config])
                config_found = True
                break
        # If no default config found, let Spock handle it (will require --config from command line)
        if not config_found:
            print("Warning: No default config file found. Please specify --config <path>")
            print("  Common locations: config/mon/eval_conf.yaml, config/eval_conf.yaml")
    
    eval_config = load_eval_config()
    
    # Create evaluator
    try:
        evaluator = SceneEvaluator(
            eval_config.EvalConf,
            MONActor(eval_config.EvalConf),
            scene_id=scene_id,
            episode_ids=episode_ids,
            max_episodes=max_episodes
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        print("\n" + "="*80)
        print("Evaluation completed!")
        print("="*80)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

