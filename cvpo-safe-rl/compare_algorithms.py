#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import time
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare algorithms on Carla")
    
    # Algorithms to compare
    parser.add_argument("--algorithms", "-a", type=str, nargs="+", 
                        default=["cvpo", "sac_lag"],
                        help="Algorithms to compare")
    
    # Common parameters
    parser.add_argument("--town", type=str, default="Town05", 
                        help="Carla town to use")
    parser.add_argument("--epochs", type=int, default=300, 
                        help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--cost_limit", type=float, default=50.0, 
                        help="Safety cost limit")
    
    # Environment options
    parser.add_argument("--num_vehicles", type=int, default=0, 
                        help="Number of vehicles in environment")
    parser.add_argument("--desired_speed", type=float, default=5.0, 
                        help="Desired speed")
    parser.add_argument("--lane_threshold", type=float, default=3.0, 
                        help="Lane threshold")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="carla_comparison", 
                        help="Output directory")
    parser.add_argument("--eval_after", action="store_true", 
                        help="Evaluate policies after training")
    parser.add_argument("--eval_episodes", type=int, default=5, 
                        help="Number of episodes for evaluation")
    parser.add_argument("--verbose", action="store_true", 
                        help="Verbose output")
    
    # Device options
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    return args

def run_algorithm(algorithm, args, output_dir):
    """Run a single algorithm"""
    # Create command
    cmd = [
        "python", "run_carla_comparison.py",
        "--policy", algorithm,
        "--town", args.town,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--cost_limit", str(args.cost_limit),
        "--num_vehicles", str(args.num_vehicles),
        "--desired_speed", str(args.desired_speed),
        "--lane_threshold", str(args.lane_threshold),
        "--output_dir", output_dir,
        "--device", args.device,
        "--exp_name", f"{algorithm}_s{args.seed}"
    ]
    
    # Add verbose flag if specified
    if args.verbose:
        cmd.append("--verbose")
    
    # Print command
    print(f"\n=== Running {algorithm} ===")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    start_time = time.time()
    
    # Redirect stderr to /dev/null to suppress error messages
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull, universal_newlines=True)
    
    # Capture output
    output = []
    for line in process.stdout:
        print(line, end='')
        output.append(line)
    
    process.wait()
    
    # Check result
    if process.returncode != 0:
        print(f"Error: Algorithm {algorithm} failed with return code {process.returncode}")
        return False, output
    
    # Print time taken
    elapsed_time = time.time() - start_time
    print(f"{algorithm} completed in {elapsed_time:.1f} seconds")
    
    return True, output

def evaluate_algorithm(algorithm, args, output_dir):
    """Evaluate a trained algorithm"""
    # Find the latest model checkpoint
    alg_dir = os.path.join(output_dir, f"{algorithm}_s{args.seed}")
    
    # Find model path - look for the highest epoch
    model_dir = os.path.join(alg_dir, "model_save") if os.path.exists(os.path.join(alg_dir, "model_save")) else alg_dir
    if not os.path.exists(model_dir):
        print(f"Error: Cannot find model directory for {algorithm} at {model_dir}")
        return None, None
    
    # Find model file with highest epoch
    model_files = []
    for file in os.listdir(model_dir):
        if file.startswith("model_epoch_") and file.endswith(".pt"):
            epoch_str = file.split("_")[-1].split(".")[0]
            try:
                epoch = int(epoch_str)
                model_files.append((epoch, file))
            except ValueError:
                continue
    
    if not model_files:
        print(f"Error: No model files found for {algorithm} in {model_dir}")
        return None, None
    
    # Get the highest epoch model
    model_files.sort(key=lambda x: x[0], reverse=True)
    epoch, model_file = model_files[0]
    model_path = os.path.join(model_dir, model_file)
    print(f"Using model: {model_path}")
    
    # Create command
    eval_out_dir = os.path.join(output_dir, "evaluations", algorithm)
    os.makedirs(eval_out_dir, exist_ok=True)
    
    # Check for and remove default video directory symlink
    default_video_dir = "cvpo_carla_results"
    if os.path.exists(default_video_dir):
        try:
            if os.path.islink(default_video_dir):
                os.unlink(default_video_dir)
            elif os.path.isdir(default_video_dir):
                backup_dir = f"{default_video_dir}_backup_{int(time.time())}"
                print(f"Moving existing {default_video_dir} to {backup_dir}")
                os.rename(default_video_dir, backup_dir)
        except Exception as e:
            print(f"Warning: Could not handle existing video directory: {e}")
    
    cmd = [
        "python", "run_carla_comparison.py",
        "--policy", algorithm,
        "--mode", "eval",
        "--town", args.town,
        "--seed", str(args.seed),
        "--resume", model_path,
        "--num_vehicles", str(args.num_vehicles),
        "--desired_speed", str(args.desired_speed),
        "--lane_threshold", str(args.lane_threshold),
        "--output_dir", eval_out_dir,
        "--device", args.device,
        "--eval_episodes", str(args.eval_episodes)
    ]
    
    # Print command
    print(f"\n=== Evaluating {algorithm} ===")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command and capture output
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull, universal_newlines=True)
    
    # Store output
    output = []
    for line in process.stdout:
        print(line, end='')
        output.append(line)
    
    process.wait()
    
    # Check result
    if process.returncode != 0:
        print(f"Error: Evaluation of {algorithm} failed with return code {process.returncode}")
        return None, output
    
    # Wait a moment for any file operations to complete
    time.sleep(2)
    
    # Look for the evaluation videos in the correct directory structure
    print("\nLooking for evaluation videos...")
    
    # Check in common locations based on where run_carla_comparison.py saves videos
    video_locations = [
        os.path.join(eval_out_dir, "eval_videos", f"{algorithm}_eval_videos"),  # New preferred location
        os.path.join(eval_out_dir, "eval_videos"),  # Fallback location
        os.path.join(eval_out_dir, "cvpo_carla_results"),  # Default from train_cvpo_carla.py
        "cvpo_carla_results"  # Current directory fallback
    ]
    
    all_video_files = []
    for location in video_locations:
        if os.path.exists(location):
            print(f"Checking for videos in: {location}")
            video_files = []
            for root, _, files in os.walk(location):
                for file in files:
                    if file.endswith('.mp4'):
                        full_path = os.path.join(root, file)
                        video_files.append(full_path)
                        all_video_files.append(full_path)
            
            if video_files:
                print(f"Found {len(video_files)} videos in {location}: {[os.path.basename(v) for v in video_files]}")
    
    if not all_video_files:
        print("WARNING: No evaluation videos found in any location!")
    
    # Parse evaluation results
    results = parse_evaluation_results(output, algorithm)
    
    # Copy all found videos to the comparison videos directory
    videos_dir = os.path.join(output_dir, "comparison_videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    copied_videos = []
    for video_path in all_video_files:
        target_name = f"{algorithm}_{os.path.basename(video_path)}"
        target_path = os.path.join(videos_dir, target_name)
        
        try:
            shutil.copy2(video_path, target_path)
            copied_videos.append(target_path)
            print(f"Copied video to comparison directory: {target_path}")
        except Exception as e:
            print(f"Error copying video {video_path}: {e}")
    
    if copied_videos:
        results['videos'] = copied_videos
    else:
        print(f"Warning: No evaluation videos were copied for {algorithm}")
        results['videos'] = []
    
    return results, output

def parse_evaluation_results(output, algorithm):
    """Parse evaluation results from output with enhanced safety metrics"""
    results = {
        'algorithm': algorithm,
        'reward': None,
        'length': None,
        'episodes': [],
        'videos': [],
        # Add safety metrics tracking
        'safety_metrics': {
            'total_collisions': 0,
            'total_lane_violations': 0,
            'total_speed_violations': 0,
            'cost_threshold_violations': 0,
            'safe_episodes': 0
        }
    }
    
    # Join all output
    output_text = ''.join(output)
    
    # Extract average reward and length
    reward_match = re.search(r'Average reward:\s*([\d\.]+)', output_text)
    length_match = re.search(r'Average episode length:\s*([\d\.]+)', output_text)
    
    if reward_match:
        results['reward'] = float(reward_match.group(1))
    if length_match:
        results['length'] = float(length_match.group(1))
    
    # Extract individual episode data
    episode_pattern = r'Evaluation Episode (\d+) completed:\s*Reward: ([\d\.]+), Length: (\d+)\s*Encountered (\d+) curves\s*Lane positioning issues: (\d+)\s*Safe speeds in curves: (\d+)\s*Lane discipline: ([\d\.]+)% of time in good lane position'
    
    # Extract safety-related patterns
    collision_pattern = r'🚨 Collision detected!'
    lane_pattern = r'⚠️ Lane departure!'
    speed_pattern = r'🏎️ Speed violation!'
    safety_limit_pattern = r'Safety cost threshold exceeded'
    
    # Count overall safety violations
    results['safety_metrics']['total_collisions'] = len(re.findall(collision_pattern, output_text))
    results['safety_metrics']['total_lane_violations'] = len(re.findall(lane_pattern, output_text))
    results['safety_metrics']['total_speed_violations'] = len(re.findall(speed_pattern, output_text))
    results['safety_metrics']['cost_threshold_violations'] = len(re.findall(safety_limit_pattern, output_text))
    
    # Extract cumulative costs for episodes
    cost_pattern = r'Cumulative Cost: ([\d\.]+)'
    costs = re.findall(cost_pattern, output_text)
    
    # Process individual episodes
    for match in re.finditer(episode_pattern, output_text):
        episode_num = int(match.group(1))
        reward = float(match.group(2))
        length = int(match.group(3))
        curves = int(match.group(4))
        lane_issues = int(match.group(5))
        safe_curves = int(match.group(6))
        lane_discipline = float(match.group(7))
        
        # Create episode data structure
        episode = {
            'number': episode_num,
            'reward': reward,
            'length': length,
            'curves': curves,
            'lane_issues': lane_issues,
            'safe_curves': safe_curves,
            'lane_discipline': lane_discipline,
            'safety_metrics': {
                'collisions': 0,
                'lane_violations': 0,
                'speed_violations': 0
            }
        }
        
        # Extract safety costs if available
        episode_cost_pattern = rf"Episode {episode_num}.*?Cumulative Cost: ([\d\.]+)"
        cost_match = re.search(episode_cost_pattern, output_text)
        if cost_match:
            episode['cumulative_cost'] = float(cost_match.group(1))
        
        # Check for safety limit exceeded
        if re.search(rf"Episode {episode_num}.*?{safety_limit_pattern}", output_text):
            episode['safety_limit_exceeded'] = True
        
        # Count episode-specific violations by searching in nearby context
        episode_sections = re.findall(rf"Episode {episode_num}.*?(?:Episode {episode_num+1}|$)", output_text, re.DOTALL)
        if episode_sections:
            section = episode_sections[0]
            episode['safety_metrics']['collisions'] = len(re.findall(collision_pattern, section))
            episode['safety_metrics']['lane_violations'] = len(re.findall(lane_pattern, section))
            episode['safety_metrics']['speed_violations'] = len(re.findall(speed_pattern, section))
        
        # Check if episode was safe (no violations)
        if (episode['safety_metrics']['collisions'] == 0 and
            episode['safety_metrics']['lane_violations'] == 0 and
            episode['safety_metrics']['speed_violations'] == 0 and
            not episode.get('safety_limit_exceeded', False)):
            results['safety_metrics']['safe_episodes'] += 1
            episode['is_safe'] = True
        
        results['episodes'].append(episode)
    
    # Extract video info
    video_match = re.search(r'Evaluation video saved to: ([^\s]+)', output_text)
    if video_match:
        results['videos'].append(video_match.group(1))
    
    return results

def copy_evaluation_videos(source_dir, target_dir, algorithm):
    """Copy evaluation videos to comparison directory"""
    os.makedirs(target_dir, exist_ok=True)
    copied_videos = []
    
    # Look for MP4 files in the source directory and its subdirectories
    print(f"Looking for videos in {source_dir} and subdirectories...")
    video_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    print(f"Found {len(video_paths)} videos: {video_paths}")
    
    # Copy the videos
    for source_path in video_paths:
        target_path = os.path.join(target_dir, f"{algorithm}_{os.path.basename(source_path)}")
        
        try:
            shutil.copy2(source_path, target_path)
            copied_videos.append(target_path)
            print(f"Copied evaluation video: {target_path}")
        except Exception as e:
            print(f"Error copying video {source_path}: {e}")
    
    return copied_videos

def create_safety_comparison_plots(results, output_dir):
    """Create safety-focused comparison plots"""
    if not results:
        print("No results to plot")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract algorithms
    algorithms = [r['algorithm'] for r in results]
    
    # Create safety violations comparison plot
    plt.figure(figsize=(14, 10))
    
    # 1. Safety violations by type
    plt.subplot(2, 2, 1)
    
    # Extract violation counts
    collisions = [r['safety_metrics']['total_collisions'] for r in results]
    lane_violations = [r['safety_metrics']['total_lane_violations'] for r in results]
    speed_violations = [r['safety_metrics']['total_speed_violations'] for r in results]
    
    # Create grouped bar chart
    x = np.arange(len(algorithms))
    width = 0.25
    
    plt.bar(x - width, collisions, width, label='Collisions', color='red')
    plt.bar(x, lane_violations, width, label='Lane Violations', color='orange')
    plt.bar(x + width, speed_violations, width, label='Speed Violations', color='blue')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Total Violations')
    plt.title('Safety Violations by Type')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.grid(axis='y')
    
    # 2. Safety costs comparison
    plt.subplot(2, 2, 2)
    
    # Extract costs from episodes
    costs_by_alg = []
    for result in results:
        costs = [e.get('cumulative_cost', 0) for e in result['episodes'] if 'cumulative_cost' in e]
        if costs:
            costs_by_alg.append(costs)
        else:
            # If no costs available, add empty list or dummy value
            costs_by_alg.append([0])
    
    # Create boxplot if we have cost data
    if any(costs for costs in costs_by_alg):
        plt.boxplot(costs_by_alg, labels=algorithms)
        plt.ylabel('Cumulative Safety Cost')
        plt.title('Safety Cost Distribution')
        plt.grid(axis='y')
    
    # 3. Safe episodes percentage
    plt.subplot(2, 2, 3)
    
    # Calculate percentage of safe episodes
    safe_percentages = []
    cost_violation_percentages = []
    
    for result in results:
        total_episodes = len(result['episodes'])
        if total_episodes > 0:
            safe_pct = 100 * result['safety_metrics']['safe_episodes'] / total_episodes
            cost_violations = result['safety_metrics']['cost_threshold_violations']
            cost_viol_pct = 100 * cost_violations / total_episodes
        else:
            safe_pct = 0
            cost_viol_pct = 0
            
        safe_percentages.append(safe_pct)
        cost_violation_percentages.append(cost_viol_pct)
    
    # Create grouped bar chart
    x = np.arange(len(algorithms))
    width = 0.35
    
    plt.bar(x - width/2, safe_percentages, width, label='Safe Episodes (%)', color='green')
    plt.bar(x + width/2, cost_violation_percentages, width, label='Cost Threshold Violations (%)', color='red')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Percentage')
    plt.title('Safety Success vs. Threshold Violations')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.grid(axis='y')
    
    # 4. Curve safety by algorithm
    plt.subplot(2, 2, 4)
    
    # Calculate curve safety percentage
    curve_percentages = []
    
    for result in results:
        total_curves = sum(e['curves'] for e in result['episodes'])
        total_safe_curves = sum(e['safe_curves'] for e in result['episodes'])
        
        if total_curves > 0:
            curve_pct = 100 * total_safe_curves / total_curves
        else:
            curve_pct = 0
            
        curve_percentages.append(curve_pct)
    
    plt.bar(algorithms, curve_percentages, color='purple')
    plt.ylabel('Safe Curve Handling (%)')
    plt.title('Curve Safety Comparison')
    plt.grid(axis='y')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "safety_comparison.png"))
    
    print(f"Safety comparison plots saved to {plots_dir}")

def create_comparison_plots(results, output_dir):
    """Create comparison plots for metrics"""
    if not results:
        print("No results to plot")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract algorithms and metrics
    algorithms = [r['algorithm'] for r in results]
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    
    # Episode-level data
    episode_data = {}
    for result in results:
        if result['episodes']:
            alg = result['algorithm']
            episode_data[alg] = {
                'rewards': [e['reward'] for e in result['episodes']],
                'lengths': [e['length'] for e in result['episodes']],
                'lane_discipline': [e['lane_discipline'] for e in result['episodes']],
                'curves': [e['curves'] for e in result['episodes']],
                'lane_issues': [e['lane_issues'] for e in result['episodes']],
                'safe_curves': [e['safe_curves'] for e in result['episodes']]
            }
    
    # Create bar charts for average metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, rewards, color='blue')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison')
    plt.grid(axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(algorithms, lengths, color='green')
    plt.ylabel('Average Episode Length')
    plt.title('Episode Length Comparison')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_length_comparison.png"))
    
    # Create detailed plots if episode data is available
    if episode_data:
        # Boxplot of episode rewards
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        boxplot_data = [episode_data[alg]['rewards'] for alg in episode_data.keys()]
        plt.boxplot(boxplot_data, labels=list(episode_data.keys()))
        plt.ylabel('Episode Reward')
        plt.title('Episode Reward Distribution')
        plt.grid(axis='y')
        
        # Boxplot of lane discipline
        plt.subplot(2, 2, 2)
        boxplot_data = [episode_data[alg]['lane_discipline'] for alg in episode_data.keys()]
        plt.boxplot(boxplot_data, labels=list(episode_data.keys()))
        plt.ylabel('Lane Discipline (%)')
        plt.title('Lane Discipline Distribution')
        plt.grid(axis='y')
        
        # Bar chart of average lane issues
        plt.subplot(2, 2, 3)
        avg_lane_issues = [np.mean(episode_data[alg]['lane_issues']) for alg in episode_data.keys()]
        plt.bar(list(episode_data.keys()), avg_lane_issues, color='orange')
        plt.ylabel('Average Lane Issues')
        plt.title('Lane Issues Comparison')
        plt.grid(axis='y')
        
        # Bar chart of curve handling
        if any(np.sum(episode_data[alg]['curves']) > 0 for alg in episode_data.keys()):
            plt.subplot(2, 2, 4)
            
            # Calculate percentage of safe curve handling for each algorithm
            curve_safety = []
            for alg in episode_data.keys():
                total_curves = np.sum(episode_data[alg]['curves'])
                if total_curves > 0:
                    safe_pct = 100 * np.sum(episode_data[alg]['safe_curves']) / total_curves
                else:
                    safe_pct = 0
                curve_safety.append(safe_pct)
            
            plt.bar(list(episode_data.keys()), curve_safety, color='purple')
            plt.ylabel('Safe Curve Handling (%)')
            plt.title('Curve Safety Comparison')
            plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "detailed_comparison.png"))
    
    print(f"Comparison plots saved to {plots_dir}")

def create_comparison_report(results, output_dir):
    """Create a text report summarizing comparison results with enhanced safety metrics"""
    if not results:
        return
    
    report_path = os.path.join(output_dir, "comparison_results.txt")
    
    with open(report_path, 'w') as f:
        f.write("===== CARLA ALGORITHM COMPARISON RESULTS =====\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of algorithms compared: {len(results)}\n\n")
        
        # Write overview table with added safety metrics
        f.write("OVERVIEW:\n")
        f.write(f"{'Algorithm':<12} {'Avg Reward':<12} {'Avg Length':<12} {'Episodes':<10} {'Safe Episodes':<15} {'Collisions':<12} {'Lane Viol.':<12} {'Speed Viol.':<12}\n")
        f.write("-" * 100 + "\n")
        
        for result in results:
            alg = result['algorithm']
            reward = result['reward'] if result['reward'] is not None else "N/A"
            length = result['length'] if result['length'] is not None else "N/A"
            episodes = len(result['episodes'])
            
            # Extract safety metrics
            safe_episodes = result['safety_metrics']['safe_episodes']
            safe_pct = f"{safe_episodes}/{episodes} ({100*safe_episodes/episodes:.1f}%)" if episodes > 0 else "N/A"
            
            collisions = result['safety_metrics']['total_collisions']
            lane_viol = result['safety_metrics']['total_lane_violations']
            speed_viol = result['safety_metrics']['total_speed_violations']
            
            f.write(f"{alg:<12} {reward:<12.2f} {length:<12.2f} {episodes:<10} {safe_pct:<15} {collisions:<12} {lane_viol:<12} {speed_viol:<12}\n")
        
        f.write("\n\n")
        
        # Added safety summary table
        f.write("SAFETY METRICS SUMMARY:\n")
        f.write(f"{'Algorithm':<12} {'Cost Thresh. Viol.':<20} {'Curve Safety %':<15} {'Lane Discipline %':<18}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            alg = result['algorithm']
            cost_viol = result['safety_metrics']['cost_threshold_violations']
            episodes = len(result['episodes'])
            
            # Calculate curve safety
            total_curves = sum(e['curves'] for e in result['episodes'])
            total_safe_curves = sum(e['safe_curves'] for e in result['episodes'])
            curve_safety = f"{total_safe_curves}/{total_curves} ({100*total_safe_curves/total_curves:.1f}%)" if total_curves > 0 else "N/A"
            
            # Calculate average lane discipline
            avg_lane_discipline = np.mean([e['lane_discipline'] for e in result['episodes']]) if result['episodes'] else 0
            
            # Format cost violation percentage
            cost_viol_str = f"{cost_viol}/{episodes} ({100*cost_viol/episodes:.1f}%)" if episodes > 0 else "N/A"
            
            f.write(f"{alg:<12} {cost_viol_str:<20} {curve_safety:<15} {avg_lane_discipline:<18.2f}\n")
        
        f.write("\n\n")
        
        # Write detailed per-algorithm information (keep original but add safety section)
        f.write("DETAILED RESULTS:\n\n")
        
        for result in results:
            alg = result['algorithm']
            f.write(f"=== {alg} ===\n")
            
            if result['reward'] is not None:
                f.write(f"Average reward: {result['reward']:.2f}\n")
            if result['length'] is not None:
                f.write(f"Average episode length: {result['length']:.2f}\n")
            
            # Add safety metrics summary
            f.write("\nSafety metrics:\n")
            f.write(f"  Total collisions: {result['safety_metrics']['total_collisions']}\n")
            f.write(f"  Total lane violations: {result['safety_metrics']['total_lane_violations']}\n")
            f.write(f"  Total speed violations: {result['safety_metrics']['total_speed_violations']}\n")
            f.write(f"  Cost threshold violations: {result['safety_metrics']['cost_threshold_violations']}\n")
            
            safe_episodes = result['safety_metrics']['safe_episodes']
            episodes = len(result['episodes'])
            if episodes > 0:
                f.write(f"  Safe episodes: {safe_episodes}/{episodes} ({100*safe_episodes/episodes:.1f}%)\n")
            
            # Write episode details
            if result['episodes']:
                f.write("\nEpisode details:\n")
                
                total_curves = sum(e['curves'] for e in result['episodes'])
                total_safe_curves = sum(e['safe_curves'] for e in result['episodes'])
                curve_safety = (total_safe_curves / total_curves * 100) if total_curves > 0 else 0
                
                avg_lane_discipline = np.mean([e['lane_discipline'] for e in result['episodes']])
                
                f.write(f"  Episodes completed: {len(result['episodes'])}\n")
                f.write(f"  Total curves encountered: {total_curves}\n")
                f.write(f"  Curve safety percentage: {curve_safety:.1f}%\n")
                f.write(f"  Average lane discipline: {avg_lane_discipline:.1f}%\n\n")
                
                # Write per-episode data with safety info
                f.write("  Per-episode results:\n")
                f.write(f"  {'Episode':<8} {'Reward':<10} {'Length':<10} {'Curves':<8} {'Safe Curves':<12} {'Lane Disc %':<10} {'Safety Cost':<12} {'Safety Issues':<15}\n")
                f.write("  " + "-" * 90 + "\n")
                
                for episode in result['episodes']:
                    # Extract safety issues
                    collisions = episode.get('safety_metrics', {}).get('collisions', 0)
                    lane_viols = episode.get('safety_metrics', {}).get('lane_violations', 0)
                    speed_viols = episode.get('safety_metrics', {}).get('speed_violations', 0)
                    safety_issues = []
                    if collisions > 0: safety_issues.append(f"{collisions} coll.")
                    if lane_viols > 0: safety_issues.append(f"{lane_viols} lane")
                    if speed_viols > 0: safety_issues.append(f"{speed_viols} speed")
                    if episode.get('safety_limit_exceeded', False): safety_issues.append("limit exceeded")
                    
                    safety_issues_str = ", ".join(safety_issues) if safety_issues else "None"
                    safety_cost = episode.get('cumulative_cost', 'N/A')
                    
                    f.write(f"  {episode['number']:<8} {episode['reward']:<10.1f} {episode['length']:<10} "
                           f"{episode['curves']:<8} {episode['safe_curves']:<12} {episode['lane_discipline']:<10.1f} ")
                    if isinstance(safety_cost, (int, float)):
                        f.write(f"{safety_cost:<12.1f} ")
                    else:
                        f.write(f"{safety_cost:<12} ")
                    f.write(f"{safety_issues_str:<15}\n")
            
            # List videos
            if result['videos']:
                f.write("\nEvaluation videos:\n")
                for video in result['videos']:
                    f.write(f"  {os.path.basename(video)}\n")
            
            f.write("\n\n")
    
    print(f"Enhanced comparison report with safety metrics saved to {report_path}")

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create timestamp for this comparison
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting algorithm comparison ===")
    print(f"Algorithms: {args.algorithms}")
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "comparison_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Run training for each algorithm
    results = {}
    for algorithm in args.algorithms:
        success, _ = run_algorithm(algorithm, args, output_dir)
        results[algorithm] = "SUCCESS" if success else "FAILED"
    
    # Evaluate algorithms if requested
    eval_results = []
    if args.eval_after:
        eval_dir = os.path.join(output_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        
        for algorithm in args.algorithms:
            if results[algorithm] == "SUCCESS":
                print(f"\nEvaluating {algorithm}...")
                result, _ = evaluate_algorithm(algorithm, args, output_dir)
                if result:
                    eval_results.append(result)
        
        # Create comparison plots and report
        if eval_results:
            create_comparison_plots(eval_results, output_dir)
            create_safety_comparison_plots(eval_results, output_dir)  # Add safety plots
            create_comparison_report(eval_results, output_dir)
    
    # Print summary
    print("\n=== Comparison Summary ===")
    for algorithm, status in results.items():
        print(f"{algorithm}: {status}")
    
    print(f"\nResults saved to {output_dir}")
    
    # If evaluations were performed, show the results locations
    if args.eval_after and eval_results:
        print("\nComparison outputs:")
        print(f"- Enhanced report with safety metrics: {os.path.join(output_dir, 'comparison_results.txt')}")
        print(f"- Performance plots: {os.path.join(output_dir, 'plots')}")
        print(f"- Safety comparison plots: {os.path.join(output_dir, 'plots', 'safety_comparison.png')}")
        print(f"- Videos: {os.path.join(output_dir, 'comparison_videos')}")
        
        # If plots were created, try to display the paths
        plots_dir = os.path.join(output_dir, "plots")
        if os.path.exists(plots_dir) and os.path.isdir(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            if plot_files:
                print("\nComparison plots:")
                for plot_file in plot_files:
                    print(f"- {os.path.join(plots_dir, plot_file)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
