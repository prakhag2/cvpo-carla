#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import time
from datetime import datetime
import subprocess
import pygame
import matplotlib.pyplot as plt
import traceback

# Set the MODEL_DEVICE environment variable
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["MODEL_DEVICE"] = device
print(f"Using device: {device}")

# Import CVPO and Carla wrapper
from debug_cvpo import DebugCVPO
from carla_wrapper import make_carla_env
from safe_rl.util.logger import EpochLogger

# Create output directory
output_dir = "cvpo_carla_results"
os.makedirs(output_dir, exist_ok=True)

def capture_env_frame(env, frame_path):
    """Capture a frame from the Carla environment"""
    try:
        surface = env.display
        frame_array = pygame.surfarray.array3d(surface)
        plt.imsave(frame_path, frame_array)
        return True
    except Exception as e:
        print(f"Frame capture error: {e}")
        return False

def create_video_from_frames(frame_dir, output_path, fps=10):
    """Create a video from frames using matplotlib animation"""
    import glob
    from matplotlib import animation

    try:
        # Find all frame files
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        
        if not frame_files:
            print(f"No frames found in {frame_dir} to create video")
            return False
        
        print(f"Creating video from {len(frame_files)} frames...")
        
        # Load frames
        frames = []
        for frame_file in frame_files:
            img = plt.imread(frame_file)
            frames.append(img)
        
        # Create animation
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        im = plt.imshow(frames[0])
        
        def update(i):
            im.set_array(frames[i])
            return [im]
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), 
            interval=1000/fps, blit=True
        )
        
        # Save as mp4
        ani.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        
        print(f"Video saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        traceback.print_exc()
        return False

def save_model(agent, filepath):
    """Save the model to disk"""
    try:
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'qc': agent.qc.state_dict() if hasattr(agent, 'qc') else None
        }, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(agent, filepath):
    """Load a model from disk"""
    try:
        checkpoint = torch.load(filepath)
        if 'actor' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor'])
        if 'critic' in checkpoint:
            agent.critic.load_state_dict(checkpoint['critic'])
        if 'qc' in checkpoint and hasattr(agent, 'qc'):
            agent.qc.load_state_dict(checkpoint['qc'])
        print(f"Model loaded from {filepath}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def collect_experience(env, agent, max_steps=1000, buffer=None, capture_video=False, verbose=True, epoch=0):
    """Collect experience with the agent in the environment"""
    total_steps = 0
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0
    episode_length = 0
    
    # Metrics for enhanced logging
    curve_encounters = 0
    lane_violations = 0
    high_speed_in_curves = 0
    smooth_steering_count = 0
    
    # Video capture setup
    frames_captured = 0
    if capture_video:
        # Use unique directory for each epoch/episode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_dir = os.path.join(output_dir, f"frames_epoch{epoch}_ep{episode_count}_{timestamp}")
        os.makedirs(frame_dir, exist_ok=True)
        print(f"\nCapturing video frames to {frame_dir}")
    
    obs = env.reset()
    done = False
    
    while total_steps < max_steps:
        action, _ = agent.act(obs)
        
        # Add exploration noise occasionally to force learning
        if np.random.random() < 0.2:  # 20% random exploration
            noise = np.random.normal(0, 0.3, size=action.shape)
            action = action + noise
        
        # Capture frame if requested
        if capture_video and total_steps % 4 == 0:
            frame_path = os.path.join(frame_dir, f"step_{total_steps:04d}.png")
            success = capture_env_frame(env, frame_path)
            if success:
                frames_captured += 1
                if frames_captured % 50 == 0:
                    print(f"  Captured {frames_captured} frames so far...")
        
        next_obs, reward, done, info = env.step(action)
        
        # Get curve status from info dictionary (consistent with reward calculation)
        is_curve = info.get('is_curve', False)
        curve_direction = info.get('curve_direction', None)
        
        # If curve detected, increment counter
        if is_curve:
            curve_encounters += 1
            
        # Get speed information
        current_speed = 0
        try:
            if hasattr(env.env, 'ego'):
                v = env.env.ego.get_velocity()
                current_speed = np.sqrt(v.x**2 + v.y**2)
        except Exception as e:
            pass
        
        # Get lane position
        lane_pos = 0
        try:
            if hasattr(env.env, 'ego') and hasattr(env.env, 'waypoints'):
                from gym_carla.envs.misc import get_pos, get_lane_dis
                ego_x, ego_y = get_pos(env.env.ego)
                lane_pos, _ = get_lane_dis(env.env.waypoints, ego_x, ego_y)
        except Exception as e:
            pass
            
        # Detailed road status debugging every 20 steps
        if total_steps % 20 == 0 and hasattr(env.env, 'waypoints'):
            print(f"\nWaypoints at step {total_steps}:")
            for i, wp in enumerate(env.env.waypoints[:5]):  # Show first 5 waypoints
                print(f"  Waypoint {i}: {wp}")
                
            # Calculate angles between waypoints to detect curves
            if len(env.env.waypoints) >= 3:
                angles = []
                for i in range(len(env.env.waypoints) - 2):
                    if i >= 3:  # Only check first few waypoints
                        break
                    p1 = env.env.waypoints[i]
                    p2 = env.env.waypoints[i+1]
                    p3 = env.env.waypoints[i+2]
                    
                    if len(p1) >= 2 and len(p2) >= 2 and len(p3) >= 2:
                        # Calculate vectors
                        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                        
                        # Normalize vectors
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        
                        if v1_norm > 0 and v2_norm > 0:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm
                            
                            # Calculate dot product and angle
                            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                            angle_rad = np.arccos(dot_product)
                            angle_deg = np.degrees(angle_rad)
                            angles.append(angle_deg)
                            
                            cross_product = v1[0]*v2[1] - v1[1]*v2[0]
                            turn_direction = "right" if cross_product < 0 else "left"
                            
                            print(f"  Segment {i} to {i+2}: angle = {angle_deg:.2f}° ({turn_direction} turn)")
                
                avg_angle = np.mean(angles) if angles else 0
                print(f"  Average angle change: {avg_angle:.2f}° (>4° indicates curve)")
                print(f"  Car position: ({ego_x:.2f}, {ego_y:.2f})")
                print(f"  Lane position: {lane_pos:.3f} (threshold: {getattr(env.env, 'out_lane_thres', 'N/A')})")
        
        # Describe the current situation and action in natural language
        if verbose and total_steps % 20 == 0:
            # Describe road situation with more details for better tracking
            if is_curve:
                if curve_direction:
                    road_desc = f"{curve_direction.capitalize()} curve"
                else:
                    road_desc = "Curve"
                road_desc += f" (step {total_steps})"
            else:
                road_desc = f"Straight road (step {total_steps})"
            
            # Describe car position with more details
            if abs(lane_pos) < 0.3:
                position_desc = "perfectly centered"
            elif abs(lane_pos) < 0.8:
                side = "right" if lane_pos > 0 else "left"
                position_desc = f"slightly to the {side} (pos={lane_pos:.2f})"
            else:
                side = "right" if lane_pos > 0 else "left"
                severity = "significantly" if abs(lane_pos) < 1.5 else "far"
                position_desc = f"{severity} to the {side} (pos={lane_pos:.2f})"
                
            # Describe speed
            speed_desc = "maintaining speed"
            if abs(action[0]) > 0.3:
                speed_desc = "accelerating" if action[0] > 0 else "braking"
                intensity = "gently" if abs(action[0]) < 1.0 else "strongly" if abs(action[0]) < 2.0 else "aggressively"
                speed_desc = f"{intensity} {speed_desc} ({action[0]:.2f})"
                
            # Describe steering
            steering_desc = "driving straight"
            if abs(action[1]) > 0.1:
                direction = "right" if action[1] > 0 else "left"
                intensity = "slightly" if abs(action[1]) < 0.3 else "moderately" if abs(action[1]) < 0.6 else "sharply"
                steering_desc = f"turning {intensity} {direction} ({action[1]:.2f})"
                
            # Combine into a situation description
            print(f"\nStep {total_steps}:")
            print(f"  Situation: {road_desc}. Car is {position_desc}.")
            print(f"  Action: {steering_desc}, {speed_desc}. Speed: {current_speed:.1f}.")
        
        # Get detailed reward components if available
        if 'reward_debug' in info:
            reward_components = info['reward_debug']
            
            # Log reward components occasionally
            if verbose and total_steps % 20 == 0:
                print("  Reward components:")
                for comp_name, comp_value in reward_components.items():
                    print(f"    {comp_name}: {comp_value:.3f}")
                    
                # Check for violations/behaviors
                if 'lane_reward' in reward_components and reward_components['lane_reward'] < 1.0:
                    lane_violations += 1
                    print("    ⚠️ Poor lane positioning detected!")
                    
                if 'curve_speed_penalty' in reward_components and reward_components['curve_speed_penalty'] < -0.2:
                    high_speed_in_curves += 1
                    print("    ⚠️ Too fast for curve!")
                    
                if is_curve and 'curve_anticipation' in reward_components and reward_components['curve_anticipation'] > 0:
                    print(f"    ✓ Good curve anticipation - turning properly for {curve_direction} curve")
                
                if 'steering_smoothness' in reward_components and reward_components['steering_smoothness'] > -0.1:
                    smooth_steering_count += 1
                    print("    ✓ Smooth steering control")
                
                if 'stationary_penalty' in reward_components and reward_components['stationary_penalty'] < 0:
                    print("    ⚠️ Car not moving enough! Acceleration needed.")
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        total_steps += 1
        
        # Store experience in buffer if provided
        if buffer is not None:
            buffer['obs'].append(obs.copy())
            buffer['act'].append(action.copy())
            buffer['rew'].append(reward)
            buffer['next_obs'].append(next_obs.copy())
            buffer['done'].append(done)
            buffer['cost'].append(info.get('cost', 0))
        
        # Check if episode ended
        if done or episode_length >= max_steps:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            print(f"\nEpisode {episode_count} completed - Length: {episode_length}, Reward: {episode_reward:.2f}")
            
            # Print episode summary statistics
            if verbose:
                print(f"  Episode statistics:")
                print(f"    Curve encounters: {curve_encounters}")
                print(f"    Lane positioning issues: {lane_violations}")
                print(f"    Speed issues in curves: {high_speed_in_curves}")
                print(f"    Smooth steering instances: {smooth_steering_count}")
                
                # Reset counters for next episode
                curve_encounters = 0
                lane_violations = 0
                high_speed_in_curves = 0
                smooth_steering_count = 0
            
            # Create video if frames were captured
            if capture_video and frames_captured > 0:
                video_path = os.path.join(output_dir, f"epoch{epoch}_episode_{episode_count}.mp4")
                success = create_video_from_frames(
                    frame_dir, 
                    video_path
                )
                if success:
                    print(f"Training video saved to: {video_path}")
                
                # Create a new frame directory for the next episode
                if total_steps < max_steps:  # If we're continuing to another episode
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_dir = os.path.join(output_dir, f"frames_epoch{epoch}_ep{episode_count+1}_{timestamp}")
                    os.makedirs(frame_dir, exist_ok=True)
                    print(f"\nCapturing video frames to {frame_dir}")
                    frames_captured = 0
            
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs
    
    print(f"Completed {episode_count} episodes, {total_steps} total steps")
    
    return {
        'total_steps': total_steps,
        'episode_count': episode_count,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'average_reward': np.mean(episode_rewards),
        'average_length': np.mean(episode_lengths)
    }

def evaluate_policy(env, agent, num_episodes=3, capture_video=True, epoch=0):
    """Evaluate the current policy without exploration"""
    print(f"\n=== Evaluating policy for {num_episodes} episodes ===")
    
    total_rewards = []
    episode_lengths = []
    
    # Setup video capture if needed
    frames_captured = 0
    if capture_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_frame_dir = os.path.join(output_dir, f"eval_frames_epoch{epoch}_{timestamp}")
        os.makedirs(eval_frame_dir, exist_ok=True)
        print(f"Capturing evaluation frames to {eval_frame_dir}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Metrics for reporting
        curve_count = 0
        lane_deviations = 0
        safe_speed_count = 0
        
        while not done and steps < 1000:
            # Use deterministic actions (no exploration)
            action, _ = agent.act(obs, deterministic=True)
            
            # Capture frame if requested
            if capture_video and steps % 4 == 0:
                frame_path = os.path.join(eval_frame_dir, f"eval_ep{episode}_step_{steps:04d}.png")
                success = capture_env_frame(env, frame_path)
                if success:
                    frames_captured += 1
                    if frames_captured % 50 == 0:
                        print(f"  Captured {frames_captured} evaluation frames...")
            
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Get road status and situation information
            is_curve = False
            curve_direction = None
            try:
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                    
                if hasattr(base_env, '_is_on_curve'):
                    is_curve = base_env._is_on_curve()
                    if is_curve:
                        curve_count += 1
                    
                    # Determine curve direction if on a curve
                    if is_curve and hasattr(base_env, 'waypoints') and len(base_env.waypoints) >= 3:
                        wp1 = base_env.waypoints[0]
                        wp3 = base_env.waypoints[2]
                        
                        if len(wp1) >= 2 and len(wp3) >= 2:
                            dx = wp3[0] - wp1[0]
                            curve_direction = "left" if dx < 0 else "right"
            except:
                pass
                
            # Get speed
            current_speed = 0
            try:
                if hasattr(base_env, 'ego'):
                    v = base_env.ego.get_velocity()
                    current_speed = np.sqrt(v.x**2 + v.y**2)
            except:
                pass
                
            # Get lane position
            lane_pos = 0
            try:
                if hasattr(base_env, 'ego') and hasattr(base_env, 'waypoints'):
                    from gym_carla.envs.misc import get_pos, get_lane_dis
                    ego_x, ego_y = get_pos(base_env.ego)
                    lane_pos, _ = get_lane_dis(base_env.waypoints, ego_x, ego_y)
                    if abs(lane_pos) > 0.5:
                        lane_deviations += 1
            except:
                pass
                
            # Check for safe speeds in curves
            if is_curve and hasattr(base_env, 'desired_speed'):
                desired_speed = base_env.desired_speed
                if current_speed <= desired_speed * 0.7:  # Safe curve speed
                    safe_speed_count += 1
            
            # Log situational description every 50 steps
            if steps % 50 == 0:
                # Describe road situation
                if is_curve:
                    if curve_direction:
                        road_desc = f"{curve_direction.capitalize()} curve"
                    else:
                        road_desc = "Curve"
                    road_desc += f" (step {steps})"
                else:
                    road_desc = f"Straight road (step {steps})"
                
                # Describe car position with more details
                if abs(lane_pos) < 0.3:
                    position_desc = "perfectly centered"
                elif abs(lane_pos) < 0.8:
                    side = "right" if lane_pos > 0 else "left"
                    position_desc = f"slightly to the {side} (pos={lane_pos:.2f})"
                else:
                    side = "right" if lane_pos > 0 else "left"
                    severity = "significantly" if abs(lane_pos) < 1.5 else "far"
                    position_desc = f"{severity} to the {side} (pos={lane_pos:.2f})"
                    
                # Describe speed
                speed_desc = "maintaining speed"
                if abs(action[0]) > 0.3:
                    speed_desc = "accelerating" if action[0] > 0 else "braking"
                    intensity = "gently" if abs(action[0]) < 1.0 else "strongly" if abs(action[0]) < 2.0 else "aggressively"
                    speed_desc = f"{intensity} {speed_desc} ({action[0]:.2f})"
                    
                # Describe steering
                steering_desc = "driving straight"
                if abs(action[1]) > 0.1:
                    direction = "right" if action[1] > 0 else "left"
                    intensity = "slightly" if abs(action[1]) < 0.3 else "moderately" if abs(action[1]) < 0.6 else "sharply"
                    steering_desc = f"turning {intensity} {direction} ({action[1]:.2f})"
                
                print(f"\n  Eval Ep {episode+1}, Step {steps}:")
                print(f"    Situation: {road_desc}. Car is {position_desc}.")
                print(f"    Action: {steering_desc}, {speed_desc}. Speed: {current_speed:.1f}.")
                print(f"    Reward: {reward:.2f}")
                
                # Check reward components
                if 'reward_debug' in info:
                    reward_components = info['reward_debug']
                    if 'curve_speed_penalty' in reward_components and reward_components['curve_speed_penalty'] < -0.2:
                        print(f"    ⚠️ Too fast for curve!")
                    if is_curve and 'curve_anticipation' in reward_components and reward_components['curve_anticipation'] > 0:
                        print(f"    ✓ Good curve anticipation")
            
            obs = next_obs
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Enhanced episode summary
        print(f"\nEvaluation Episode {episode+1} completed:")
        print(f"  Reward: {episode_reward:.2f}, Length: {steps}")
        print(f"  Encountered {curve_count} curves")
        print(f"  Lane positioning issues: {lane_deviations}")
        print(f"  Safe speeds in curves: {safe_speed_count}")
        
        # Calculate stats
        if curve_count > 0:
            safe_pct = (safe_speed_count / curve_count) * 100
            print(f"  Safety in curves: {safe_pct:.1f}% of curves handled with safe speed")
            
        lane_pct = (lane_deviations / steps) * 100 if steps > 0 else 0
        print(f"  Lane discipline: {100-lane_pct:.1f}% of time in good lane position")
    
    # Create evaluation video if frames were captured
    if capture_video and frames_captured > 0:
        video_path = os.path.join(output_dir, f"evaluation_epoch{epoch}.mp4")
        success = create_video_from_frames(
            eval_frame_dir, 
            video_path
        )
        if success:
            print(f"Evaluation video saved to: {video_path}")
    
    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"Evaluation Results: Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.2f}")
    
    return avg_reward, avg_length

def apply_curriculum(env, epoch):
    """Apply curriculum learning based on training epoch"""
    try:
        base_env = env.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
    
        # Start with simpler scenarios
        if epoch < 20:
            base_env.desired_speed = 3  # Very slow
        elif epoch < 50:
            base_env.desired_speed = 5  # Medium speed
        else:
            base_env.desired_speed = 8  # Target speed
        
        # Adjust lane threshold to be more forgiving in early epochs
        if epoch < 30:
            base_env.out_lane_thres = 3.0  # More forgiving
        elif epoch < 60:
            base_env.out_lane_thres = 2.5  # Medium difficulty
        else:
            base_env.out_lane_thres = 2.0  # Target difficulty
        
        print(f"\nCurriculum - Epoch {epoch+1}:")
        print(f"  Desired speed: {base_env.desired_speed}")
        print(f"  Lane threshold: {base_env.out_lane_thres}")
    except Exception as e:
        print(f"Error applying curriculum: {e}")

def track_policy_evolution(cvpo_agent, env, epoch):
    """Track how policy responds to specific scenarios over training"""
    try:
        # Create standard test scenarios
        test_obs = env.reset()
        
        # 1. Process test observation
        if hasattr(cvpo_agent, '_process_observation'):
            test_obs = cvpo_agent._process_observation(test_obs)
        
        # Get deterministic actions from policy
        action, _ = cvpo_agent.act(test_obs, deterministic=True)
        
        # Get stats on policy's current behavior
        stats = {
            'epoch': epoch,
            'acceleration': action[0],
            'steering': action[1],
            'policy_update_magnitude': getattr(cvpo_agent, 'last_update_magnitude', 0)
        }
        
        print(f"\nPolicy evaluation at epoch {epoch}:")
        print(f"  Standard scenario: acceleration={action[0]:.4f}, steering={action[1]:.4f}")
        
        return stats
    except Exception as e:
        print(f"Error in policy evolution tracking: {e}")
        return None

def plot_policy_evolution(stats_list, output_dir):
    """Plot how policy has evolved over training"""
    if not stats_list:
        return
        
    epochs = [s['epoch'] for s in stats_list]
    accelerations = [s['acceleration'] for s in stats_list]
    steerings = [s['steering'] for s in stats_list]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, accelerations, 'b-o', label='Acceleration')
    plt.xlabel('Epoch')
    plt.ylabel('Acceleration')
    plt.title('Policy Acceleration Evolution')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, steerings, 'r-o', label='Steering')
    plt.xlabel('Epoch')
    plt.ylabel('Steering')
    plt.title('Policy Steering Evolution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_evolution.png'))

def train_cvpo_on_carla(resume_from=None):
    """
    Train CVPO on Carla environment with simplified settings
    
    Args:
        resume_from (str, optional): Path to a checkpoint file to resume training from
    
    Returns:
        tuple: (trained_agent, epoch_metrics, eval_metrics)
    """
    print("\n=== Starting CVPO training on Carla environment ===")
    
    # Configure Carla environment with simplified parameters
    params = {
        'number_of_vehicles': 0,       # No vehicles
        'number_of_walkers': 0,        # No pedestrians
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.5, 0.5],  # Wider range for better turns
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 2000,
        'town': 'Town05',              # Town with good curves
        'task_mode': 'random',
        'max_time_episode': 500,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 3.0,         # More forgiving lane threshold
        'desired_speed': 5,            # Moderate speed
        'max_ego_spawn_times': 200,
        'display_route': True,
        'pixor_size': 64,
        'pixor': False,
    }
    
    # Create environment
    env = make_carla_env(
        params=params, 
        max_episode_steps=500,
        verbose=False
    )
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_kwargs = {
        'output_dir': os.path.join(output_dir, f"run_{timestamp}"),
        'exp_name': 'cvpo_carla'
    }
    logger = EpochLogger(**logger_kwargs)
    
    # Configure CVPO agent with safety constraints
    cvpo_config = {
        "ac_kwargs": {"hidden_sizes": [256, 128]},
        "use_cost_value": True,           # Enable safety constraints
        "steps_per_epoch": 1000,
        "max_ep_len": 500,
        "worker_config": {
            "reset_on_done": True,
            "buffer_size": 100000,
            "warmup_steps": 2000,
        },
        "sample_action_num": 32,
        "mstep_iteration_num": 3,
        "kl_mean_constraint": 0.05,       # Looser KL constraint
        "kl_var_constraint": 0.005,       # Looser variance constraint
        "alpha_mean_scale": 0.5,
        "alpha_var_scale": 50.0,
        "alpha_scale": 10.0,
        "alpha_mean_max": 0.5,
        "alpha_var_max": 50.0,
        "alpha_max": 5.0,
        "pi_lr": 3e-3,                    # Higher learning rate
        "vf_lr": 1e-3,                    # Higher critic learning rate
        "gamma": 0.99,
        "polyak": 0.995,
        "num_q": 2,                       # Used for safety critic
        "num_qc": 2,                      # Used for safety critic
        "cost_limit": 50,                 # Cost threshold for safety
        "use_cost_decay": True,           # Gradually decrease the cost threshold
        "cost_start": 100,                # Initial cost threshold (more lenient)
        "cost_end": 25,                   # Final cost threshold (stricter)
        "decay_epoch": 250,               # How many epochs to decay over
    }
    
    # Create CVPO agent
    print(f"Creating CVPO agent with device: {device}")
    cvpo_agent = DebugCVPO(env, logger, **cvpo_config)
    print("CVPO agent created successfully!")
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if resume_from is not None:
        if os.path.exists(resume_from):
            success = load_model(cvpo_agent, resume_from)
            if success:
                print(f"Successfully resumed training from checkpoint: {resume_from}")
                
                # Extract epoch number from the checkpoint filename if possible
                try:
                    epoch_str = os.path.basename(resume_from)
                    if "epoch_" in epoch_str:
                        start_epoch = int(epoch_str.split("epoch_")[1].split(".")[0])
                        print(f"Starting from epoch {start_epoch}")
                except:
                    start_epoch = 0
            else:
                print(f"Failed to load checkpoint. Starting from scratch.")
        else:
            print(f"Checkpoint file not found: {resume_from}. Starting from scratch.")
    
    # Initialize policy evolution tracking
    policy_evolution_stats = []
    
    # Warmup phase (only if not resuming or at epoch 0)
    if start_epoch == 0 and "worker_config" in cvpo_config and "warmup_steps" in cvpo_config["worker_config"]:
        warmup_steps = cvpo_config["worker_config"]["warmup_steps"]
        if warmup_steps > 0:
            print(f"\n=== Collecting {warmup_steps} warmup steps ===")
            warmup_buffer = {
                'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'done': [], 'cost': []
            }
            collect_experience(
                env, cvpo_agent, max_steps=warmup_steps,
                buffer=warmup_buffer, capture_video=False,
                epoch=0
            )
            print(f"Warmup complete, collected {len(warmup_buffer['obs'])} samples")

    # Training parameters
    num_epochs = 100                 # Fewer epochs for simplicity  
    steps_per_epoch = 1000
    updates_per_epoch = int(steps_per_epoch / 64 * 2)  # 2 updates per 64 samples
    eval_frequency = 5
    
    # Initialize metrics tracking
    epoch_metrics = []
    eval_metrics = []
    
    # Start training
    print(f"\n=== Starting training from epoch {start_epoch+1} to {num_epochs} ===")
    
    total_start_time = time.time()
    
    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # Apply curriculum learning
        apply_curriculum(env, epoch)
        
        # Create buffer for storing experiences
        buffer = {
            'obs': [],
            'act': [],
            'rew': [],
            'next_obs': [],
            'done': [],
            'cost': []
        }
        
        # Collect experience
        metrics = collect_experience(
            env, 
            cvpo_agent, 
            max_steps=steps_per_epoch,
            buffer=buffer,
            capture_video=(epoch % 10 == 0),  # Record video every 10 epochs
            epoch=epoch
        )
        
        # Process collected experiences for training
        if len(buffer['obs']) >= 64:  # Minimum batch size
            print(f"\n=== Training on {len(buffer['obs'])} experiences ===")
            
            # Convert lists to numpy arrays
            train_data = {
                'obs': np.array(buffer['obs']),
                'act': np.array(buffer['act']),
                'rew': np.array(buffer['rew']),
                'obs2': np.array(buffer['next_obs']),  # Used 'obs2' instead of 'next_obs'
                'done': np.array(buffer['done']),
                'cost': np.array(buffer['cost'])
            }
            
            # Update the agent with collected experiences
            policy_updates_magnitude = []
            policy_losses = []
            
            for update_step in range(updates_per_epoch):
                update_info = cvpo_agent.learn_on_batch(train_data)
                
                # Track policy update info if available
                if update_info is not None:
                    if 'update_magnitude' in update_info:
                        policy_updates_magnitude.append(update_info.get('update_magnitude', 0))
                    if 'loss' in update_info:
                        policy_losses.append(update_info.get('loss', 0))
                
                if update_step % 5 == 0:
                    print(f"Completed update step {update_step+1}/{updates_per_epoch}")
                    
                    # Print policy update info
                    if update_info:
                        print("  Policy update information:")
                        for key, value in update_info.items():
                            if isinstance(value, (int, float)):
                                print(f"    {key}: {value:.6f}")
            
            # Store the last update magnitude for policy evolution tracking
            if policy_updates_magnitude:
                cvpo_agent.last_update_magnitude = np.mean(policy_updates_magnitude)
            
            # Log policy update summary
            if policy_updates_magnitude:
                avg_update = sum(policy_updates_magnitude) / len(policy_updates_magnitude)
                print(f"\nPolicy update summary:")
                print(f"  Average update magnitude: {avg_update:.6f}")
                
                if len(policy_updates_magnitude) > 1:
                    # Check if updates are getting smaller (potential convergence)
                    first_half = sum(policy_updates_magnitude[:len(policy_updates_magnitude)//2]) / (len(policy_updates_magnitude)//2)
                    second_half = sum(policy_updates_magnitude[len(policy_updates_magnitude)//2:]) / (len(policy_updates_magnitude)//2)
                    
                    if second_half < first_half * 0.8:  # 20% decrease
                        print(f"  ✓ Updates getting smaller - potential convergence")
                    elif second_half > first_half * 1.2:  # 20% increase
                        print(f"  ⚠️ Updates getting larger - potential instability")
                    else:
                        print(f"  ✓ Updates relatively stable")
            else:
                print("  No policy update magnitude information available")
        
        # Save model after each epoch
        model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        save_model(cvpo_agent, model_path)
        
        # Track epoch metrics
        epoch_time = time.time() - epoch_start_time
        metrics['epoch'] = epoch + 1
        metrics['epoch_time'] = epoch_time
        epoch_metrics.append(metrics)
        
        # Track policy evolution
        policy_stats = track_policy_evolution(cvpo_agent, env, epoch) if 'track_policy_evolution' in globals() else None
        if policy_stats:
            policy_evolution_stats.append(policy_stats)
        
        # Evaluate policy periodically
        if epoch % eval_frequency == 0 or epoch == num_epochs - 1:
            eval_reward, eval_length = evaluate_policy(
                env, cvpo_agent, num_episodes=2, capture_video=True, epoch=epoch
            )
            eval_metrics.append({
                'epoch': epoch + 1,
                'reward': eval_reward,
                'length': eval_length
            })
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"Episodes: {metrics['episode_count']}")
        print(f"Average reward: {metrics['average_reward']:.2f}")
        print(f"Average episode length: {metrics['average_length']:.2f}")
    
    # Training complete
    total_time = time.time() - total_start_time
    print(f"\n=== Training completed in {total_time:.1f}s ===")
    
    # Plot policy evolution
    if 'plot_policy_evolution' in globals() and policy_evolution_stats:
        plot_policy_evolution(policy_evolution_stats, output_dir)
    
    # Final evaluation
    print("\n=== Final Policy Evaluation ===")
    final_reward, final_length = evaluate_policy(
        env, cvpo_agent, num_episodes=3, capture_video=True, epoch=num_epochs
    )
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Total epochs: {num_epochs}")
    if epoch_metrics:
        print(f"Initial average reward: {epoch_metrics[0]['average_reward']:.2f}")
        print(f"Final average reward: {epoch_metrics[-1]['average_reward']:.2f}")
    print(f"Final evaluation reward: {final_reward:.2f}")
    
    return cvpo_agent, epoch_metrics, eval_metrics

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train CVPO on Carla')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--town', type=str, default='Town05', help='Carla town to use')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        # Train CVPO on Carla
        agent, metrics, eval_metrics = train_cvpo_on_carla(resume_from=args.resume)
        print("\n=== Training completed successfully! ===")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
