#!/usr/bin/env python
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import gym
import gym_carla
import carla
import numpy as np
import os
import time
from PIL import Image
import json

# Create output directory for frames and metadata
output_dir = "carla_frames"
os.makedirs(output_dir, exist_ok=True)

class FrameRecorder:
    def __init__(self, output_dir, episode, view_types=None):
        """
        Initialize frame recorder
        
        Args:
            output_dir: directory to save frames
            episode: episode number
            view_types: which view types to record (default: all)
        """
        self.output_dir = output_dir
        self.episode = episode
        self.view_types = view_types or ['camera', 'lidar', 'birdeye', 'combined']
        self.frame_count = 0
        self.metadata = {
            'episode': episode,
            'frames': [],
            'view_types': self.view_types,
            'fps': 20  # Suggested FPS for video creation
        }
        
        # Create episode directory
        self.episode_dir = os.path.join(output_dir, f"episode_{episode}")
        os.makedirs(self.episode_dir, exist_ok=True)
    
    def add_frame(self, obs, reward=None, done=None, info=None):
        """Save a frame of the simulation"""
        frame_data = {
            'step': self.frame_count,
            'timestamp': time.time(),
            'reward': reward,
            'done': done
        }
        
        # Save individual views
        if 'camera' in self.view_types and 'camera' in obs:
            img = Image.fromarray(obs['camera'])
            img_path = os.path.join(self.episode_dir, f"step_{self.frame_count:04d}_camera.png")
            img.save(img_path)
            frame_data['camera_path'] = img_path
        
        if 'lidar' in self.view_types and 'lidar' in obs:
            img = Image.fromarray(obs['lidar'])
            img_path = os.path.join(self.episode_dir, f"step_{self.frame_count:04d}_lidar.png")
            img.save(img_path)
            frame_data['lidar_path'] = img_path
        
        if 'birdeye' in self.view_types and 'birdeye' in obs:
            img = Image.fromarray(obs['birdeye'])
            img_path = os.path.join(self.episode_dir, f"step_{self.frame_count:04d}_birdeye.png")
            img.save(img_path)
            frame_data['birdeye_path'] = img_path
        
        # Save combined view
        if 'combined' in self.view_types and all(k in obs for k in ['camera', 'lidar', 'birdeye']):
            h, w = obs['camera'].shape[0], obs['camera'].shape[1]
            combined = np.zeros((h, w*3, 3), dtype=np.uint8)
            combined[:, :w, :] = obs['camera']
            combined[:, w:2*w, :] = obs['lidar']
            combined[:, 2*w:, :] = obs['birdeye']
            img = Image.fromarray(combined)
            img_path = os.path.join(self.episode_dir, f"step_{self.frame_count:04d}_combined.png")
            img.save(img_path)
            frame_data['combined_path'] = img_path
        
        # Save state information if available
        if 'state' in obs:
            frame_data['state'] = obs['state'].tolist()
        
        # Add any additional info
        if info:
            # Convert numpy arrays to lists for JSON serialization
            clean_info = {}
            for k, v in info.items():
                if isinstance(v, np.ndarray):
                    clean_info[k] = v.tolist()
                elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                    clean_info[k] = [arr.tolist() for arr in v]
                else:
                    clean_info[k] = v
            frame_data['info'] = clean_info
        
        # Add frame data to metadata
        self.metadata['frames'].append(frame_data)
        self.frame_count += 1
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = os.path.join(self.episode_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Also save a script to create videos
        ffmpeg_path = os.path.join(self.output_dir, "create_videos.sh")
        with open(ffmpeg_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# This script creates videos from the recorded frames\n")
            f.write("# You need to have ffmpeg installed\n\n")
            
            # Add commands for each episode and view type
            for view in self.view_types:
                if view != 'combined':  # Skip combined if not needed
                    f.write(f"echo 'Creating episode {self.episode} {view} video...'\n")
                    f.write(f"ffmpeg -y -framerate 20 -pattern_type glob -i '{self.episode_dir}/step_*_{view}.png' " +
                           f"-c:v libx264 -pix_fmt yuv420p {self.output_dir}/episode_{self.episode}_{view}.mp4\n\n")
            
            # Also create a combined video if available
            if 'combined' in self.view_types:
                f.write(f"echo 'Creating episode {self.episode} combined video...'\n")
                f.write(f"ffmpeg -y -framerate 20 -pattern_type glob -i '{self.episode_dir}/step_*_combined.png' " +
                       f"-c:v libx264 -pix_fmt yuv420p {self.output_dir}/episode_{self.episode}_combined.mp4\n\n")
        
        # Make the script executable
        os.chmod(ffmpeg_path, 0o755)
        print(f"Saved metadata to {metadata_path}")
        print(f"Created video creation script at {ffmpeg_path}")
        return metadata_path

def print_state_info(obs, reward, done, info):
    """Print state information"""
    if 'state' in obs:
        state = obs['state']
        print(f"State: lateral_dis={state[0]:.2f}, delta_yaw={state[1]:.2f}, speed={state[2]:.2f}, vehicle_front={state[3]:.2f}")
    
    print(f"Reward: {reward:.2f}")
    print(f"Done: {done}")
    
    if 'waypoints' in info:
        print(f"Num waypoints: {len(info['waypoints'])}")
    
    if 'vehicle_front' in info:
        print(f"Vehicle front: {info['vehicle_front']}")

def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 100,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
    }
    
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    
    # Run for a fixed number of episodes
    num_episodes = 3
    max_steps_per_episode = 100
    
    for episode in range(num_episodes):
        print(f"\n=== Starting Episode {episode+1}/{num_episodes} ===")
        
        # Create frame recorder for this episode
        recorder = FrameRecorder(output_dir, episode+1)
        
        # Reset the environment
        obs = env.reset()
        print("Environment reset complete")
        
        # Record initial frame
        recorder.add_frame(obs)
        
        episode_reward = 0
        
        # Run for a fixed number of steps
        for step in range(max_steps_per_episode):
            print(f"\n--- Episode {episode+1}, Step {step+1} ---")
            
            # Take a simple action (constant acceleration, no steering)
            action = [2.0, 0.0]
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            
            # Accumulate reward
            episode_reward += reward
            
            # Print state information
            print_state_info(obs, reward, done, info)
            
            # Record frame
            recorder.add_frame(obs, reward, done, info)
            
            # If the episode is done (collision, out of lane, etc.), break the loop
            if done:
                print(f"Episode terminated after {step+1} steps")
                break
                
            # Small sleep to avoid overwhelming the terminal with output
            time.sleep(0.01)
        
        # Save metadata and create video creation script
        recorder.save_metadata()
        
        print(f"Episode {episode+1} completed with total reward: {episode_reward:.2f}")
    
    print("\nTest completed. Frames saved to", output_dir)
    print("To create videos from the frames, run the create_videos.sh script (requires ffmpeg)")
    env.close()

if __name__ == '__main__':
    main()
