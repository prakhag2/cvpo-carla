#!/usr/bin/env python

"""
This file registers the Carla environment with the appropriate wrappers for CVPO.
Place this file in the same directory as your main script or in a location that gets
imported during initialization.
"""

import gym
from gym.envs.registration import register
import gym_carla  # This imports the base environment
from carla_wrapper import CarlaWrapper

# Check if the environment is already registered to avoid errors
env_name = 'SafetyCarla-v0'
registered = False
for env_spec in gym.envs.registry.all():
    if env_spec.id == env_name:
        registered = True
        break

# Register the wrapped version of the Carla environment for safety RL
if not registered:
    register(
        id=env_name,
        entry_point='register_env:make_safety_carla_env',
        max_episode_steps=1000,
    )

def make_safety_carla_env(params=None):
    """
    Factory function to create a wrapped Carla environment suitable for safety RL.
    
    Args:
        params: Dictionary of parameters for the Carla environment
        
    Returns:
        Wrapped Carla environment
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'number_of_vehicles': 10,
            'number_of_walkers': 5,
            'display_size': 256,
            'max_past_step': 1,
            'dt': 0.1,
            'discrete': False,
            'continuous_accel_range': [-3.0, 3.0],
            'continuous_steer_range': [-0.3, 0.3],
            'ego_vehicle_filter': 'vehicle.lincoln*',
            'port': 2000,
            'town': 'Town03',
            'task_mode': 'random',
            'max_time_episode': 1000,
            'max_waypt': 12,
            'obs_range': 32,
            'lidar_bin': 0.125,
            'd_behind': 12,
            'out_lane_thres': 2.0,
            'desired_speed': 8,
            'max_ego_spawn_times': 200,
            'display_route': True,
            'pixor': False,
        }
    
    # Create the base Carla environment
    env = gym.make('carla-v0', params=params)
    
    # Wrap the environment for safety RL
    wrapped_env = CarlaWrapper(env)
    
    return wrapped_env

# You can test the environment creation with:
if __name__ == "__main__":
    env = gym.make(env_name)
    print(f"Successfully created {env_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    env.close()
