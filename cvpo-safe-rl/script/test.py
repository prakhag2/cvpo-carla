#!/usr/bin/env python
import sys
import os
import gym
from gym.envs.registration import registry, spec, register

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import your environment directly
from highway_safety_gym.envs.highway_env import HighwaySafetyEnv

# Register your environment directly with disable_env_checker=True
register(
    id='highway-safety-gym-v0',
    entry_point='highway_safety_gym.envs.highway_env:HighwaySafetyEnv',
    max_episode_steps=100,
    disable_env_checker=True  # This disables the environment checker
)

def test_highway_safety_env():
    try:
        env = gym.make('highway-safety-gym-v0')
        print("\n✅ Environment created successfully!")
        
        # Test reset - expecting a tuple (obs, info) from the new API
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            print("\nInitial Observation Shape:", obs.shape)
            print("Initial Info:", info)
        else:
            # Handle the case where reset doesn't return the expected tuple
            obs = reset_result
            print("\nInitial Observation Shape:", obs.shape)
            print("Warning: reset() didn't return (obs, info) tuple. Got:", type(reset_result))
        
        # Test step - expecting a tuple (obs, reward, terminated, truncated, info) from the new API
        action = env.action_space.sample()
        step_result = env.step(action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            print("\nStep Results (new API):")
            print("Observation Shape:", obs.shape)
            print("Reward:", reward)
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Info:", info)
        elif len(step_result) == 4:
            # Fallback for old API
            obs, reward, done, info = step_result
            print("\nStep Results (old API):")
            print("Observation Shape:", obs.shape)
            print("Reward:", reward)
            print("Done:", done)
            print("Info:", info)
        else:
            print(f"\nUnexpected step result format. Got {len(step_result)} values:", step_result)
        
    except Exception as e:
        print(f"\n❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_highway_safety_env()
