#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import argparse
import time
from datetime import datetime

# Import Carla environment and policies
from carla_wrapper import make_carla_env
from debug_cvpo import DebugCVPO
from safe_rl.policy import SAC, TD3, SACLagrangian, TD3Lagrangian
from safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from safe_rl.util.torch_util import export_device_env_variable, seed_torch

# Import utilities from train_cvpo_carla.py 
from train_cvpo_carla import (collect_experience, evaluate_policy, 
                             save_model, load_model)

class CarlaRunner:
    """
    Basic runner for training and evaluating policies in the Carla environment
    """
    # Policy mapping to classes
    POLICY_LIB = {
        "cvpo": DebugCVPO,
        "sac": SAC,
        "sac_lag": SACLagrangian,
        "td3": TD3,
        "td3_lag": TD3Lagrangian,
    }
    
    def __init__(self,
                 policy="cvpo",
                 town="Town05",
                 exp_name=None,
                 seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 num_epochs=300,
                 steps_per_epoch=1000,
                 cost_limit=50,
                 max_episode_steps=500,
                 resume_from=None,
                 eval_frequency=5,
                 save_frequency=20,
                 carla_params=None,
                 policy_params=None,
                 output_dir=None,
                 verbose=True):
        """
        Initialize a Carla runner to train and evaluate policies
        """
        # Set up random seeds and device
        seed_torch(seed)
        export_device_env_variable(device)
        self.device = device
        print(f"Using device: {device}")
        
        # Store parameters
        self.policy_type = policy
        self.num_epochs = num_epochs
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.verbose = verbose
        self.max_episode_steps = max_episode_steps
        self.resume_from = resume_from
        self.cost_limit = cost_limit
        
        # Generate experiment name if not provided
        if exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{policy}_{town}_{timestamp}"
        
        # Set up output directory
        if output_dir is None:
            output_dir = "carla_results"
        self.output_dir = os.path.join(output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up basic configurations
        self._setup_configs(town, carla_params, policy_params, steps_per_epoch)
        
        # Set up logger
        logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=output_dir)
        self.logger = EpochLogger(**logger_kwargs)
        
        # Create environment
        self.env = self._create_environment()
        
        # Create policy
        self.policy = self._create_policy()
        
        # Load checkpoint if provided
        if self.resume_from is not None:
            if load_model(self.policy, self.resume_from):
                print(f"Successfully loaded checkpoint from {self.resume_from}")
            else:
                print(f"Failed to load checkpoint from {self.resume_from}")
    
    def _setup_configs(self, town, carla_params, policy_params, steps_per_epoch):
        """Set up basic configuration for the environment and policy"""
        # Set up Carla environment parameters
        self.carla_params = {
            'number_of_vehicles': 0,
            'number_of_walkers': 0,
            'display_size': 256,
            'max_past_step': 1,
            'dt': 0.1,
            'discrete': False,
            'discrete_acc': [-3.0, 0.0, 3.0],  # Required even for continuous control
            'discrete_steer': [-0.2, 0.0, 0.2],  # Required even for continuous control
            'continuous_accel_range': [-3.0, 3.0],
            'continuous_steer_range': [-0.5, 0.5],
            'ego_vehicle_filter': 'vehicle.lincoln*',
            'port': 2000,
            'town': town,
            'task_mode': 'random',
            'max_time_episode': 500,
            'max_waypt': 12,
            'obs_range': 32,
            'lidar_bin': 0.125,
            'd_behind': 12,
            'out_lane_thres': 3.0,
            'desired_speed': 5,
            'max_ego_spawn_times': 200,
            'display_route': True,
            'pixor_size': 64,
            'pixor': False,
        }
        
        # Override Carla parameters if provided
        if carla_params is not None:
            self.carla_params.update(carla_params)
        
        # Set up basic policy parameters
        if self.policy_type == "cvpo":
            self.policy_params = {
                "ac_kwargs": {"hidden_sizes": [256, 128]},
                "use_cost_value": True,
                "steps_per_epoch": steps_per_epoch,
                "max_ep_len": 500,
                "worker_config": {
                    "reset_on_done": True,
                    "buffer_size": 100000,
                    "warmup_steps": 2000,
                    "batch_size": 64
                },
                "sample_action_num": 32,
                "mstep_iteration_num": 3,
                "kl_mean_constraint": 0.05,
                "kl_var_constraint": 0.005,
                "alpha_mean_scale": 0.5,
                "alpha_var_scale": 50.0,
                "alpha_scale": 10.0,
                "alpha_mean_max": 0.5,
                "alpha_var_max": 50.0,
                "alpha_max": 5.0,
                "pi_lr": 3e-3,
                "vf_lr": 1e-3,
                "gamma": 0.99,
                "polyak": 0.995,
                "num_q": 2,
                "num_qc": 2,
                "cost_limit": self.cost_limit,
            }
                
        elif self.policy_type in ["sac", "sac_lag"]:
            # SAC parameters
            self.policy_params = {
                "actor_lr": 3e-3,
                "critic_lr": 1e-3,
                "ac_kwargs": {"hidden_sizes": [256, 128]},
                "gamma": 0.99,
                "polyak": 0.995,
                "alpha": 0.2,
                "num_q": 2,
                "steps_per_epoch": steps_per_epoch,
                "worker_config": {
                    "batch_size": 64,
                    "buffer_size": 100000,
                    "warmup_steps": 2000
                }
            }
            
            if self.policy_type == "sac_lag":
                # Lagrangian parameters
                self.policy_params.update({
                    "num_qc": 2,
                    "cost_limit": self.cost_limit,
                    "use_cost_decay": True,
                    "cost_start": 100,
                    "cost_end": 25,
                    "decay_epoch": 250,
                    "KP": 0,
                    "KI": 0.1,
                    "KD": 0,
                    "per_state": True
                })
                
        elif self.policy_type in ["td3", "td3_lag"]:
            # TD3 parameters
            self.policy_params = {
                "actor_lr": 3e-3,
                "critic_lr": 1e-3,
                "ac_kwargs": {"hidden_sizes": [256, 128]},
                "gamma": 0.99,
                "polyak": 0.995,
                "act_noise": 0.1,
                "target_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "num_q": 2,
                "steps_per_epoch": steps_per_epoch,
                "worker_config": {
                    "batch_size": 64,
                    "buffer_size": 100000,
                    "warmup_steps": 2000
                }
            }
            
            if self.policy_type == "td3_lag":
                # Lagrangian parameters
                self.policy_params.update({
                    "num_qc": 2,
                    "cost_limit": self.cost_limit,
                    "use_cost_decay": True,
                    "cost_start": 100,
                    "cost_end": 25,
                    "decay_epoch": 250,
                    "KP": 0,
                    "KI": 0.1,
                    "KD": 0,
                    "per_state": True
                })
        
        # Override policy parameters if provided
        if policy_params is not None:
            self.policy_params.update(policy_params)
            
        print("\nEnvironment parameters:")
        for key, value in self.carla_params.items():
            print(f"  {key}: {value}")
            
        print("\nPolicy parameters:")
        for key, value in self.policy_params.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    
    def _create_environment(self):
        """Create the Carla environment"""
        print("\nCreating Carla environment...")
        env = make_carla_env(
            params=self.carla_params,
            max_episode_steps=self.max_episode_steps,
            cost_threshold=self.cost_limit,
            verbose=self.verbose
        )
        print("Environment created successfully!")
        return env
    
    def _create_policy(self):
        """Create the policy based on the selected type"""
        print(f"\nCreating {self.policy_type} policy...")
        
        if self.policy_type not in self.POLICY_LIB:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        policy_class = self.POLICY_LIB[self.policy_type]
        
        # Wait for environment observation space to be properly initialized
        if self.env.observation_space is None:
            print("Warning: Environment observation space not initialized. Running a reset to initialize it.")
            self.env.reset()
            
        if hasattr(self.env, 'observation_space') and self.env.observation_space is not None:
            print(f"Observation space: {self.env.observation_space}")
        else:
            raise RuntimeError("Environment observation space is still None after reset. Cannot initialize policy.")
        
        # Check observation space type and adapt if needed
        import gym
        original_obs_space = self.env.observation_space
        
        # If the observation space is a Dict, adapt it for algorithms that expect a flat Box
        if isinstance(original_obs_space, gym.spaces.Dict):
            print(f"Adapting Dict observation space for {self.policy_type}...")
            
            # Choose the 'state' component as it's usually the main vector of observations
            if 'state' in original_obs_space.spaces:
                state_dim = original_obs_space['state'].shape[0]
                print(f"Using 'state' component with dimension {state_dim}")
                
                # Create a modified environment with a simpler observation space for the policy
                class ObsSpaceWrapper(gym.Wrapper):
                    def __init__(self, env):
                        super().__init__(env)
                        self.original_observation_space = env.observation_space
                        # Use state dimension as the observation dimension
                        self.observation_space = gym.spaces.Box(
                            low=-float('inf'), 
                            high=float('inf'),
                            shape=(state_dim,),
                            dtype=np.float32
                        )
                    
                    def reset(self):
                        obs = self.env.reset()
                        # Return only the state component
                        return np.array(obs['state'], dtype=np.float32)
                    
                    def step(self, action):
                        obs, reward, done, info = self.env.step(action)
                        # Return only the state component
                        return np.array(obs['state'], dtype=np.float32), reward, done, info
                
                self.env = ObsSpaceWrapper(self.env)
                print(f"Wrapped environment created with new observation space: {self.env.observation_space}")
            else:
                raise ValueError("Dict observation space doesn't contain 'state' component. Cannot adapt for policy.")
        
        # Create policy instance
        try:
            policy = policy_class(self.env, self.logger, **self.policy_params)
            print("Policy created successfully!")
            return policy
        except Exception as e:
            print(f"Error creating policy: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def train(self):
        """Train the policy over multiple epochs"""
        print(f"\n=== Starting training for {self.num_epochs} epochs ===")
        
        # Create a directory for videos
        videos_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Create a symlink to redirect cvpo_carla_results to our directory
        default_video_dir = "cvpo_carla_results"
        
        # Remove existing symlink or directory if it exists
        if os.path.exists(default_video_dir):
            if os.path.islink(default_video_dir):
                os.unlink(default_video_dir)
            elif os.path.isdir(default_video_dir):
                try:
                    # Rename existing directory to avoid conflicts
                    backup_dir = f"{default_video_dir}_backup_{int(time.time())}"
                    print(f"Moving existing {default_video_dir} to {backup_dir}")
                    os.rename(default_video_dir, backup_dir)
                except Exception as e:
                    print(f"Warning: Could not move existing directory: {e}")
        
        # Create algorithm-specific directory for videos
        alg_video_dir = os.path.join(videos_dir, f"{self.policy_type}_videos")
        os.makedirs(alg_video_dir, exist_ok=True)
        
        # Create symlink to redirect default output to our directory
        try:
            os.symlink(alg_video_dir, default_video_dir, target_is_directory=True)
            print(f"Created symlink: {default_video_dir} -> {alg_video_dir}")
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
            # If symlink fails, just create the directory
            os.makedirs(default_video_dir, exist_ok=True)
            
        # Run training for specified number of epochs
        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            # Collect experience
            buffer = {
                'obs': [],
                'act': [],
                'rew': [],
                'next_obs': [],
                'done': [],
                'cost': []
            }
            
            # We'll use the default video capture which now goes to our algorithm directory
            capture_video = (epoch % 10 == 0)  # Capture video every 10 epochs
            
            if capture_video:
                print(f"Capturing video for {self.policy_type} at epoch {epoch+1}")
            
            metrics = collect_experience(
                self.env, 
                self.policy, 
                max_steps=self.policy_params["steps_per_epoch"],
                buffer=buffer,
                capture_video=capture_video,
                verbose=self.verbose,
                epoch=epoch
            )
            
            # Train policy on collected experience
            if len(buffer['obs']) >= 64:  # Minimum batch size
                print(f"\n=== Training on {len(buffer['obs'])} experiences ===")
                
                # Process data for training and convert to tensors
                train_data = {}
                for key, value in buffer.items():
                    if key == 'next_obs':
                        # obs2 is the key that SAC expects for next observations
                        train_data['obs2'] = torch.FloatTensor(np.array(value))
                    else:
                        train_data[key] = torch.FloatTensor(np.array(value))
                
                # Update policy multiple times
                updates_per_epoch = int(self.policy_params["steps_per_epoch"] / 64 * 2)
                
                # Capture state of key parameters before training for comparison
                if self.verbose:
                    print("\nTracking policy parameter changes during training...")
                    initial_params = {}
                    try:
                        # Get actor parameters
                        for name, param in self.policy.actor.named_parameters():
                            if 'weight' in name or 'bias' in name:
                                initial_params[name] = {
                                    'mean': param.data.mean().item(),
                                    'std': param.data.std().item(),
                                    'norm': torch.norm(param.data).item()
                                }
                        print(f"Tracking {len(initial_params)} actor parameters")
                    except Exception as e:
                        print(f"Could not capture initial parameters: {e}")
                
                # Track loss values from logger if available
                loss_history = {
                    'critic_loss': [], 
                    'actor_loss': [],
                    'entropy': []
                }
                
                for update_step in range(updates_per_epoch):
                    if hasattr(self.policy, 'learn_on_batch'):
                        update_info = self.policy.learn_on_batch(train_data)
                        
                        # Try to extract loss information from the logger
                        if hasattr(self.policy, 'logger') and hasattr(self.policy.logger, 'epoch_dict'):
                            for key in list(self.policy.logger.epoch_dict.keys()):
                                if 'LossQ' in key and self.policy.logger.epoch_dict[key]:
                                    loss_history['critic_loss'].append(self.policy.logger.epoch_dict[key][-1])
                                if 'LossPi' in key and self.policy.logger.epoch_dict[key]:
                                    loss_history['actor_loss'].append(self.policy.logger.epoch_dict[key][-1])
                                if 'LogPi' in key and self.policy.logger.epoch_dict[key]:
                                    loss_history['entropy'].append(-self.policy.logger.epoch_dict[key][-1])
                    
                    if update_step % 10 == 0 or update_step == updates_per_epoch - 1:
                        print(f"Completed update step {update_step+1}/{updates_per_epoch}")
                        
                        # Print loss statistics every 10 steps if available
                        if any(len(values) > 0 for values in loss_history.values()):
                            print("  Current loss statistics:")
                            for key, values in loss_history.items():
                                if values:
                                    # Check if values[-1] is a scalar or array
                                    if isinstance(values[-1], (int, float)):
                                        print(f"    {key}: {values[-1]:.4f} (avg: {np.mean(values):.4f})")
                                    else:
                                        # Handle numpy arrays or other non-scalar types
                                        print(f"    {key}: (avg: {np.mean(values):.4f})")
                
                # Compute parameter changes after all updates
                if self.verbose and 'initial_params' in locals():
                    print("\n=== Policy Parameter Changes ===")
                    param_changes = {}
                    param_metrics = {'mean_change': 0, 'max_change': 0, 'num_params': 0}
                    
                    try:
                        # Calculate changes in parameters
                        for name, param in self.policy.actor.named_parameters():
                            if name in initial_params:
                                current = {
                                    'mean': param.data.mean().item(),
                                    'std': param.data.std().item(),
                                    'norm': torch.norm(param.data).item()
                                }
                                
                                # Calculate relative change in norm
                                if initial_params[name]['norm'] > 0:
                                    rel_change = abs(current['norm'] - initial_params[name]['norm']) / initial_params[name]['norm']
                                    param_changes[name] = rel_change
                                    param_metrics['mean_change'] += rel_change
                                    param_metrics['max_change'] = max(param_metrics['max_change'], rel_change)
                                    param_metrics['num_params'] += 1
                        
                        # Calculate average change
                        if param_metrics['num_params'] > 0:
                            param_metrics['mean_change'] /= param_metrics['num_params']
                            
                            # Print summary of changes
                            print(f"Average parameter change: {param_metrics['mean_change']:.6f}")
                            print(f"Maximum parameter change: {param_metrics['max_change']:.6f}")
                            
                            # Interpret the results
                            if param_metrics['mean_change'] < 0.0001:
                                print("⚠️ Very small parameter changes - policy might not be learning effectively")
                            elif param_metrics['mean_change'] > 0.1:
                                print("⚠️ Very large parameter changes - learning might be unstable")
                            else:
                                print("✓ Parameter changes look reasonable")
                                
                            # Print details of largest changes
                            if param_changes:
                                sorted_changes = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)
                                print("\nLargest parameter changes:")
                                for name, change in sorted_changes[:3]:  # Show top 3
                                    print(f"  {name}: {change:.6f}")
                    except Exception as e:
                        print(f"Error computing parameter changes: {e}")
                
                # Print summary of loss history
                if any(len(values) > 0 for values in loss_history.values()):
                    print("\n=== Loss Summary ===")
                    for key, values in loss_history.items():
                        if values:
                            # Check if values contain scalar items
                            if all(isinstance(v, (int, float)) for v in values):
                                print(f"{key}:")
                                print(f"  Initial: {values[0]:.4f}")
                                print(f"  Final: {values[-1]:.4f}")
                                print(f"  Change: {values[-1] - values[0]:.4f}")
                                
                                # Check for convergence trends
                                if len(values) > 10:
                                    first_half = np.mean(values[:len(values)//2])
                                    second_half = np.mean(values[len(values)//2:])
                                    change = (second_half - first_half) / (abs(first_half) + 1e-8)
                                    
                                    if key == 'critic_loss' or key == 'actor_loss':
                                        if change < -0.1:  # Decreasing loss
                                            print(f"  ✓ {key} is decreasing ({change:.2%} change) - good sign")
                                        elif change > 0.1:  # Increasing loss
                                            print(f"  ⚠️ {key} is increasing ({change:.2%} change) - potential issue")
                                        else:
                                            print(f"  {key} is stable")
                            else:
                                # Handle case where values contain non-scalar items
                                mean_val = np.mean(values)
                                print(f"{key}:")
                                print(f"  Average: {mean_val:.4f}")
                                print(f"  Values are non-scalar or mixed types")
                        
                else:
                    print("\nNo loss statistics available - the policy doesn't log loss values")
            
            # Save model periodically
            if (epoch + 1) % self.save_frequency == 0 or epoch == self.num_epochs - 1:
                model_path = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pt")
                save_model(self.policy, model_path)
                print(f"Model saved to {model_path}")
            
            # Evaluate policy periodically
            if (epoch + 1) % self.eval_frequency == 0 or epoch == self.num_epochs - 1:
                eval_reward, eval_length = evaluate_policy(
                    self.env, self.policy, num_episodes=2, 
                    capture_video=True, epoch=epoch
                )
                print(f"Evaluation - Reward: {eval_reward:.2f}, Length: {eval_length:.2f}")
    
    def evaluate(self, num_episodes=5):
        """Evaluate the policy"""
        print(f"\n=== Evaluating policy for {num_episodes} episodes ===")
        
        # Create algorithm-specific directory for eval videos
        videos_dir = os.path.join(self.output_dir, "eval_videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Handle the default video directory
        default_video_dir = "cvpo_carla_results"
        
        # Remove any existing symlink
        if os.path.exists(default_video_dir):
            if os.path.islink(default_video_dir):
                os.unlink(default_video_dir)
            elif os.path.isdir(default_video_dir):
                try:
                    # Rename existing directory to avoid conflicts
                    backup_dir = f"{default_video_dir}_backup_{int(time.time())}"
                    print(f"Moving existing {default_video_dir} to {backup_dir}")
                    os.rename(default_video_dir, backup_dir)
                except Exception as e:
                    print(f"Warning: Could not move existing directory: {e}")
        
        # Create algorithm-specific directory for videos
        alg_video_dir = os.path.join(videos_dir, f"{self.policy_type}_eval_videos")
        os.makedirs(alg_video_dir, exist_ok=True)
        
        # Create symlink to redirect default output
        try:
            os.symlink(alg_video_dir, default_video_dir, target_is_directory=True)
            print(f"Created symlink: {default_video_dir} -> {alg_video_dir}")
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
            # If symlink fails, just create the directory
            os.makedirs(default_video_dir, exist_ok=True)
        
        print(f"Videos will be saved to: {alg_video_dir}")
        
        # Run the evaluation
        reward, length = evaluate_policy(
            self.env, 
            self.policy, 
            num_episodes=num_episodes, 
            capture_video=True, 
            epoch=0
        )
        
        print(f"Evaluation results:")
        print(f"  Average reward: {reward:.2f}")
        print(f"  Average episode length: {length:.2f}")
        print(f"  Videos saved to: {alg_video_dir}")
        
        return reward, length

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate policies in Carla")
    
    # Basic parameters
    parser.add_argument("--policy", type=str, default="cvpo", 
                        choices=["cvpo", "sac", "sac_lag", "td3", "td3_lag"],
                        help="Policy to use")
    parser.add_argument("--town", type=str, default="Town05", 
                        help="Carla town to use")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=300, 
                        help="Number of epochs to train")
    parser.add_argument("--cost_limit", type=float, default=50.0, 
                        help="Safety cost limit")
    
    # Mode and paths
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode to run in")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="carla_results", 
                        help="Directory to save outputs")
    parser.add_argument("--exp_name", type=str, default=None, 
                        help="Experiment name")
    
    # Evaluation options
    parser.add_argument("--eval_episodes", type=int, default=5, 
                        help="Number of episodes to evaluate")
    
    # Environment options
    parser.add_argument("--num_vehicles", type=int, default=0, 
                        help="Number of vehicles in environment")
    parser.add_argument("--desired_speed", type=float, default=5.0, 
                        help="Desired speed")
    parser.add_argument("--lane_threshold", type=float, default=3.0, 
                        help="Lane threshold")
    
    # Other options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    return args

def main():
    """Main entry point"""
    args = parse_args()
    
    # Configure environment parameters
    carla_params = {
        "number_of_vehicles": args.num_vehicles,
        "desired_speed": args.desired_speed,
        "out_lane_thres": args.lane_threshold,
        "town": args.town
    }
    
    # Create the runner
    runner = CarlaRunner(
        policy=args.policy,
        town=args.town,
        exp_name=args.exp_name,
        seed=args.seed,
        device=args.device,
        num_epochs=args.epochs,
        cost_limit=args.cost_limit,
        resume_from=args.resume,
        carla_params=carla_params,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Run the appropriate mode
    if args.mode == "train":
        runner.train()
    else:
        runner.evaluate(num_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
