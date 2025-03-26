#!/usr/bin/env python
import numpy as np
import torch
from safe_rl.policy.cvpo import CVPO
from gym.spaces import Dict, Box
from scipy.optimize import minimize
from safe_rl.util.torch_util import to_ndarray
import os
import matplotlib.pyplot as plt

class DebugCVPO(CVPO):
    """
    Simplified Debug version of CVPO for Carla focusing on curve navigation.
    Safety constraints are enabled to ensure safe driving behavior.
    """
    def __init__(self, env, logger, **kwargs):
        # Tracking variables
        self.step_count = 0
        self.episode_count = 0
        self.steering_history = []
        self.curvature_history = []
        
        # Previous action for smoothing
        self.previous_action = None
        
        # Enable safety constraints
        self.use_cost_value = True
        
        # Handle dictionary observation space for Carla
        self.is_dict_obs = isinstance(env.observation_space, Dict)

        if self.is_dict_obs:
            # Save the original observation space
            self.original_obs_space = env.observation_space

            # Use only the 'state' part of the observation
            if 'state' in self.original_obs_space.spaces:
                self.obs_key = 'state'
                self.base_obs_dim = self.original_obs_space['state'].shape[0]
            else:
                # If no 'state', find the first Box space
                for key, space in self.original_obs_space.spaces.items():
                    if isinstance(space, Box) and len(space.shape) == 1:
                        self.obs_key = key
                        self.base_obs_dim = space.shape[0]
                        break
            
            # Add room for road features
            self.road_feature_dim = 3  # Add 3 features for road curvature
            self.obs_dim = self.base_obs_dim + self.road_feature_dim
            
            print(f"Using {self.obs_key} of dim {self.base_obs_dim} + {self.road_feature_dim} road features = {self.obs_dim} total")

            # Override observation space for CVPO internal use
            env.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32
            )

            # Save original env.step and reset functions
            self.original_step = env.step
            self.original_reset = env.reset

            # Override env.step to preprocess observations
            def step_wrapper(action):
                obs, reward, done, info = self.original_step(action)
                processed_obs = self._process_observation(obs)
                return processed_obs, reward, done, info

            # Override env.reset to preprocess observations
            def reset_wrapper():
                obs = self.original_reset()
                processed_obs = self._process_observation(obs)
                return processed_obs

            # Apply the wrappers
            env.step = step_wrapper
            env.reset = reset_wrapper

        # Override kwargs to ensure safety constraints are enabled
        kwargs["use_cost_value"] = True
        
        # Make sure required parameters are set
        if "ac_kwargs" not in kwargs:
            kwargs["ac_kwargs"] = {"hidden_sizes": [256, 256]}
            
        # Set cost threshold for safety
        if "cost_limit" not in kwargs:
            kwargs["cost_limit"] = 50  # Higher than default to allow learning
        
        # Increase learning rate and KL constraints
        if "pi_lr" not in kwargs:
            kwargs["pi_lr"] = 3e-3  # Higher learning rate
        if "kl_mean_constraint" not in kwargs:
            kwargs["kl_mean_constraint"] = 0.05  # Looser KL constraint
        if "kl_var_constraint" not in kwargs:
            kwargs["kl_var_constraint"] = 0.005  # Looser variance constraint
        
        # Call the parent class constructor
        try:
            test_obs = env.reset()
            super().__init__(env, logger, **kwargs)
            
            # Check initialization
            if not hasattr(self, 'actor') or not hasattr(self, 'critic'):
                raise ValueError("CVPO initialization failed - actor or critic missing")
            
            # Initialize action variables
            self.previous_action = np.zeros(env.action_space.shape)
            
            print("CVPO initialized with safety constraints enabled")
            
        except Exception as e:
            print(f"Error in CVPO initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_road_features(self, birdeye_img):
        """Extract road curvature features from birdeye view."""
        try:
            # Convert to grayscale if it's RGB
            if len(birdeye_img.shape) == 3:
                gray = np.mean(birdeye_img, axis=2)
            else:
                gray = birdeye_img
            
            # Extract center region where the road should be
            height, width = gray.shape
            center_width = width // 3
            center = gray[:, width//2 - center_width//2:width//2 + center_width//2]
            
            # Normalize values
            if np.max(center) > 0:
                center = center / np.max(center)
            
            # Get road profiles at different distances
            near_idx = int(height * 0.7)  # Near (70% down from top)
            mid_idx = int(height * 0.5)   # Middle (50% down from top)
            far_idx = int(height * 0.3)   # Far (30% down from top)
            
            # Ensure indices are within bounds
            near_idx = min(near_idx, height-1)
            mid_idx = min(mid_idx, height-1)
            far_idx = min(far_idx, height-1)
            
            # Get horizontal slices
            near_profile = center[near_idx, :]
            mid_profile = center[mid_idx, :]
            far_profile = center[far_idx, :]
            
            # Find road center in each profile
            def get_road_center(profile):
                if np.max(profile) > 0:
                    road_center = np.argmax(profile)
                    normalized_center = (road_center / len(profile) - 0.5) * 2
                    return normalized_center
                return 0.0
            
            near_center = get_road_center(near_profile)
            mid_center = get_road_center(mid_profile)
            far_center = get_road_center(far_profile)
            
            # Calculate road curvature
            road_curvature = far_center - near_center
            
            # Store curvature for tracking
            self.curvature_history.append(road_curvature)
            
            # Return features
            return np.array([near_center, mid_center, far_center])
            
        except Exception as e:
            return np.zeros(3)  # Return zeros if extraction fails
    
    def transform_action(self, action):
        """Scale actions with focus on turn-appropriate speed."""
        # Make a copy to avoid modifying the original
        raw_action = action.copy()
        
        # Get road curvature if available
        road_curvature = 0.0
        if len(self.curvature_history) > 0:
            road_curvature = self.curvature_history[-1]
        
        # Adjust speed based on curvature
        curvature_magnitude = abs(road_curvature)
        if curvature_magnitude > 0.5:
            # Slow down in sharp turns
            speed_scale = 2.0
        else:
            # Regular speed on straight roads
            speed_scale = 4.0
        
        # Apply momentum for smoother transitions
        momentum_factor = 0.1  # Lower value = more responsive
        self.previous_action = momentum_factor * self.previous_action + (1 - momentum_factor) * raw_action
        
        # Scale actions
        scaled_action = self.previous_action.copy()
        
        # Scale acceleration with curvature-based factor
        scaled_action[0] = raw_action[0] * speed_scale + 1.0  # Add base speed offset
        
        # Responsive steering for curves
        scaled_action[1] = np.clip(raw_action[1] * 0.5, -0.9, 0.9)
        
        # Track steering for analysis
        self.steering_history.append(scaled_action[1])
        
        return scaled_action
        
    def _process_observation(self, obs):
        """Process dictionary observations with road features."""
        # Handle normal observations
        if isinstance(obs, np.ndarray) and obs.shape == (self.obs_dim,):
            return obs
        if isinstance(obs, np.ndarray) and np.prod(obs.shape) == self.obs_dim:
            return obs.reshape(self.obs_dim)

        # Handle dictionary observations
        if self.is_dict_obs:
            if isinstance(obs, dict):
                # Extract base observation component
                if self.obs_key in obs:
                    base_obs = obs[self.obs_key]
                else:
                    base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)

                # Ensure base_obs has the right shape
                if isinstance(base_obs, np.ndarray):
                    if base_obs.shape != (self.base_obs_dim,):
                        if np.prod(base_obs.shape) == self.base_obs_dim:
                            base_obs = base_obs.reshape(self.base_obs_dim)
                        else:
                            base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
                else:
                    try:
                        base_obs = np.array(base_obs, dtype=np.float32)
                        if base_obs.shape != (self.base_obs_dim,):
                            base_obs = base_obs.reshape(self.base_obs_dim)
                    except:
                        base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
                
                # Extract road features if birdeye view is available
                road_features = np.zeros(self.road_feature_dim, dtype=np.float32)
                if 'birdeye' in obs and obs['birdeye'] is not None:
                    road_features = self._extract_road_features(obs['birdeye'])
                
                # Combine base observation with road features
                return np.concatenate([base_obs, road_features])
            else:
                # Handle unexpected observation format
                return np.zeros(self.obs_dim, dtype=np.float32)

        # If not using dict observations, return as is
        return obs
    
    def _update_actor(self, data):
        """
        Override the actor update to fix loss calculation and encourage learning.
        This version properly handles safety constraints.
        """
        obs = data['obs']  # [batch, obs_dim]
        N = self.sample_action_num
        K = obs.shape[0]
        da = self.act_dim
        ds = self.obs_dim

        with torch.no_grad():
            # sample N actions per state
            b_mean, b_A = self.actor_targ.forward(obs)  # (K,)
            b = torch.distributions.MultivariateNormal(b_mean, scale_tril=b_A)  # (K,)
            sampled_actions = b.sample((N, ))  # (N, K, da)

            expanded_states = obs[None, ...].expand(N, -1, -1)  # (N, K, ds)
            target_q, _ = self.critic_forward(self.critic_targ,
                                          expanded_states.reshape(-1, ds),
                                          sampled_actions.reshape(-1, da))
            target_q = target_q.reshape(N, K)  # (N, K)
            target_q_np = to_ndarray(target_q).T  # (K, N)
            
            # Print some statistics about Q-values for debugging
            if self.step_count % 100 == 0:
                print(f"Q-values - Min: {target_q_np.min():.4f}, Max: {target_q_np.max():.4f}, Mean: {target_q_np.mean():.4f}")
                
            # Get cost values for safety constraints
            target_qc, _ = self.critic_forward(self.qc_targ,
                                            expanded_states.reshape(-1, ds),
                                            sampled_actions.reshape(-1, da))
            target_qc = target_qc.reshape(N, K)  # (N, K)
            target_qc_np = to_ndarray(target_qc).T  # (K, N)
            
            # Print cost values stats for debugging
            if self.step_count % 100 == 0:
                print(f"Cost values - Min: {target_qc_np.min():.4f}, Max: {target_qc_np.max():.4f}, Mean: {target_qc_np.mean():.4f}")

        def dual(x):
            """
            dual function of the non-parametric variational
            """
            η, lam = x
            # Add a small epsilon to eta to avoid division by zero
            η = max(η, 1e-5)
            
            # Calculate combined Q-values (reward - lambda * cost)
            target_q_np_comb = target_q_np - lam * target_qc_np
            
            # Get max Q value for each state to stabilize softmax
            max_q = np.max(target_q_np_comb, 1)
            
            # Compute dual objective with more stable calculations
            dual_obj = η * self.dual_constraint + lam * self.qc_thres + np.mean(max_q) \
                + η * np.mean(np.log(np.mean(np.exp((target_q_np_comb - max_q[:, None]) / η), axis=1) + 1e-10))
                
            return dual_obj

        # Set starting points and bounds for optimization
        initial_eta = max(1e-5, self.eta)
        initial_lam = max(1e-5, self.lam)
        
        bounds = [(1e-5, 1e4), (1e-5, 1e4)]
        options = {"ftol": 1e-3, "maxiter": 10}
        
        try:
            res = minimize(dual,
                        np.array([initial_eta, initial_lam]),
                        method='SLSQP',
                        bounds=bounds,
                        tol=1e-3,
                        options=options)
            self.eta, self.lam = res.x
            
            # Ensure eta is positive to avoid NaN issues
            self.eta = max(1e-5, self.eta)
            self.lam = max(1e-5, self.lam)
                
            # Track convergence and optimize status
            if not res.success and self.step_count % 100 == 0:
                print(f"Optimization warning: {res.message}")
        except Exception as e:
            print(f"Optimization error: {e}")
            self.eta, self.lam = initial_eta, initial_lam

        # Compute advantage weights with temperature scaling
        # Add a small epsilon to avoid extreme values
        advantage = target_q - self.lam * target_qc
        qij = torch.softmax(advantage / self.eta, dim=0)  # (N, K)

        # Initialize loss accumulator for cleaner tracking
        policy_loss = 0.0

        # M-Step of Policy Improvement
        for _ in range(self.mstep_iteration_num):
            mean, A = self.actor.forward(obs)
            π1 = torch.distributions.MultivariateNormal(loc=mean, scale_tril=b_A)  # (K,)
            π2 = torch.distributions.MultivariateNormal(loc=b_mean, scale_tril=A)  # (K,)
            
            # Log probabilities with stability improvements
            log_probs = π1.expand((N, K)).log_prob(sampled_actions) + π2.expand((N, K)).log_prob(sampled_actions)
            
            # Maximize log probability weighted by advantages
            loss_p = torch.mean(qij * log_probs)

            # Calculate KL divergence between old and new policies
            from safe_rl.policy.cvpo import gaussian_kl
            kl_μ, kl_Σ, Σi_det, Σ_det = gaussian_kl(μi=b_mean, μ=mean, Ai=b_A, A=A)

            # Add entropy bonus to encourage exploration (key fix for learning)
            entropy_bonus = torch.distributions.MultivariateNormal(loc=mean, scale_tril=A).entropy().mean() * 0.001

            # Update lagrange multipliers by gradient descent
            self.alpha_mean -= self.alpha_mean_scale * (self.kl_mean_constraint - 
                                                       kl_μ).detach().item()
            self.alpha_var -= self.alpha_var_scale * (self.kl_var_constraint -
                                                     kl_Σ).detach().item()

            self.alpha_mean = np.clip(self.alpha_mean, 0.0, self.alpha_mean_max)
            self.alpha_var = np.clip(self.alpha_var, 0.0, self.alpha_var_max)

            self.actor_optimizer.zero_grad()
            
            # Modified loss with entropy bonus and improved numerical stability
            loss_l = -(loss_p + self.alpha_mean * (self.kl_mean_constraint - kl_μ) + 
                      self.alpha_var * (self.kl_var_constraint - kl_Σ) + entropy_bonus)
            
            # Save policy loss for tracking
            policy_loss = loss_l.item()
            
            # Log significant updates
            if abs(loss_l.item()) > 10.0 and self.step_count % 100 == 0:
                print(f"Large loss value: {loss_l.item():.4f}")

            loss_l.backward()
            
            # Clip gradients to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            if grad_norm > 5.0 and self.step_count % 100 == 0:
                print(f"Large gradient norm before clipping: {grad_norm:.4f}")
                
            self.actor_optimizer.step()

            # Log actor update info
            self.logger.store(LossAll=loss_l.item(),
                           LossMLE=(-loss_p).item(),
                           mean_Σ_det=Σ_det.item(),
                           max_kl_Σ=kl_Σ.item(),
                           max_kl_μ=kl_μ.item(),
                           QcThres=self.qc_thres,
                           eta=self.eta,
                           lam=self.lam)
            
        return policy_loss
    
    def learn_on_batch(self, data):
        """Learn from batch with enhanced reporting."""
        self.step_count += 1

        try:
            # Fix key mapping to match what CVPO expects and ensure all data is tensor
            cvpo_data = {}
            
            # Get the device being used by the model
            device = next(self.actor.parameters()).device
            
            # Convert all numpy arrays to tensors and fix key names
            for key, value in data.items():
                if key == 'obs2':
                    # Keep obs2 as is
                    cvpo_key = key
                elif key == 'next_obs':
                    # Map next_obs -> obs2 if present
                    cvpo_key = 'obs2'
                else:
                    cvpo_key = key
                
                # Convert numpy arrays to tensors if needed and move to correct device
                if isinstance(value, np.ndarray):
                    cvpo_data[cvpo_key] = torch.as_tensor(value, dtype=torch.float32).to(device)
                elif isinstance(value, torch.Tensor):
                    cvpo_data[cvpo_key] = value.to(device)
                else:
                    cvpo_data[cvpo_key] = value
            
            # Debugging: print keys occasionally
            if self.step_count % 100 == 0:
                print(f"Data keys for learning: {list(cvpo_data.keys())}")
            
            # Check all required keys
            required_keys = ['obs', 'act', 'rew', 'obs2', 'done']
            for key in required_keys:
                if key not in cvpo_data:
                    raise KeyError(f"Missing required key '{key}' in data dictionary")
            
            # Capture parameter states before update
            policy_params_before = {}
            for name, param in self.actor.named_parameters():
                policy_params_before[name] = param.data.clone().cpu().detach()
            
            # Get initial parameter norms before update
            initial_actor_norm = 0.0
            param_count = 0
            for param in self.actor.parameters():
                initial_actor_norm += torch.norm(param).item()
                param_count += 1
            
            initial_actor_norm = initial_actor_norm / max(1, param_count)
            
            # Update critics
            critic_loss = self._update_critic(cvpo_data)
            
            # Update cost critic (for safety constraints)
            qc_loss = self._update_qc(cvpo_data)
            
            # Get sample actions from current policy for diagnostic
            with torch.no_grad():
                sample_obs = cvpo_data['obs'][:min(10, len(cvpo_data['obs']))]
                before_mean, _ = self.actor.forward(sample_obs)
                before_steering_mean = before_mean[:, 1].mean().item()
                before_accel_mean = before_mean[:, 0].mean().item()
            
            # Update actor with our overridden method
            actor_loss = self._update_actor(cvpo_data)
            
            # Update target networks
            self._polyak_update_target(self.critic, self.critic_targ)
            self._polyak_update_target(self.qc, self.qc_targ)
            self._polyak_update_target(self.actor, self.actor_targ)
            
            # Measure parameter changes
            policy_update_magnitude = {}
            total_update_norm = 0.0
            
            for name, param in self.actor.named_parameters():
                if name in policy_params_before:
                    # Calculate change in parameters
                    update = param.data.cpu().detach() - policy_params_before[name]
                    update_norm = torch.norm(update).item()
                    policy_update_magnitude[name] = update_norm
                    total_update_norm += update_norm
            
            # Calculate final actor norm after update
            final_actor_norm = 0.0
            for param in self.actor.parameters():
                final_actor_norm += torch.norm(param).item()
            
            final_actor_norm = final_actor_norm / max(1, param_count)
            
            # Extract action distribution after update
            with torch.no_grad():
                sample_obs = cvpo_data['obs'][:min(10, len(cvpo_data['obs']))]
                mean, A = self.actor.forward(sample_obs)
                
                # Calculate steering statistics - add a small jitter to covariance
                # to prevent zero standard deviation
                jittered_cov = A @ A.transpose(-2, -1) + torch.eye(A.size(-1), device=A.device) * 1e-4
                distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=jittered_cov)
                
                # Sample actions to get empirical mean and std
                samples = distribution.sample((20,))  # Sample 20 actions per state
                steering_samples = samples[..., 1]
                accel_samples = samples[..., 0]
                
                steering_mean = steering_samples.mean().item()
                steering_std = steering_samples.std().item()
                accel_mean = accel_samples.mean().item()
                accel_std = accel_samples.std().item()
            
            # Diagnostic: Print policy change
            if self.step_count % 100 == 0:
                print(f"Policy change - Steering: {before_steering_mean:.4f} -> {steering_mean:.4f}, "
                      f"Accel: {before_accel_mean:.4f} -> {accel_mean:.4f}")
                print(f"Actor norm changed: {initial_actor_norm:.4f} -> {final_actor_norm:.4f}")
                print(f"Cost threshold: {self.qc_thres:.4f}, Lambda: {self.lam:.4f}")
            
            return {
                'actor_norm': final_actor_norm,
                'actor_norm_change': final_actor_norm - initial_actor_norm,
                'critic_norm': torch.norm(next(self.critic.parameters())).item(),
                'qc_norm': torch.norm(next(self.qc.parameters())).item() if hasattr(self, 'qc') else 0.0,
                'update_magnitude': total_update_norm,
                'steering_mean': steering_mean,
                'steering_std': steering_std,
                'accel_mean': accel_mean,
                'accel_std': accel_std,
                'loss': actor_loss,
                'before_steering': before_steering_mean,
                'before_accel': before_accel_mean,
                'lambda': self.lam,
                'cost_threshold': self.qc_thres
            }
            
        except Exception as e:
            print(f"Error in learn_on_batch: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def act(self, obs, deterministic=False, with_logprob=False):
        """Override act method to handle observation processing and action transformation."""
        # Process observation if needed
        processed_obs = self._process_observation(obs)
        
        try:
            # Get action from CVPO's act method
            action, log_prob = super().act(processed_obs, deterministic, with_logprob)
            
            # Initialize action if not set
            if self.previous_action is None:
                self.previous_action = np.zeros_like(action)
            
            # Apply the action transformation
            scaled_action = self.transform_action(action)
            
            self.step_count += 1
            
            return scaled_action, log_prob
            
        except Exception as e:
            print(f"Error in act method: {e}")
            
            # Return a default action
            default_action = np.zeros(self.act_dim, dtype=np.float32)
            return default_action, np.array(0.0) if with_logprob else default_action
