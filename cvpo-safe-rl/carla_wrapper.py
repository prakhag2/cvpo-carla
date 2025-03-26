#!/usr/bin/env python

import gym
import numpy as np
from gym import spaces
import os 
import logging 
import carla
import gym_carla
import matplotlib.pyplot as plt

class CarlaWrapper(gym.Wrapper):
    """
    A simplified safety wrapper for Carla environments focusing on turns and speed control.
    
    Key Features:
    - Enhanced lane discipline rewards
    - Turn-appropriate speed rewards
    - Detailed diagnostics for curve handling
    """
    def __init__(self, env, max_episode_steps=1000, cost_threshold=5.0, verbose=True):
        """
        Initialize the safety-enhanced Carla environment wrapper.
        
        Args:
            env (gym.Env): Base Carla environment
            max_episode_steps (int): Maximum steps per episode
            cost_threshold (float): Cumulative safety cost threshold
            verbose (bool): Enable detailed logging
        """
        super().__init__(env)
        
        # Environment configuration
        self.env = env
        self.max_episode_steps = max_episode_steps
        self._step_counter = 0
        
        # Safety configuration
        self.cost_threshold = cost_threshold
        self.verbose = verbose
        
        # Safety configuration with tunable parameters
        self.safety_config = {
            'collision_penalty': 1.0,        # High cost for collisions
            'lane_departure_penalty': 0.8,   # Moderate cost for leaving lane
            'speed_violation_penalty': 0.3,  # Lower cost for speed violations
            'proximity_penalty': 0.2,        # Low cost for being too close to other vehicles
            'desired_speed_factor': 1.2,     # Speed threshold multiplier
        }
        
        # Reward configuration with tunable parameters - focused on turn handling
        self.reward_config = {
            'lane_center_factor': 5.0,      # Heavily prioritize lane keeping
            'steering_smoothness_factor': 0.3, # Less penalty for quick steering
            'speed_factor': 0.5,            # Reduce importance of speed
            'curve_discount': 1.2,          # More lenient on curves
            'turn_speed_penalty': 1.0,      # Penalty for high speed in turns
            'curve_anticipation_reward': 0.5, # Reward for steering into curve
            'progress_factor': 1.0,         # Higher progress reward to encourage movement
            'stationary_penalty': 1.0       # Penalty for not moving
        }
        
        # Tracking variables for safety metrics
        self.cumulative_cost = 0
        self.cumulative_reward = 0
        
        # Detailed safety metrics
        self.safety_metrics = {
            'total_collisions': 0,
            'lane_departures': 0,
            'speed_violations': 0,
            'proximity_violations': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'avg_lane_deviation': [],
            'avg_speed': [],
            'lane_time_ratio': 0.0,  # Percentage of time in lane
            'curve_speeds': [],       # Track speeds during curves
            'straight_speeds': []     # Track speeds on straight roads
        }
        
        # History tracking for visualization
        self.history = {
            'lane_deviations': [],
            'speeds': [],
            'steering_angles': [],
            'rewards': [],
            'is_curve': []           # Track when the car is on a curve
        }
        
        # Reward debugging
        self.reward_debug = {}
        self.episode_rewards = []
        self.episode_costs = []
        
        # Track previous steering and lane position for smoothness calculation
        self._last_steering = 0.0
        self._last_lane_pos = 0.0
        self._last_waypoint_idx = 0
        
        # Maintain original spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Create directory for logging
        self.log_dir = "carla_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Print environment details
        self._print_env_details()
    
    def _print_env_details(self):
        """Print detailed information about the environment"""
        print("\n=== Carla Environment Details ===")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        
        # Print observation space details if it's a Dict
        if isinstance(self.observation_space, spaces.Dict):
            print("\nObservation Space Components:")
            for key, space in self.observation_space.spaces.items():
                print(f"  {key}: {space}")
        
        print(f"\nMax episode steps: {self.max_episode_steps}")
        print(f"Cost threshold: {self.cost_threshold}")
        
        # Print reward configuration
        print("\nReward Configuration:")
        for key, value in self.reward_config.items():
            print(f"  {key}: {value}")
        
        # Print any accessible environment parameters
        try:
            if hasattr(self.env, 'desired_speed'):
                print(f"Desired speed: {self.env.desired_speed}")
            if hasattr(self.env, 'out_lane_thres'):
                print(f"Lane threshold: {self.env.out_lane_thres}")
            if hasattr(self.env, 'number_of_vehicles'):
                print(f"Number of vehicles: {self.env.number_of_vehicles}")
        except:
            pass
    
    def _calculate_lane_reward(self, lane_distance, is_curve=False, current_speed=None):
        """
        Calculate reward for staying in lane, with special handling for curves.
        
        Args:
            lane_distance: Absolute distance from lane center
            is_curve: Whether the vehicle is in a curve
            current_speed: Current vehicle speed for scaling rewards
            
        Returns:
            float: Lane keeping reward
        """
        # Maximum distance before no reward
        max_distance = self.env.out_lane_thres
        
        # If on a curve, be more lenient with lane position
        if is_curve:
            # Increase allowed distance in curves by the curve_discount factor
            effective_max = max_distance * (1 + self.reward_config['curve_discount'])
            # Sigmoid function to smoothly transition from full reward to no reward
            lane_reward = 1.0 / (1.0 + np.exp(5 * (lane_distance / effective_max - 0.5)))
        else:
            # For straight roads, use a quadratic function that rewards being exactly centered
            normalized_dist = min(1.0, lane_distance / max_distance)
            lane_reward = 1.0 - normalized_dist ** 2
        
        # Amplify the reward using the lane_center_factor
        lane_reward = lane_reward * self.reward_config['lane_center_factor']
        
        # Scale down rewards if vehicle is not moving
        if current_speed is not None and current_speed < 1.0:
            scale_factor = max(0.1, current_speed)  # Ensure minimum 10% reward
            lane_reward = lane_reward * scale_factor
        
        return lane_reward
    
    def _is_on_curve(self):
        """
        Determine if the vehicle is on a curve based on waypoints.
        Works with waypoint format as list of [x, y, angle] lists.
        
        Returns:
            bool: True if on a curve, False otherwise
        """
        try:
            # Need at least 5 waypoints to determine curvature
            if not hasattr(self.env, 'waypoints') or len(self.env.waypoints) < 5:
                return False
                
            waypoints = self.env.waypoints
            
            # We know waypoints are a list of [x, y, angle] lists
            if not isinstance(waypoints, list) or not all(isinstance(wp, list) for wp in waypoints):
                return False
                
            # Extract positions (first two elements of each waypoint are x,y)
            positions = []
            for wp in waypoints[:min(8, len(waypoints))]:  # Look at more waypoints ahead
                if len(wp) >= 2:
                    positions.append((wp[0], wp[1]))
                    
            # If we couldn't extract enough positions, fallback to straight road
            if len(positions) < 5:
                return False
                
            # Calculate change in direction between consecutive positions
            angles = []
            max_angle = 0.0
            for i in range(len(positions) - 2):
                p1 = positions[i]
                p2 = positions[i+1]
                p3 = positions[i+2]
                
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
                    angle = np.arccos(dot_product)
                    angles.append(angle)
                    max_angle = max(max_angle, angle)
            
            # Consider it a curve if either:
            # 1. The average angle change is significant (> 4 degrees)
            # 2. The maximum angle change is sharp (> 8 degrees)
            avg_angle = np.mean(angles) if angles else 0
            avg_angle_degrees = np.degrees(avg_angle)
            max_angle_degrees = np.degrees(max_angle)
            
            is_curve = avg_angle_degrees > 4.0 or max_angle_degrees > 8.0
            
            # Log curve detection occasionally
            if self._step_counter % 100 == 0:
                print(f"Road curvature detection - Avg angle: {avg_angle_degrees:.2f}°, Max angle: {max_angle_degrees:.2f}°, Is curve: {is_curve}")
                
            return is_curve
            
        except Exception as e:
            if self.verbose:
                print(f"Error in curve detection: {e}")
            return False
    
    def _get_curve_direction(self):
        """Determine curve direction based on waypoints."""
        try:
            if hasattr(self.env, 'waypoints') and len(self.env.waypoints) >= 3:
                wp1 = self.env.waypoints[0]
                wp3 = self.env.waypoints[2]
                
                if len(wp1) >= 2 and len(wp3) >= 2:
                    dx = wp3[0] - wp1[0]
                    dy = wp3[1] - wp1[1]
                    
                    # For simple direction, we'll use dx since that's the primary indicator
                    # in the coordinate system used
                    return "right" if dx > 0 else "left"
        except Exception as e:
            if self.verbose:
                print(f"Error in curve direction detection: {e}")
        
        return None
    
    def _calculate_steering_smoothness(self, current_steering):
        """
        Calculate reward for smooth steering changes.
        
        Args:
            current_steering: Current steering angle
            
        Returns:
            float: Steering smoothness reward
        """
        # Calculate absolute change in steering
        steering_change = abs(current_steering - self._last_steering)
        
        # Higher penalty for larger changes, scaled to be negative
        smoothness_reward = -steering_change * self.reward_config['steering_smoothness_factor']
        
        # Update last steering
        self._last_steering = current_steering
        
        return smoothness_reward
        
    def _calculate_progress_reward(self):
        """
        Calculate reward for making progress along the route.
        Adapted for waypoints as a list of coordinate lists.
        
        Returns:
            float: Progress reward
        """
        try:
            # If waypoints are available, check progress by looking at current position
            if hasattr(self.env, 'waypoints') and hasattr(self.env, 'ego'):
                from gym_carla.envs.misc import get_pos
                
                # Get current position
                ego_x, ego_y = get_pos(self.env.ego)
                
                # Find the closest waypoint index
                min_dist = float('inf')
                closest_idx = 0
                
                for i, wp in enumerate(self.env.waypoints):
                    if len(wp) >= 2:
                        dx = wp[0] - ego_x
                        dy = wp[1] - ego_y
                        dist = dx*dx + dy*dy  # Squared distance
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                
                # Reward for advancing waypoints
                waypoint_delta = closest_idx - self._last_waypoint_idx
                progress_reward = max(0, waypoint_delta) * self.reward_config['progress_factor']
                
                # Update last waypoint index
                self._last_waypoint_idx = closest_idx
                
                return progress_reward
            
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error calculating progress reward: {e}")
            return 0.0
    
    def _calculate_curve_speed_reward(self, speed, is_curve):
        """
        Calculate reward for appropriate speed during curves.
        
        Args:
            speed: Current vehicle speed
            is_curve: Whether the vehicle is on a curve
            
        Returns:
            float: Speed reward
        """
        if not is_curve:
            return 0.0  # No curve speed reward on straight roads
            
        # Calculate appropriate speed for curves - lower is better on curves
        desired_curve_speed = self.env.desired_speed * 0.6  # 60% of normal speed
        
        if speed <= desired_curve_speed:
            # Good - driving slowly in curves
            return 0.5
        else:
            # Penalty for high speed in curves
            excess_speed = (speed - desired_curve_speed) / desired_curve_speed
            speed_penalty = -excess_speed * self.reward_config['turn_speed_penalty']
            return speed_penalty
    
    def _calculate_reward_components(self, obs, info):
        """
        Calculate and track reward components - this helps with debugging.
        
        Returns:
            dict: Reward components
        """
        reward_components = {}
        
        try:
            # Check if vehicle is on a curve
            is_curve = self._is_on_curve()
            curve_direction = self._get_curve_direction() if is_curve else None
            
            # Store curve info in the info dictionary
            info['is_curve'] = is_curve
            info['curve_direction'] = curve_direction
            
            # Track curve status
            self.history['is_curve'].append(is_curve)
            
            # 1. Forward Progress Reward
            if hasattr(self.env, 'ego'):
                v = self.env.ego.get_velocity()
                speed = np.sqrt(v.x**2 + v.y**2)
                desired_speed = self.env.desired_speed
                
                # Track speed
                self.history['speeds'].append(speed)
                self.performance_metrics['avg_speed'].append(speed)
                
                # Add stationary penalty if not moving
                if speed < 0.5:
                    reward_components['stationary_penalty'] = -self.reward_config['stationary_penalty']
                
                # Track speeds on curves vs straight roads
                if is_curve:
                    self.performance_metrics['curve_speeds'].append(speed)
                else:
                    self.performance_metrics['straight_speeds'].append(speed)
                
                # Basic reward for moving forward at appropriate speed
                speed_reward = 0.0
                
                # Reward peaks at desired speed on straight roads
                if not is_curve:
                    if speed <= desired_speed:
                        # Linearly increase reward up to desired speed
                        speed_reward = speed / desired_speed
                    else:
                        # Penalize speeds above desired speed
                        excess = (speed - desired_speed) / desired_speed
                        speed_reward = max(0, 1 - excess)
                else:
                    # On curves, reward lower speeds
                    curve_speed = desired_speed * 0.6  # 60% of normal speed
                    if speed <= curve_speed:
                        # Good - driving slowly in curves
                        speed_reward = speed / curve_speed
                    else:
                        # Still allow some reward, but reduced for excessive speed
                        excess = (speed - curve_speed) / curve_speed
                        speed_reward = max(0, 1 - excess * 2)  # Steeper penalty in curves
                
                # Apply speed factor
                speed_reward *= self.reward_config['speed_factor']
                reward_components['speed_reward'] = speed_reward
                
                # Add specific curve speed reward/penalty
                curve_speed_reward = self._calculate_curve_speed_reward(speed, is_curve)
                if curve_speed_reward != 0:
                    reward_components['curve_speed'] = curve_speed_reward
            
            # 2. Lane Keeping Reward
            try:
                from gym_carla.envs.misc import get_pos, get_lane_dis
                
                ego_x, ego_y = get_pos(self.env.ego)
                dis, _ = get_lane_dis(self.env.waypoints, ego_x, ego_y)
                
                # Calculate enhanced lane reward - include speed for scaling
                current_speed = np.sqrt(self.env.ego.get_velocity().x**2 + 
                                        self.env.ego.get_velocity().y**2)
                lane_reward = self._calculate_lane_reward(abs(dis), is_curve, current_speed)
                reward_components['lane_reward'] = lane_reward
                
                # Track lane deviation
                self.history['lane_deviations'].append(abs(dis))
                self.performance_metrics['avg_lane_deviation'].append(abs(dis))
                
                # Track lane time ratio (time spent within lane)
                in_lane = abs(dis) < self.env.out_lane_thres
                total_steps = max(1, len(self.history['lane_deviations']))
                self.performance_metrics['lane_time_ratio'] = (
                    self.performance_metrics['lane_time_ratio'] * (total_steps - 1) + float(in_lane)
                ) / total_steps
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating lane reward: {e}")
            
            # 3. Steering Smoothness Reward
            if hasattr(self.env, 'ego'):
                curr_steering = self.env.ego.get_control().steer
                
                # Track steering angle
                self.history['steering_angles'].append(curr_steering)
                
                # Calculate steering smoothness reward
                steering_reward = self._calculate_steering_smoothness(curr_steering)
                reward_components['steering_smoothness'] = steering_reward
                
                # Add curve anticipation reward - reward steering in the direction of the curve
                if is_curve and hasattr(self.env, 'waypoints') and len(self.env.waypoints) >= 3:
                    try:
                        # Determine curve direction from waypoints
                        wp1 = self.env.waypoints[0]
                        wp3 = self.env.waypoints[2]
                        
                        if len(wp1) >= 2 and len(wp3) >= 2:
                            # Positive dx means curve to right, negative to left
                            dx = wp3[0] - wp1[0]
                            curve_direction = np.sign(dx)
                            
                            # Check if steering matches curve direction
                            if np.sign(curr_steering) == curve_direction and abs(curr_steering) > 0.1:
                                # Reward steering in correct direction
                                curve_anticipation = self.reward_config['curve_anticipation_reward']
                                reward_components['curve_anticipation'] = curve_anticipation
                    except Exception as e:
                        if self.verbose:
                            print(f"Error calculating curve anticipation: {e}")
            
            # 4. Progress Reward
            progress_reward = self._calculate_progress_reward()
            if progress_reward > 0:
                reward_components['progress'] = progress_reward
        
        except Exception as e:
            if self.verbose:
                print(f"Error calculating reward components: {e}")
        
        return reward_components
    
    def _calculate_total_reward(self, reward_components):
        """
        Calculate the final reward based on all components.
        
        Args:
            reward_components: Dictionary of reward components
            
        Returns:
            float: Total reward
        """
        # Sum all reward components
        total_reward = sum(reward_components.values())
        
        # Track total reward
        self.history['rewards'].append(total_reward)
        
        return total_reward
    
    def _calculate_safety_cost(self, obs, info):
        """
        Calculate comprehensive safety cost for the current step.
        
        Args:
            obs: Current observation
            info: Environment info dictionary
        
        Returns:
            float: Calculated safety cost
        """
        cost = 0.0
        debug_details = {}
        
        try:
            # 1. Collision Cost
            if hasattr(self.env, 'collision_hist') and len(self.env.collision_hist) > 0:
                collision_cost = self.safety_config['collision_penalty'] * len(self.env.collision_hist)
                cost += collision_cost
                self.safety_metrics['total_collisions'] += 1
                debug_details['collision'] = {
                    'count': len(self.env.collision_hist),
                    'cost': collision_cost
                }
                if self.verbose:
                    print(f"🚨 Collision detected! Cost: {collision_cost}")
        except Exception as e:
            if self.verbose:
                print(f"Collision cost calculation error: {e}")
        
        try:
            # 2. Lane Departure Cost
            from gym_carla.envs.misc import get_pos, get_lane_dis
            
            ego_x, ego_y = get_pos(self.env.ego)
            dis, _ = get_lane_dis(self.env.waypoints, ego_x, ego_y)
            
            if abs(dis) > self.env.out_lane_thres:
                lane_cost = self.safety_config['lane_departure_penalty']
                cost += lane_cost
                self.safety_metrics['lane_departures'] += 1
                debug_details['lane_departure'] = {
                    'distance': dis,
                    'threshold': self.env.out_lane_thres,
                    'cost': lane_cost
                }
                if self.verbose:
                    print(f"⚠️ Lane departure! Distance: {dis:.2f}, Cost: {lane_cost}")
        except Exception as e:
            if self.verbose:
                print(f"Lane departure cost calculation error: {e}")

        try:
            # 3. Speed Violation Cost
            v = self.env.ego.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2)
            desired_speed = self.env.desired_speed
            
            # Adjust speed threshold based on curve status
            is_curve = info.get('is_curve', self._is_on_curve())
            if is_curve:
                # Lower speed threshold on curves
                speed_threshold = desired_speed * 0.8  # 20% lower on curves
            else:
                speed_threshold = desired_speed * self.safety_config['desired_speed_factor']

            if speed > speed_threshold:
                speed_cost = self.safety_config['speed_violation_penalty']
                cost += speed_cost
                self.safety_metrics['speed_violations'] += 1
                debug_details['speed_violation'] = {
                    'current_speed': speed,
                    'desired_speed': desired_speed,
                    'threshold': speed_threshold,
                    'cost': speed_cost
                }
                if self.verbose:
                    print(f"🏎️ Speed violation! Current: {speed:.2f}, Threshold: {speed_threshold:.2f}, Cost: {speed_cost}")
        except Exception as e:
            if self.verbose:
                print(f"Speed violation cost calculation error: {e}")

        try:
            # 4. Proximity to Other Vehicles Cost
            if hasattr(self.env, 'vehicle_front') and self.env.vehicle_front:
                proximity_cost = self.safety_config['proximity_penalty']
                cost += proximity_cost
                self.safety_metrics['proximity_violations'] += 1
                debug_details['proximity'] = {
                    'vehicle_front': True,
                    'cost': proximity_cost
                }
                if self.verbose:
                    print(f"🚦 Close to vehicle! Cost: {proximity_cost}")
                    
            # 5. Add cost for remaining stationary
            v = self.env.ego.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2)
            if speed < 0.5:
                stationary_cost = 0.5  # Add cost for not moving
                cost += stationary_cost
                debug_details['stationary'] = {
                    'cost': stationary_cost
                }
                if self.verbose and self._step_counter % 20 == 0:
                    print(f"⚠️ Vehicle not moving! Speed: {speed:.2f}, Cost: {stationary_cost}")
        except Exception as e:
            if self.verbose:
                print(f"Other cost calculation error: {e}")

        # Return the total cost
        return cost
    
    def plot_performance_metrics(self):
        """
        Plot performance metrics for the episode.
        """
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 20))
            
            # Plot 1: Lane deviation over time
            if self.history['lane_deviations']:
                axes[0].plot(self.history['lane_deviations'])
                axes[0].axhline(y=self.env.out_lane_thres, color='r', linestyle='--', label='Lane Threshold')
                
                # Highlight curve areas
                if self.history['is_curve']:
                    for i, is_curve in enumerate(self.history['is_curve']):
                        if is_curve:
                            axes[0].axvspan(i, i+1, alpha=0.2, color='green')
                
                axes[0].set_title('Lane Deviation Over Time (Green areas are curves)')
                axes[0].set_xlabel('Steps')
                axes[0].set_ylabel('Lane Deviation')
                axes[0].grid(True)
                axes[0].legend()
            
            # Plot 2: Speed over time
            if self.history['speeds']:
                axes[1].plot(self.history['speeds'])
                axes[1].axhline(y=self.env.desired_speed, color='g', linestyle='--', label='Desired Speed')
                # Add curve speed threshold
                axes[1].axhline(y=self.env.desired_speed * 0.6, color='y', linestyle='--', label='Curve Speed')
                
                # Highlight curve areas
                if self.history['is_curve']:
                    for i, is_curve in enumerate(self.history['is_curve']):
                        if is_curve:
                            axes[1].axvspan(i, i+1, alpha=0.2, color='green')
                
                axes[1].set_title('Speed Over Time (Green areas are curves)')
                axes[1].set_xlabel('Steps')
                axes[1].set_ylabel('Speed')
                axes[1].grid(True)
                axes[1].legend()
            
            # Plot 3: Steering angles over time
            if self.history['steering_angles']:
                axes[2].plot(self.history['steering_angles'])
                
                # Highlight curve areas
                if self.history['is_curve']:
                    for i, is_curve in enumerate(self.history['is_curve']):
                        if is_curve:
                            axes[2].axvspan(i, i+1, alpha=0.2, color='green')
                
                axes[2].set_title('Steering Angle Over Time (Green areas are curves)')
                axes[2].set_xlabel('Steps')
                axes[2].set_ylabel('Steering Angle')
                axes[2].grid(True)
            
            # Plot 4: Speed distribution on curves vs straight roads
            if self.performance_metrics['curve_speeds'] and self.performance_metrics['straight_speeds']:
                axes[3].hist(self.performance_metrics['curve_speeds'], bins=20, alpha=0.5, label='Curves', color='green')
                axes[3].hist(self.performance_metrics['straight_speeds'], bins=20, alpha=0.5, label='Straight Roads', color='blue')
                axes[3].axvline(x=self.env.desired_speed, color='r', linestyle='--', label='Desired Speed')
                axes[3].axvline(x=self.env.desired_speed * 0.6, color='y', linestyle='--', label='Ideal Curve Speed')
                axes[3].set_title('Speed Distribution: Curves vs Straight Roads')
                axes[3].set_xlabel('Speed')
                axes[3].set_ylabel('Frequency')
                axes[3].grid(True)
                axes[3].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"episode_{len(self.episode_rewards)}_metrics.png"))
            plt.close()
            
        except Exception as e:
            if self.verbose:
                print(f"Error plotting performance metrics: {e}")

    def reset(self):
        """
        Reset the environment and safety tracking.

        Returns:
            Initial observation
        """
        # Plot metrics from previous episode
        if len(self.history['lane_deviations']) > 0:
            self.plot_performance_metrics()
        
        # Reset step counter and metrics
        self._step_counter = 0
        self.cumulative_cost = 0
        self.cumulative_reward = 0

        # Reset safety metrics
        self.safety_metrics = {
            'total_collisions': 0,
            'lane_departures': 0,
            'speed_violations': 0,
            'proximity_violations': 0
        }
        
        # Reset performance metrics
        self.performance_metrics = {
            'avg_lane_deviation': [],
            'avg_speed': [],
            'lane_time_ratio': 0.0,
            'curve_speeds': [],
            'straight_speeds': []
        }
        
        # Reset history
        self.history = {
            'lane_deviations': [],
            'speeds': [],
            'steering_angles': [],
            'rewards': [],
            'is_curve': []
        }

        # Reset reward debugging
        self.reward_debug = {}
        
        # Print episode summary if we have data
        if hasattr(self, 'episode_rewards') and len(self.episode_rewards) > 0:
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_costs.append(self.cumulative_cost)
            
            if len(self.episode_rewards) % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-5:])
                avg_cost = np.mean(self.episode_costs[-5:])
                print(f"\n=== Last 5 Episodes Stats ===")
                print(f"Average reward: {avg_reward:.2f}")
                print(f"Average cost: {avg_cost:.2f}")
                
                if len(self.performance_metrics['avg_lane_deviation']) > 0:
                    avg_deviation = np.mean(self.performance_metrics['avg_lane_deviation'])
                    print(f"Average lane deviation: {avg_deviation:.2f}")
                    print(f"Lane time ratio: {self.performance_metrics['lane_time_ratio']:.2f}")

        # Reset tracking variables
        self._last_steering = 0.0
        self._last_lane_pos = 0.0
        self._last_waypoint_idx = 0
        
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the environment with safety cost tracking.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Increment step counter
        self._step_counter += 1
        
        # Force acceleration if the car is not moving (prevent getting stuck)
        if hasattr(self.env, 'ego'):
            v = self.env.ego.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2)
            if speed < 0.5:
                # Add acceleration component to avoid stopping
                if action[0] < 0.3:  # If not already accelerating
                    action = action.copy()  # Create a copy to avoid modifying the original
                    action[0] = max(action[0], 0.5)  # Force minimum acceleration

        # Take step in the environment
        obs, reward, done, info = self.env.step(action)

        # Check if vehicle is on a curve for consistent reporting
        is_curve = self._is_on_curve()
        curve_direction = self._get_curve_direction() if is_curve else None
        
        # Store curve info in the info dictionary
        info['is_curve'] = is_curve
        info['curve_direction'] = curve_direction

        # Calculate reward components for debugging
        self.reward_debug = self._calculate_reward_components(obs, info)
        
        # Calculate total reward from components
        numeric_reward = self._calculate_total_reward(self.reward_debug)
        
        # Debug print reward components periodically
        if self._step_counter % 20 == 0 and self.verbose:
            print("\n--- Reward Components ---")
            for key, value in self.reward_debug.items():
                print(f"{key}: {value:.4f}")

        # Calculate safety cost
        safety_cost = self._calculate_safety_cost(obs, info)

        # Update cumulative metrics
        self.cumulative_cost += safety_cost
        self.cumulative_reward += numeric_reward

        # Check if cost threshold is exceeded
        if self.cumulative_cost > self.cost_threshold:
            done = True
            info['safety_limit_exceeded'] = True
            if self.verbose:
                print(f"🛑 Safety cost threshold exceeded: {self.cumulative_cost:.2f}")

        # Time limit handling
        timeout = self._step_counter >= self.max_episode_steps
        if timeout and not done:
            info['TimeLimit.truncated'] = True
            done = True
            if self.verbose:
                print(f"⏰ Maximum steps reached: {self._step_counter}")

        # Add detailed info to info dict
        info['cost'] = safety_cost
        info['cumulative_cost'] = self.cumulative_cost
        info['safety_metrics'] = self.safety_metrics
        info['reward_debug'] = self.reward_debug
        info['performance_metrics'] = {
            'avg_lane_deviation': np.mean(self.performance_metrics['avg_lane_deviation']) if self.performance_metrics['avg_lane_deviation'] else 0,
            'avg_speed': np.mean(self.performance_metrics['avg_speed']) if self.performance_metrics['avg_speed'] else 0,
            'lane_time_ratio': self.performance_metrics['lane_time_ratio']
        }

        # Detailed logging
        if self.verbose and self._step_counter % 20 == 0:
            print(f"Step {self._step_counter}: "
                  f"Reward: {numeric_reward:.2f}, "
                  f"Cost: {safety_cost:.2f}, "
                  f"Cumulative Cost: {self.cumulative_cost:.2f}")
            
            # Print speed
            if hasattr(self.env, 'ego'):
                v = self.env.ego.get_velocity()
                speed = np.sqrt(v.x**2 + v.y**2)
                print(f"  Current speed: {speed:.2f}, Desired speed: {self.env.desired_speed}")

        return obs, numeric_reward, done, info

def make_carla_env(params=None, max_episode_steps=1000, cost_threshold=5.0, verbose=True):
    """
    Create a Carla environment with safety constraints.

    Args:
        params (dict, optional): Environment configuration parameters
        max_episode_steps (int): Maximum steps per episode
        cost_threshold (float): Cumulative safety cost threshold
        verbose (bool): Enable detailed logging

    Returns:
        Wrapped Carla environment
    """
    # Default environment parameters - simplified for turn learning
    if params is None:
        params = {
            # Vehicle and world configuration - empty roads for learning turns
            'number_of_vehicles': 0,
            'number_of_walkers': 0,
            'display_size': 256,
            'max_past_step': 1,
            'dt': 0.1,

            # Control space configuration - wider steering range
            'discrete': False,
            'discrete_acc': [-3.0, 0.0, 3.0],
            'discrete_steer': [-0.2, 0.0, 0.2],
            'continuous_accel_range': [-3.0, 3.0],
            'continuous_steer_range': [-0.5, 0.5],  # Wider range for better turns

            # Ego vehicle configuration
            'ego_vehicle_filter': 'vehicle.lincoln*',

            # Server and world parameters - Town05 has more curved roads
            'port': 2000,
            'town': 'Town05',  # Changed from Town02 to one with more curves
            'task_mode': 'random',

            # Episode and navigation parameters
            'max_time_episode': 1000,
            'max_waypt': 12,
            'obs_range': 32,
            'lidar_bin': 0.125,
            'd_behind': 12,

            # Safety and performance parameters - more forgiving
            'out_lane_thres': 2.5,  # More forgiving lane threshold during training
            'desired_speed': 5,     # Moderate speed for easier training
            'max_ego_spawn_times': 200,

            # Rendering and perception
            'display_route': True,
            'pixor_size': 64,
            'pixor': False
        }

    # Create the Carla environment
    env = gym.make('carla-v0', params=params)

    # Wrap the environment with safety constraints
    wrapped_env = CarlaWrapper(
        env,
        max_episode_steps=max_episode_steps,
        cost_threshold=cost_threshold,
        verbose=verbose
    )

    return wrapped_env
