import torch
import torch.serialization
from safe_rl.policy.model.mlp_ac import CholeskyGaussianActor, EnsembleQCritic
import numpy as np
import time
import matplotlib.pyplot as plt

# Add the necessary classes to the safe globals list
torch.serialization.add_safe_globals([CholeskyGaussianActor, EnsembleQCritic])

# Load the trained model
model_path = "/home/prakhargautam/experiments/cvpo-safe-rl/data/EnhancedSafeHighway-v0_cost_10/cvpo_epoch_10/cvpo_epoch_10_s0//model_save/model.pt"
model = torch.load(model_path, weights_only=False)
print("Model loaded successfully!")

# Get the actual model components
actor, critic, qc = model
print(f"Actor type: {type(actor)}")

# Direct instantiation of the environment
from highway_safety_gym.envs.enhanced_highway_env import EnhancedHighwaySafetyEnv

# Create the environment directly without gym.make()
env = EnhancedHighwaySafetyEnv(cost_limit=10)

# For visualization
def visualize_episode(trajectory, obstacles):
    """Visualize an episode's trajectory"""
    plt.figure(figsize=(10, 10))
    
    # Plot environment boundaries
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.8, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0.8, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=-0.8, color='r', linestyle='--', alpha=0.3)
    
    # Plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle((obstacle["x"], obstacle["y"]), 
                           obstacle["radius"], 
                           color='red', alpha=0.5)
        plt.gca().add_patch(circle)
    
    # Plot trajectory
    x_vals = [point[0] for point in trajectory]
    y_vals = [point[1] for point in trajectory]
    
    # Color gradient for trajectory (blue to green)
    colors = np.linspace(0, 1, len(trajectory))
    plt.scatter(x_vals, y_vals, c=colors, cmap='cool', s=10)
    plt.plot(x_vals, y_vals, 'k-', alpha=0.3)
    
    # Mark start and end
    plt.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Start')
    plt.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label='End')
    
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.title('Vehicle Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'trajectory_episode_{episode+1}.png')
    plt.close()

# Run evaluation loop
episodes = 10
total_rewards = []
total_costs = []
cost_breakdown = {
    "boundary": [],
    "obstacle": [],
    "speed": []
}

for episode in range(episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_cost = 0
    step_count = 0
    episode_costs_breakdown = {
        "boundary": 0,
        "obstacle": 0,
        "speed": 0
    }
    
    while not done:
        # Visualize current state
        env.render()
        
        # Use the trained policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get action from the trained policy
        with torch.no_grad():
            mean, _ = actor(obs_tensor)
            action = mean.squeeze().numpy()
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        
        # Accumulate reward and cost
        episode_reward += reward
        if 'cost' in info:
            episode_cost += info['cost']
        
        # Track cost breakdown
        if 'cost_factors' in info:
            for key, value in info['cost_factors'].items():
                episode_costs_breakdown[key] += value
        
        step_count += 1
        time.sleep(0.1)  # Slow down for visualization
            
    # Record metrics
    total_rewards.append(episode_reward)
    total_costs.append(episode_cost)
    
    # Record cost breakdowns
    for key in episode_costs_breakdown:
        cost_breakdown[key].append(episode_costs_breakdown[key])
    
    # Visualize trajectory
    visualize_episode(env.trajectory, env.obstacles)
    
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, Steps = {step_count}")
    print(f"Cost breakdown: {episode_costs_breakdown}")

print(f"\nAverage Reward: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"Average Cost: {sum(total_costs)/len(total_costs):.2f}")
print(f"Average cost breakdown:")
for key in cost_breakdown:
    avg = sum(cost_breakdown[key])/len(cost_breakdown[key])
    print(f"  {key}: {avg:.2f}")

# Plot performance metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(range(1, episodes+1), total_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(range(1, episodes+1), total_costs)
plt.title('Episode Costs')
plt.xlabel('Episode')
plt.ylabel('Total Cost')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
for key in cost_breakdown:
    plt.plot(range(1, episodes+1), cost_breakdown[key], label=key)
plt.title('Cost Breakdown')
plt.xlabel('Episode')
plt.ylabel('Cost Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_metrics.png')
plt.show()
