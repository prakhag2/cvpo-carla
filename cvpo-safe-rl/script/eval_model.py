import torch
import torch.serialization
from safe_rl.policy.model.mlp_ac import CholeskyGaussianActor, EnsembleQCritic
import numpy as np
import time

# Add the necessary classes to the safe globals list
torch.serialization.add_safe_globals([CholeskyGaussianActor, EnsembleQCritic])

# Load and verify the model
model_path = "/home/prakhargautam/experiments/cvpo-safe-rl/data/SafeHighwayV2-v0_cost_10/cvpo_epoch_300/cvpo_epoch_300_s0/model_save/model.pt"
model = torch.load(model_path, weights_only=False)
print("Model loaded successfully!")

# Get the actual model components
actor, critic, qc = model
print(f"Actor type: {type(actor)}")

# Create our own simplified environment directly
class SimpleHighwayEnv:
    def __init__(self, cost_limit=10):
        # Define state space and action space
        self.cost_limit = cost_limit
        self.current_cost = 0
        self.state = None
        self.steps = 0
        self.max_steps = 100
        
    def step(self, action):
        # Simple dynamics update
        self.steps += 1
        
        if self.state is None:
            self.reset()
        
        # Extract current state values
        x, y, vx, vy, heading = self.state
        
        # Extract steering and acceleration from action
        steering = action[0]  # -1 to 1
        acceleration = action[1]  # -1 to 1
        
        # Update heading based on steering
        heading += 0.1 * steering
        
        # Update velocity based on acceleration
        speed_change = 0.1 * acceleration
        vx += speed_change * np.cos(heading)
        vy += speed_change * np.sin(heading)
        
        # Update position based on velocity
        x += vx * 0.1
        y += vy * 0.1
        
        # Clip values 
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)
        vx = np.clip(vx, -5.0, 5.0)
        vy = np.clip(vy, -5.0, 5.0)
        heading = np.clip(heading, -np.pi, np.pi)
        
        # Update state
        self.state = np.array([x, y, vx, vy, heading], dtype=np.float32)
        
        # Calculate reward
        reward = float(1.0 - abs(x) - abs(y))  # Simple reward: higher when closer to center
        
        # Check for "crash" based on position
        crashed = abs(x) > 0.8 or abs(y) > 0.8
        
        # Calculate safety cost
        cost = 1 if crashed else 0
        self.current_cost += cost
        
        # Initialize termination flag
        done = False
        
        # Check termination conditions
        if self.current_cost > self.cost_limit:
            done = True
            reward -= 10  # Cost penalty
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            done = True
        
        # Create info dictionary
        info = {
            'cost': cost,
            'total_cost': self.current_cost,
            'crashed': crashed,
            'steps': self.steps
        }
        
        return self.state, reward, done, info

    def reset(self):
        # Reset to a random initial state
        self.state = np.array([
            np.random.uniform(-0.1, 0.1),  # x position
            np.random.uniform(-0.1, 0.1),  # y position
            np.random.uniform(-0.5, 0.5),  # x velocity
            np.random.uniform(-0.5, 0.5),  # y velocity
            np.random.uniform(-0.1, 0.1)   # heading
        ], dtype=np.float32)
        
        self.current_cost = 0
        self.steps = 0
        
        return self.state
        
    def render(self):
        print(f"State: {self.state}, Cost: {self.current_cost}")

# Create the environment
env = SimpleHighwayEnv(cost_limit=10)

# Run evaluation loop
episodes = 10
total_rewards = []
total_costs = []

for episode in range(episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_cost = 0
    step_count = 0
    
    while not done:
        # For visualization
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
        
        step_count += 1
        time.sleep(0.1)  # Slow down for visualization
            
    total_rewards.append(episode_reward)
    total_costs.append(episode_cost)
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, Steps = {step_count}")

print(f"\nAverage Reward: {sum(total_rewards)/len(total_rewards):.2f}")
print(f"Average Cost: {sum(total_costs)/len(total_costs):.2f}")
