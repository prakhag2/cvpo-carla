import pandas as pd
import matplotlib.pyplot as plt

# Load the data
log_path = "/home/prakhargautam/experiments/cvpo-safe-rl/data/SafeHighwayV2-v0_cost_10/cvpo_epoch_300/cvpo_epoch_300_s0/progress.txt"
data = pd.read_table(log_path)

# Plot rewards
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data['TotalEnvInteracts'], data['EpRet'], label='Return')
plt.plot(data['TotalEnvInteracts'], data['TestEpRet'], label='Test Return')
plt.legend()
plt.title('Returns')

# Plot costs
plt.subplot(3, 1, 2)
plt.plot(data['TotalEnvInteracts'], data['EpCost'], label='Cost')
plt.plot(data['TotalEnvInteracts'], data['TestEpCost'], label='Test Cost')
plt.axhline(y=10, color='r', linestyle='--', label='Cost Limit')
plt.legend()
plt.title('Safety Costs')

# Plot episode lengths
plt.subplot(3, 1, 3)
plt.plot(data['TotalEnvInteracts'], data['EpLen'], label='Length')
plt.legend()
plt.title('Episode Length')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
