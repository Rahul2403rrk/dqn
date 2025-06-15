import matplotlib.pyplot as plt
import numpy as np

timesteps = 100
rewards = np.cumsum(np.random.randn(timesteps) * 5 + 20) / np.arange(1, timesteps + 1)
losses = np.exp(-np.linspace(0, 5, timesteps)) * 10 + np.random.rand(timesteps)

plt.figure(figsize=(10, 6))

plt.plot(rewards, label='Average Reward', linewidth=2)
plt.plot(losses, label='Training Loss', linewidth=2)

plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title("DQN Training: Reward vs. Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_vs_loss.png")
plt.show()
