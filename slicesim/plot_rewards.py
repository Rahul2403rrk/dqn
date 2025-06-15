import matplotlib.pyplot as plt
import numpy as np

# Simulated reward trend: noisy but improving
timesteps = 100
rewards = np.cumsum(np.random.randn(timesteps) * 5 + 20) / np.arange(1, timesteps + 1)

plt.figure(figsize=(10, 5))
plt.plot(rewards, label='DQN Agent Reward')
plt.xlabel("Timestep")
plt.ylabel("Average Reward")
plt.title("Simulated DQN Reward Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_reward_demo.png")  # Saves as image for presentation
plt.show()
