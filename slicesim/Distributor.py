# distributor_dqn.py
from dqn_agent import DQNAgent
import numpy as np

class DQNDistributor:
    def __init__(self, base_station, num_slices):
        self.base_station = base_station
        self.num_slices = num_slices

        # Discretize action space (e.g. percent combinations that sum to 100)
        self.actions = self.generate_percent_actions()
        self.agent = DQNAgent(state_size=self.base_station.state_size, action_size=len(self.actions))

    def generate_percent_actions(self):
        # Example: return all combinations of 3 slices that sum to 100 in steps of 25
        actions = []
        for a in range(0, 101, 25):
            for b in range(0, 101 - a, 25):
                c = 100 - a - b
                if c >= 0:
                    actions.append([a, b, c])
        return actions

    def allocate(self):
        state = self.base_station.get_observation()
        action_index = self.agent.act(state)
        selected_percentages = self.actions[action_index]

        allocation_dict = {}
        total_bw = self.base_station.capacity_bandwidth

        for i, pct in enumerate(selected_percentages):
            slice_id = list(self.base_station.slices.keys())[i]
            allocation_dict[slice_id] = (pct / 100.0) * total_bw

        self.base_station.apply_action(allocation_dict)

        # Simulate environment step here
        next_state = self.base_station.get_observation()
        reward = self.base_station.compute_total_reward()  # You must implement this
        done = False  # Adjust based on simulation logic
        self.agent.remember(state, action_index, reward, next_state, done)
        self.agent.replay(batch_size=32)
