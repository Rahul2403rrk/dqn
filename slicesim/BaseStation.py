import numpy as np

class BaseStation:
    def __init__(self, id, pk, coverage, capacity_bandwidth, slices=None):
        self.pk = pk
        self.id = id
        self.coverage = coverage
        self.capacity_bandwidth = capacity_bandwidth
        self.slices = slices
        print(self)

    def __str__(self):
        return f'BS_{self.pk:<2}\t cov:{str(self.coverage)}\t with cap {self.capacity_bandwidth:<5}'

    def get_observation(self):
        return np.array([
            self.capacity_bandwidth,
            self.slices[0].connected_users,
            self.slices[1].connected_users,
            self.slices[2].connected_users
        ], dtype=np.float32)

    def apply_action(self, allocation_dict):
        for slice_id, prbs in allocation_dict.items():
            self.slices[slice_id].assign_prbs(prbs)

    def compute_total_reward(self):
        return sum(slice.compute_reward() for slice in self.slices.values())
