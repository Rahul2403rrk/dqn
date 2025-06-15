import random
import simpy  # ✅ import simpy for Container

class Slice:
    def __init__(self, name, env, ratio,
                 connected_users, user_share, delay_tolerance, qos_class,
                 bandwidth_guaranteed, bandwidth_max, init_capacity,
                 usage_pattern):
        self.name = name
        self.env = env  # ✅ store env
        self.connected_users = connected_users
        self.user_share = user_share
        self.delay_tolerance = delay_tolerance
        self.qos_class = qos_class
        self.ratio = ratio
        self.bandwidth_guaranteed = bandwidth_guaranteed
        self.bandwidth_max = bandwidth_max
        self.init_capacity = min(init_capacity, bandwidth_max)
        self.capacity = simpy.Container(env, init=self.init_capacity, capacity=self.bandwidth_max)
        self.usage_pattern = usage_pattern

    def get_consumable_share(self):
        if self.connected_users <= 0:
            return min(self.capacity.level, self.bandwidth_max)
        else:
            return min(self.capacity.level / self.connected_users, self.bandwidth_max)

    def is_avaliable(self):
        real_cap = min(self.capacity.level, self.bandwidth_max)
        bandwidth_next = real_cap / (self.connected_users + 1)
        return bandwidth_next >= self.bandwidth_guaranteed

    def __str__(self):
        return f'{self.name:<10} init={self.init_capacity:<5} cap={self.capacity.level:<5} diff={(self.init_capacity - self.capacity.level):<5}'

    def assign_prbs(self, prbs):
        delta = prbs - self.capacity.level
        if delta > 0:
            self.capacity.put(delta)
        elif delta < 0:
            self.capacity.get(-delta)

    def compute_reward(self):
        return self.capacity.level / self.bandwidth_max

class BasicUsagePattern:
    def __init__(self, low=1000, high=5000):
        self.low = low
        self.high = high

    def generate(self):
        return random.randint(self.low, self.high)