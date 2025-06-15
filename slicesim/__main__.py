# __main__.py (DQN-enabled version with fast debug mode)
import sys
import yaml
import simpy
from slicesim.BaseStation import BaseStation
from slicesim.Slice import Slice, BasicUsagePattern
from slicesim.Client import Client
from slicesim.distributor_dqn import DQNDistributor
from slicesim.Stats import Stats
import matplotlib.pyplot as plt

# Load config
config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Create SimPy environment
env = simpy.Environment()

# Create base stations
base_stations = []
for bs_cfg in config['base_stations']:
    slices = {}
    for i, (slice_name, ratio) in enumerate(bs_cfg['slice_ratios'].items()):
        s_cfg = next(s for s in config['slices'] if s['name'] == slice_name)
        slices[i] = Slice(
            name=slice_name,
            env=env,
            ratio=ratio,
            connected_users=0,
            user_share=s_cfg['client_weight'],
            delay_tolerance=s_cfg['delay_tolerance'],
            qos_class=s_cfg['qos_class'],
            bandwidth_guaranteed=s_cfg['bandwidth_guaranteed'],
            bandwidth_max=s_cfg['bandwidth_max'],
            init_capacity=bs_cfg['bandwidth'] * ratio,
            usage_pattern=BasicUsagePattern()
        )
    assert isinstance(slices, dict) or isinstance(slices, list)
    for s in (slices.values() if isinstance(slices, dict) else slices):
        assert hasattr(s, "capacity")
    bs = BaseStation(
        id=bs_cfg.get('id', 'BS'),
        pk=bs_cfg.get('pk', 0),
        coverage=bs_cfg['coverage'],
        capacity_bandwidth=bs_cfg['bandwidth'],
        slices=slices
    )
    base_stations.append(bs)

# Create clients (without starting their processes yet)
clients = []
for i in range(config['settings']['num_clients']):
    client = Client(
        env=env,
        pk=i,
        x=0, y=0,  # or random if you want
        mobility_pattern=None,  # optional to add later
        usage_freq=0.2,
        subscribed_slice_index=i % 3,
        base_station=base_stations[i % len(base_stations)],
        stat_collector=None  # will be set next
    )
    clients.append(client)

# Create stat collector and start collecting before clients run
stat_collector = Stats(env, base_stations, clients, area=(1000, 1000))
env.process(stat_collector.collect())

# Patch stat collector into clients and start them
for client in clients:
    client.stat_collector = stat_collector
    client.action = env.process(client.iter())  # ✅ start now, after stats initialized

# Create DQN distributor
distributor = DQNDistributor(base_station=base_stations[0], num_slices=3)

# Run simulation loop (FAST MODE)
for t in range(config['settings']['simulation_time']):
    distributor.allocate()  # DQN agent takes action
    if t % 10 == 0:
        distributor.agent.replay(batch_size=32)  # train every 10 steps
    env.step()  # Advance simulation

# Print final stats
stat_collector.print_summary()

# Plot DQN reward trend
plt.plot(distributor.rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("DQN Agent Reward Over Time")
plt.grid(True)
plt.savefig("output.png", dpi=150)  # <-- Save BEFORE showing
plt.show()