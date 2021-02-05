# Optical RL-Gym

Use the following templates as a reference for your heuristic. You can determine parameters like load, number of episodes, length of each episode, and random seed. The envrironments have built-in metrics that you can use to evaluate the heuristic.

## Template for RMCSA
```python
import gym
from optical_rl_gym.utils import evaluate_heuristic
import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 50  # Traffic load, measured in Erlangs
seed = 20  # Seed of environment
episodes = 10  # Number of episodes per execution
episode_length = 1000  # Episode Length
num_spatial_resources = 3  # Number of cores in environment

# Specifies the topology to use
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)


# Environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64, num_spatial_resources=num_spatial_resources)
init_env = gym.make('RMCSA-v0', **env_args)

# Algorithm Heuristic
class Algorithm():

	def algorithm_code(self):
		pass

# Initial Metrics for Environment
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)
print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))
print('Heuristic:'.ljust(8), f'{mean_reward:.4f}  {std_reward:.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)

# Additional Metrics For Environment
print('Throughput:', init_env.topology.graph['throughput'])
print('Compactness:', init_env.topology.graph['compactness'])
print('Resource Utilization:', np.mean(init_env.utilization))
for key, value in init_env.core_utilization.items():  # Displays individual core usage
    print('Utilization per core ({}): {}'.format(key, np.mean(init_env.core_utilization[key]))) 
```

## Template for RMSA/RWA/PA-RMSA
```python
import gym
from optical_rl_gym.utils import evaluate_heuristic
import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 50  # Traffic load, measured in Erlangs
seed = 20  # Seed of environment
episodes = 10  # Number of episodes per execution
episode_length = 1000  # Episode Length

with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
   topology = pickle.load(f)

# Environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
               episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

# Algorithm Heuristic
class Algorithm():

	def algorithm_code(self):
		pass

# Initial Metrics for RMSA Environment - Remove as necessarry
init_env = gym.make('RMSA-v0', **env_args)
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)

# Initial Metrics for RWA Environment - Remove as necessarry
init_env = gym.make('RWA-v0', **env_args)
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)

# Initial Metrics for PA-RMSA Environment - Remove as necessarry
init_env = gym.make('PowerAwareRMSA-v0', **env_args)
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)

print('Heuristic:'.ljust(8), f'{mean_reward:.4f}  {std_reward:>7.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print(init_env.topology.graph['throughput'])
exit(0)

```

## Template for QoSConstrainedRA
```python
import gym
import optical_rl_gym
from optical_rl_gym.utils import Route
from optical_rl_gym.utils import evaluate_heuristic

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 50  # Traffic load, measured in Erlangs
seed = 20  # Seed of environment
episodes = 10  # Number of episodes per execution
episode_length = 1000  # Episode Length

monitor_files = []
policies = []
k_paths = 5
num_service_classes=2
classes_arrival_probabilities=[.5, .5]
classes_reward=[10., 1.]
num_spectrum_resources = 16

with open(f'../examples/topologies/nsfnet_chen_5-paths.h5', 'rb') as f:
   topology = pickle.load(f)

# Environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
               episode_length=episode_length, num_service_classes=num_service_classes,
               classes_arrival_probabilities=classes_arrival_probabilities,
               classes_reward=classes_reward, num_spectrum_resources=num_spectrum_resources, k_paths=k_paths)

# Algorithm Heuristic
class Algorithm():

	def algorithm_code(self):
		pass


# Initial Metrics for QoSRA Environment
init_env = gym.make('QoSConstrainedRA-v0', **env_args)
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)
print('Heuristic:', mean_reward, std_reward)

```


## Template for DeepRMSA
```python
import gym
from optical_rl_gym.utils import evaluate_heuristic

import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

episodes = 10  # Number of episodes per execution

topology_name = 'nsfnet_chen_eon'
k_paths = 5
with open(f'../examples/topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
   topology = pickle.load(f)

node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
                                      0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
                                      0.07607608, 0.12012012, 0.01901902, 0.16916917])

# Environment arguments for the simulation
env_args = dict(topology=topology, seed=10,
               allow_rejection=False,
               mean_service_holding_time=7.5,
               mean_service_inter_arrival_time=1./12.,
               j=1,
               episode_length=50, node_request_probabilities=node_request_probabilities)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

# Algorithm Heuristic
class Algorithm():

	def algorithm_code(self):
		pass


# Initial Metrics for DeepRMSA Environment
init_env = gym.make('DeepRMSA-v0', **env_args)
mean_reward, std_reward = evaluate_heuristic(init_env, $ALGORITHM_METHOD, n_eval_episodes=episodes)
print('FF:'.ljust(5), f'{mean_reward:.4f}  {std_reward:>7.4f}')

```

