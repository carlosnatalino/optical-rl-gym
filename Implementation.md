# Optical RL-Gym

The guide is for programmers who are using the simulation to test out their algorithm. The following guide
will demonstrate on how to use the simulator in your algorithm.


1.  Create a file anywhere within the project.

2. Copy the following template of code for your file depending on which environment you are gonna use

## Template for PA-RMSA
```python
import gym
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from optical_rl_gym.utils import evaluate_heuristic


load = 250 #Adjusts the load of throughput
logging.getLogger('rmsacomplexenv').setLevel(logging.INFO)

seed = 20 #Seed for environment
episodes = 10 #Number of episodes per execution
episode_length = 1000 #Episode Length

#specifies the topology to use
with open(f'../examples/topologies/germany50_eon_gnpy_5-paths.h5', 'rb') as f:
           topology = pickle.load(f) 

#environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25, episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

#Your algorithm code will go in here
class Algorithm():

	def algorithm_code(self):
		pass


# Initial Metrics for Environment
init_env = gym.make('PowerAwareRMSA-v0', **env_args)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, "algorithm_here", n_eval_episodes=episodes)
print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned)
     / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted)
     / init_env.episode_services_processed)

```
## Template for RMCSA
```python
import gym
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from optical_rl_gym.utils import evaluate_heuristic

load = 100 #Adjusts the load of throughput
seed = 20 #Seed of environment
episodes = 10 #Number of episodes per execution
episode_length = 1000 #Episode Length
num_spatial_resources = 3 #Number of cores in environment

#specifies the topology to use
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)


#environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64, num_spatial_resources=num_spatial_resources)
env_sap = gym.make('RMCSA-v0', **env_args)

#Your algorithm code will go in here
class Algorithm():

	def algorithm_code(self):
		pass


mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, 'algorithm_here', n_eval_episodes=episodes)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

# Initial Metrics for Environment
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('Bit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
print('Request blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)

# Additional Metrics For Environment
print('Throughput:', env_sap.topology.graph['throughput'])
print('Compactness:', env_sap.topology.graph['compactness'])
print('Resource Utilization:', np.mean(env_sap.utilization))
for key, value in env_sap.core_utilization.items():
    print('Utilization per core ({}): {}'.format(key, np.mean(env_sap.core_utilization[key])))
```

## Template for RMSA/RWA
```python
import gym
from optical_rl_gym.utils import evaluate_heuristic
import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 50 #Adjusts the load of throughput
#change to environment using
logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20 #Seed of environment
episodes = 10 #Number of episodes per execution
episode_length = 1000 #Episode Length

# topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
   topology = pickle.load(f)

#environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
               episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

#Your algorithm code will go in here
class Algorithm():

	def algorithm_code(self):
		pass


init_env = gym.make('RMSA-v0', **env_args)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, "algorithm_here", n_eval_episodes=episodes)

print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')

# Initial Metrics for RMSA Environment
print('Bit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('Request blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print(init_env.topology.graph['throughput'])
exit(0)

# Initial Metrics for RwA Environment
env_rnd = gym.make('RWA-v0', **env_args)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, 'algorithm_here', n_eval_episodes=episodes)
print('Rnd:', mean_reward_rnd, std_reward_rnd)

```

## Template for QoSConstrainedRA
```python
import gym
import optical_rl_gym
from optical_rl_gym.utils import Path
from optical_rl_gym.utils import evaluate_heuristic

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 250 #Adjusts the load of throughput
logging.getLogger('qosconstrainedenv').setLevel(logging.INFO)

seed = 20 #Seed of environment
episodes = 10 #Number of episodes per execution
episode_length = 1000 #Episode Length

monitor_files = []
policies = []
k_paths = 5
num_service_classes=2
classes_arrival_probabilities=[.5, .5]
classes_reward=[10., 1.]
num_spectrum_resources = 16
load = 50

with open(f'../examples/topologies/nsfnet_chen_5-paths.h5', 'rb') as f:
   topology = pickle.load(f)

#environment arguments for the simulation
env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
               episode_length=episode_length, num_service_classes=num_service_classes,
               classes_arrival_probabilities=classes_arrival_probabilities,
               classes_reward=classes_reward, num_spectrum_resources=num_spectrum_resources, k_paths=k_paths)

#Your algorithm code will go in here
class Algorithm():

	def algorithm_code(self):
		pass


# Initial Metrics for QoSRA Environment
env_rnd = gym.make('QoSConstrainedRA-v0', **env_args)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, "algorithm_here", n_eval_episodes=episodes)
print('Rnd:', mean_reward_rnd, std_reward_rnd)

```


## Template for DeepRMSA
```python
import gym
from optical_rl_gym.utils import evaluate_heuristic

import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

episodes = 10 #Number of episodes per execution

topology_name = 'nsfnet_chen_eon'
k_paths = 5
with open(f'../examples/topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
   topology = pickle.load(f)

node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
                                      0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
                                      0.07607608, 0.12012012, 0.01901902, 0.16916917])

#environment arguments for the simulation
env_args = dict(topology=topology, seed=10,
               allow_rejection=False,
               mean_service_holding_time=7.5,
               mean_service_inter_arrival_time=1./12.,
               j=1,
               episode_length=50, node_request_probabilities=node_request_probabilities)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

#Your algorithm code will go in here
class Algorithm():

	def algorithm_code(self):
		pass


# Initial Metrics for DeepRMSA Environment
env_sp = gym.make('DeepRMSA-v0', **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, "algorithm_here", n_eval_episodes=episodes)
print('SP:'.ljust(5), f'{mean_reward_sp:.4f}  {std_reward_sp:>7.4f}')

```

* All that must be done is to implement your algorithm!



