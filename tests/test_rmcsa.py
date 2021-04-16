import os
import gym
from optical_rl_gym.utils import evaluate_heuristic, random_policy
from optical_rl_gym.envs.rmcsa_env import shortest_available_path_first_core_first_fit, SimpleMatrixObservation

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 250
logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 1000
num_spatial_resources = 7
worst_xt = -84.7
# Note: The simulator can automatically handle worst_crosstalk values for 7, 12, and 19 cores.
# Besides that, a worst_xt value in dB must be specified and put in the enviroment.

modulation_formats = np.array([
    { # BPSK
        'mod_factor':1, # Bits per symbol of each format
        'snr_min':4.2,  # minimum signal-to-noise for each modulation format
        'inband_xt':-14 # Crosstalk in dB
    }, # BPSK
    { # QPSK
        'mod_factor':2,
        'snr_min':7.2,
        'inband_xt':-17
    }, # QPSK
    { # 16QAM
        'mod_factor':4,
        'snr_min':13.9,
        'inband_xt':-23
    }, # 16QAM
    { # 64QAM
        'mod_factor':6,
        'snr_min':19.8,
        'inband_xt':-29
    }, # 64QAM
])


monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(os.path.join('..', 'examples', 'topologies', 'nsfnet_chen_eon_5-paths.h5'), 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64, num_spatial_resources=num_spatial_resources,
                modulation_formats=modulation_formats, worst_xt=worst_xt
                )

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

init_env = gym.make('RMCSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('\tBit rate blocking:', (init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) / init_env.episode_bit_rate_requested)
print('\tRequest blocking:', (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print('Throughput:', init_env.topology.graph['throughput'])

# env_sap = gym.make('RMCSA-v0', **env_args)
# mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_core_first_fit, n_eval_episodes=episodes)

# print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))
#
# # Initial Metrics for Environment
# print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
# print('\tBit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
# print('\tRequest blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)
#
# # Additional Metrics For Environment
# print('\tThroughput:', env_sap.topology.graph['throughput'])
# print('\tCompactness:', env_sap.topology.graph['compactness'])
# print('\tResource Utilization:', np.mean(env_sap.utilization))
# for key, value in env_sap.core_utilization.items():
#     print('\t\tUtilization per core ({}): {}'.format(key, np.mean(env_sap.core_utilization[key])))

"""
#Specific - modify
env_sp = gym.make('RMCSA-v0', **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
print('SP-FF:'.ljust(8), f'{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}')
print('Bit rate blocking:', (env_sp.episode_bit_rate_requested - env_sp.episode_bit_rate_provisioned) / env_sp.episode_bit_rate_requested)
print('Request blocking:', (env_sp.episode_services_processed - env_sp.episode_services_accepted) / env_sp.episode_services_processed)

env_sap = gym.make('RMCSA-v0', **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('Bit rate blocking:', (env_sap.episode_bit_rate_requested - env_sap.episode_bit_rate_provisioned) / env_sap.episode_bit_rate_requested)
print('Request blocking:', (env_sap.episode_services_processed - env_sap.episode_services_accepted) / env_sap.episode_services_processed)

env_llp = gym.make('RMCSA-v0', **env_args)
mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
print('LLP-FF:'.ljust(8), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
print('Bit rate blocking:', (env_llp.episode_bit_rate_requested - env_llp.episode_bit_rate_provisioned) / env_llp.episode_bit_rate_requested)
print('Request blocking:', (env_llp.episode_services_processed - env_llp.episode_services_accepted) / env_llp.episode_services_processed)
"""