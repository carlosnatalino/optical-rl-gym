import gym
from optical_rl_gym.envs.deeprmsa_env import shortest_path_first_fit, shortest_available_path_first_fit
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

logging.getLogger('rmsaenv').setLevel(logging.INFO)

episodes = 10

monitor_files = []
policies = []

topology_name = 'nsfnet_chen_eon'
k_paths = 5
with open(f'../examples/topologies/{topology_name}_{k_paths}-paths.h5', 'rb') as f:
    topology = pickle.load(f)

node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505,
                                       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
                                       0.07607608, 0.12012012, 0.01901902, 0.16916917])
env_args = dict(topology=topology, seed=10, 
                allow_rejection=False,
                mean_service_holding_time=7.5,
                mean_service_inter_arrival_time=1./12.,
                j=1, 
                episode_length=50, node_request_probabilities=node_request_probabilities)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

init_env = gym.make('DeepRMSA-v0', **env_args)
env_rnd = init_env
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
print('Rnd:'.ljust(5), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')

env_sp = gym.make('DeepRMSA-v0', **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
print('SP:'.ljust(5), f'{mean_reward_sp:.4f}  {std_reward_sp:>7.4f}')

env_sap = gym.make('DeepRMSA-v0', **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
print('SAP:'.ljust(5), f'{mean_reward_sap:.4f}  {std_reward_sap:>7.4f}')

# env_llp = gym.make('DeepRMSA-v0', **env_args)
# mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
# print('LLP:'.ljust(5), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
