import gym
from optical_rl_gym.envs.rmsa_env import shortest_path_first_fit, shortest_available_path_first_fit, \
    least_loaded_path_first_fit, SimpleMatrixObservation
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 50
logging.getLogger('rmsaenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 1000

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(f'../examples/topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64)

print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

init_env = gym.make('RMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, random_policy, n_eval_episodes=episodes)
print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('Bit rate blocking:', (init_env.bit_rate_requested_since_reset - init_env.bit_rate_provisioned_since_reset) / init_env.bit_rate_requested_since_reset)
print('Request blocking:', (init_env.services_processed_since_reset - init_env.services_accepted_since_reset) / init_env.services_processed_since_reset)

env_sp = gym.make('RMSA-v0', **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(env_sp, shortest_path_first_fit, n_eval_episodes=episodes)
print('SP-FF:'.ljust(8), f'{mean_reward_sp:.4f}  {std_reward_sp:<7.4f}')
print('Bit rate blocking:', (env_sp.bit_rate_requested_since_reset - env_sp.bit_rate_provisioned_since_reset) / env_sp.bit_rate_requested_since_reset)
print('Request blocking:', (env_sp.services_processed_since_reset - env_sp.services_accepted_since_reset) / env_sp.services_processed_since_reset)

env_sap = gym.make('RMSA-v0', **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes)
print('SAP-FF:'.ljust(8), f'{mean_reward_sap:.4f}  {std_reward_sap:.4f}')
print('Bit rate blocking:', (env_sap.bit_rate_requested_since_reset - env_sap.bit_rate_provisioned_since_reset) / env_sap.bit_rate_requested_since_reset)
print('Request blocking:', (env_sap.services_processed_since_reset - env_sap.services_accepted_since_reset) / env_sap.services_processed_since_reset)

env_llp = gym.make('RMSA-v0', **env_args)
mean_reward_llp, std_reward_llp = evaluate_heuristic(env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes)
print('LLP-FF:'.ljust(8), f'{mean_reward_llp:.4f}  {std_reward_llp:.4f}')
print('Bit rate blocking:', (env_llp.bit_rate_requested_since_reset - env_llp.bit_rate_provisioned_since_reset) / env_llp.bit_rate_requested_since_reset)
print('Request blocking:', (env_llp.services_processed_since_reset - env_llp.services_accepted_since_reset) / env_llp.services_processed_since_reset)
