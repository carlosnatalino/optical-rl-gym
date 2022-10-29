import logging
import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from optical_rl_gym.envs.rwa_env import (
    PathOnlyFirstFitAction,
    least_loaded_path_first_fit,
    shortest_available_path_first_fit,
    shortest_available_path_last_fit,
    shortest_path_first_fit,
)
from optical_rl_gym.utils import Path, evaluate_heuristic, random_policy

load = 450
logging.getLogger("rwaenv").setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 1000

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(
    os.path.join("..", "examples", "topologies", "nsfnet_chen_5-paths.h5"), "rb"
) as f:
    topology = pickle.load(f)

env_args = dict(
    topology=topology,
    seed=10,
    allow_rejection=True,
    load=load,
    mean_service_holding_time=25,
    episode_length=episode_length,
)

# creating an environment
env_rnd = gym.make("RWA-v0", **env_args)
# evaluating the environment that acts completely random both for path and wavelength
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(
    env_rnd, random_policy, n_eval_episodes=episodes
)
# env_rnd.render()  # uncomment to visualize a representation of the environment
print("\nRnd:", mean_reward_rnd, std_reward_rnd)
rnd_path_action_probability = np.sum(env_rnd.actions_output, axis=1) / np.sum(
    env_rnd.actions_output
)
rnd_wavelength_action_probability = np.sum(env_rnd.actions_output, axis=0) / np.sum(
    env_rnd.actions_output
)
print(
    "\tPath action probability:",
    np.sum(env_rnd.actions_output, axis=1) / np.sum(env_rnd.actions_output),
)
# print('Wavelength action probability:', np.sum(env_rnd.actions_output, axis=0) / np.sum(env_rnd.actions_output))

# creating an envionrment that only needs the path selection, then selects the first-fit wavelength automatically
env_rnd_ff = PathOnlyFirstFitAction(gym.make("RWA-v0", **env_args))
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(
    env_rnd_ff, random_policy, n_eval_episodes=episodes
)
# env_rnd.render()  # uncomment to visualize a representation of the environment
print("\nRnd-FF:", mean_reward_rnd, std_reward_rnd)
rnd_ff_path_action_probability = np.sum(env_rnd_ff.actions_output, axis=1) / np.sum(
    env_rnd_ff.actions_output
)
rnd_ff_wavelength_action_probability = np.sum(
    env_rnd_ff.actions_output, axis=0
) / np.sum(env_rnd_ff.actions_output)
print(
    "\tPath action probability:",
    np.sum(env_rnd.actions_output, axis=1) / np.sum(env_rnd.actions_output),
)
# print('Wavelength action probability:', np.sum(env_rnd.actions_output, axis=0) / np.sum(env_rnd.actions_output))

env_sp = gym.make("RWA-v0", **env_args)
mean_reward_sp, std_reward_sp = evaluate_heuristic(
    env_sp, shortest_path_first_fit, n_eval_episodes=episodes
)
sp_path_action_probability = np.sum(env_sp.actions_output, axis=1) / np.sum(
    env_sp.actions_output
)
sp_wavelength_action_probability = np.sum(env_sp.actions_output, axis=0) / np.sum(
    env_sp.actions_output
)
print("\nSP-FF:", mean_reward_sp, std_reward_sp)
print(
    "\tPath action probability:",
    np.sum(env_sp.actions_output, axis=1) / np.sum(env_sp.actions_output),
)
# print('Wavelength action probability:', np.sum(env_sp.actions_output, axis=0) / np.sum(env_sp.actions_output))

env_sap = gym.make("RWA-v0", **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(
    env_sap, shortest_available_path_first_fit, n_eval_episodes=episodes
)
sap_path_action_probability = np.sum(env_sap.actions_output, axis=1) / np.sum(
    env_sap.actions_output
)
sap_wavelength_action_probability = np.sum(env_sap.actions_output, axis=0) / np.sum(
    env_sap.actions_output
)
print("\nSAP-FF:", mean_reward_sap, std_reward_sap)
print(
    "\tPath action probability:",
    np.sum(env_sap.actions_output, axis=1) / np.sum(env_sap.actions_output),
)
# print('Wavelength action probability:', np.sum(env_sap.actions_output, axis=0) / np.sum(env_sap.actions_output))

env_sap_lf = gym.make("RWA-v0", **env_args)
mean_reward_sap, std_reward_sap = evaluate_heuristic(
    env_sap_lf, shortest_available_path_last_fit, n_eval_episodes=episodes
)
sap_lf_path_action_probability = np.sum(env_sap_lf.actions_output, axis=1) / np.sum(
    env_sap_lf.actions_output
)
sap_lf_wavelength_action_probability = np.sum(
    env_sap_lf.actions_output, axis=0
) / np.sum(env_sap_lf.actions_output)
print("\nSAP-LF:", mean_reward_sap, std_reward_sap)
print(
    "\tPath action probability:",
    np.sum(env_sap.actions_output, axis=1) / np.sum(env_sap.actions_output),
)
# print('Wavelength action probability:', np.sum(env_sap.actions_output, axis=0) / np.sum(env_sap.actions_output))

env_llp = gym.make("RWA-v0", **env_args)
mean_reward_llp, std_reward_llp = evaluate_heuristic(
    env_llp, least_loaded_path_first_fit, n_eval_episodes=episodes
)
llp_path_action_probability = np.sum(env_llp.actions_output, axis=1) / np.sum(
    env_llp.actions_output
)
llp_wavelength_action_probability = np.sum(env_llp.actions_output, axis=0) / np.sum(
    env_llp.actions_output
)
print("\nLLP:", mean_reward_llp, std_reward_llp)
print(
    "\tPath action probability:",
    np.sum(env_llp.actions_output, axis=1) / np.sum(env_llp.actions_output),
)
# print('Wavelength action probability:', np.sum(env_llp.actions_output, axis=0) / np.sum(env_llp.actions_output))

plt.figure()
plt.semilogy(rnd_path_action_probability, label="Rnd")
plt.semilogy(rnd_ff_path_action_probability, label="Rnd-FF")
plt.semilogy(sp_path_action_probability, label="SP-FF")
plt.semilogy(sap_path_action_probability, label="SAP-FF")
plt.semilogy(sap_lf_path_action_probability, label="SAP-LF")
plt.semilogy(llp_path_action_probability, label="LLP-FF")
plt.xlabel("Path index")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig(f"rwa_path_action_probability_{load}.svg")
plt.close()

plt.figure()
plt.semilogy(rnd_wavelength_action_probability, label="Rnd")
plt.semilogy(rnd_ff_wavelength_action_probability, label="Rnd-FF")
plt.semilogy(sp_wavelength_action_probability, label="SP-FF")
plt.semilogy(sap_wavelength_action_probability, label="SAP-FF")
plt.semilogy(sap_lf_wavelength_action_probability, label="SAP-LF")
plt.semilogy(llp_wavelength_action_probability, label="LLP-FF")
plt.xlabel("Wavelength index")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig(f"rwa_wavelength_action_probability_{load}.svg")
plt.close()
