from typing import Tuple

import gym
import numpy as np

from .rmsa_env import RMSAEnv


class DeepRMSAEnv(RMSAEnv):
    def __init__(
        self,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        seed=None,
        allow_rejection=False,
    ):
        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=False,
        )

        self.j = j
        shape = (
            1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.j + self.reject_action
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_episode_counters=False)

    def step(self, action: int):
        if action < self.k_paths * self.j:  # action is for assigning a route
            route, block = self._get_route_block_id(action)

            initial_indices, lengths = self.get_available_blocks(route)
            if block < len(initial_indices):
                return super().step([route, initial_indices[block]])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources])
        else:
            return super().step([self.k_paths, self.num_spectrum_resources])

    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(
            self.current_service.source_id, self.current_service.destination_id
        )
        max_node = max(
            self.current_service.source_id, self.current_service.destination_id
        )
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths, 2 * self.j + 3), fill_value=-1.0)
        for idp, route in enumerate(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ]
        ):
            available_slots = self.get_available_slots(route)
            num_slots = self.get_number_slots(route)
            initial_indices, lengths = self.get_available_blocks(idp)

            for idb, (initial_index, length) in enumerate(
                zip(initial_indices, lengths)
            ):
                # initial slot index
                spectrum_obs[idp, idb * 2 + 0] = (
                    2
                    * (initial_index - 0.5 * self.num_spectrum_resources)
                    / self.num_spectrum_resources
                )

                # number of contiguous FS available
                spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp, self.j * 2] = (
                num_slots - 5.5
            ) / 3.5  # number of FSs necessary

            idx, values, lengths = DeepRMSAEnv.rle(available_slots)

            spectrum_obs[idp, self.j * 2 + 1] = (
                2
                * (np.sum(available_slots) - 0.5 * self.num_spectrum_resources)
                / self.num_spectrum_resources
            )  # total number available FSs
            av_indices = np.argwhere(values == 1)  # getting indices which have value 1
            if av_indices.shape[0] > 0:
                spectrum_obs[idp, self.j * 2 + 2] = (
                    np.mean(lengths[av_indices]) - 4
                ) / 4  # avg. number of FS blocks available
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.current_service.bit_rate / 100

        return np.concatenate(
            (
                bit_rate_obs,
                source_destination_tau.reshape(
                    (1, np.prod(source_destination_tau.shape))
                ),
                spectrum_obs.reshape((1, np.prod(spectrum_obs.shape))),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)

    def reward(self):
        return 1 if self.current_service.accepted else -1

    def reset(self, only_episode_counters=True):
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int]:
        route = action // self.j
        block = action % self.j
        return route, block


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, _ = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, _ in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        initial_indices, _ = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j
