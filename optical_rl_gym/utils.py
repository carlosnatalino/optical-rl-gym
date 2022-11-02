import typing
from dataclasses import dataclass, field
from itertools import islice
from typing import Optional, Sequence, Tuple, Union

import typing
import networkx as nx
import numpy as np

if typing.TYPE_CHECKING:
    from optical_rl_gym.envs.optical_network_env import OpticalNetworkEnv


@dataclass
class Modulation:
    name: str
    # maximum length in km
    maximum_length: Union[int, float]
    # number of bits per Hz per sec.
    spectral_efficiency: int
    # minimum OSNR that allows it to work
    minimum_osnr: Optional[float] = field(default=None)
    # maximum in-band cross-talk
    inband_xt: Optional[float] = field(default=None)


@dataclass
class Path:
    path_id: int
    node_list: Tuple[str]
    hops: int
    length: Union[int, float]
    best_modulation: Optional[Modulation] = field(default=None)
    current_modulation: Optional[Modulation] = field(default=None)


@dataclass(repr=False)
class Service:
    service_id: int
    source: str
    source_id: int
    destination: Optional[str] = field(default=None)
    destination_id: Optional[str] = field(default=None)
    arrival_time: Optional[float] = field(default=None)
    holding_time: Optional[float] = field(default=None)
    bit_rate: Optional[float] = field(default=None)
    path: Optional[Path] = field(default=None)
    best_modulation: Optional[Modulation] = field(default=None)
    service_class: Optional[int] = field(default=None)
    number_slots: Optional[int] = field(default=None)
    core: Optional[int] = field(default=None)
    launch_power: Optional[float] = field(default=None)
    accepted: bool = field(default=False)

    def __str__(self):
        msg = "{"
        msg += "" if self.bit_rate is None else f"br: {self.bit_rate}, "
        msg += "" if self.service_class is None else f"cl: {self.service_class}, "
        return f"Serv. {self.service_id} ({self.source} -> {self.destination})" + msg


def start_environment(env: "OpticalNetworkEnv", steps: int) -> "OpticalNetworkEnv":
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight="length"):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def get_best_modulation_format(
    length: float, modulations: Sequence[Modulation]
) -> Modulation:
    # sorts modulation from the most to the least spectrally efficient
    sorted_modulations = sorted(
        modulations, key=lambda x: x.spectral_efficiency, reverse=True
    )
    for i in range(len(modulations)):
        if length <= sorted_modulations[i].maximum_length:
            return sorted_modulations[i]
    raise ValueError(
        "It was not possible to find a suitable MF for a path with {} km".format(length)
    )


def random_policy(env):
    return env.action_space.sample()


def evaluate_heuristic(
    env: "OpticalNetworkEnv",
    heuristic,
    n_eval_episodes=10,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
):
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        _ = env.reset()
        done, _ = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = heuristic(env)
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert (
            mean_reward > reward_threshold
        ), "Mean reward below threshold: " "{:.2f} < {:.2f}".format(
            mean_reward, reward_threshold
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
