import gym
import copy
import math
import heapq
import logging
import functools
import numpy as np

from optical_rl_gym.utils import Service, Path
from .optical_network_env import OpticalNetworkEnv


class RMSAEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'service_blocking_rate_since_reset',
                    'bit_rate_blocking_rate', 'bit_rate_blocking_rate_since_reset']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=100,
                 node_request_probabilities=None,
                 bit_rate_lower_bound=25,
                 bit_rate_higher_bound=100,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False,
                 reset=True):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed, allow_rejection=allow_rejection,
                         k_paths=k_paths)
        assert 'modulations' in self.topology.graph
        # specific attributes for elastic optical networks
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.bit_rate_requested_since_reset = 0
        self.bit_rate_provisioned_since_reset = 0

        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros(self.k_paths * self.num_spectrum_resources + self.reject_action, dtype=int)
        self.actions_output_since_reset = np.zeros(self.k_paths * self.num_spectrum_resources + self.reject_action, dtype=int)
        self.actions_taken = np.zeros(self.k_paths * self.num_spectrum_resources + self.reject_action, dtype=int)
        self.actions_taken_since_reset = np.zeros(self.k_paths * self.num_spectrum_resources + self.reject_action, dtype=int)
        self.action_space = gym.spaces.Discrete(self.k_paths * self.num_spectrum_resources + self.reject_action)
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action: int):
        self.actions_output[action] += 1
        if action < self.k_paths * self.num_spectrum_resources:  # action is for assigning a path
            path, initial_slot = self._get_path_slot_id(action)
            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])
            self.logger.debug('{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id, action, path, initial_slot, slots))
            if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                 initial_slot, slots):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                     initial_slot, slots)
                self.service.accepted = True
                self.actions_taken[action] += 1
                self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths * self.num_spectrum_resources] += 1

        self.services_processed += 1
        self.services_processed_since_reset += 1
        self.bit_rate_requested += self.service.bit_rate
        self.bit_rate_requested_since_reset += self.service.bit_rate

        self.topology.graph['services'].append(self.service)

        reward = self.reward()
        info = {
                   'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
                   'service_blocking_rate_since_reset': (self.services_processed_since_reset - self.services_accepted_since_reset) / self.services_processed_since_reset,
                   'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
                   'bit_rate_blocking_rate_since_reset': (self.bit_rate_requested_since_reset - self.bit_rate_provisioned_since_reset) / self.bit_rate_requested_since_reset
               }

        self._new_service = False
        self._next_service()
        return self.observation(), reward, self.services_processed_since_reset == self.episode_length, info

    def reset(self, only_counters=True):
        self.bit_rate_requested_since_reset = 0
        self.bit_rate_provisioned_since_reset = 0
        self.services_processed_since_reset = 0
        self.services_accepted_since_reset = 0
        self.actions_output_since_reset = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.actions_taken_since_reset = np.zeros(self.k_paths + self.reject_action, dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones((self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int)

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    def _provision_path(self, path: Path, initial_slot, number_slots):
        # usage
        if not self.is_path_free(path, initial_slot, number_slots):
            raise ValueError("Path {} has not enough capacity on slots {}-{}".format(path.node_list, path, initial_slot,
                                                                                     initial_slot + number_slots))

        self.logger.debug('{} assigning path {} on initial slot {} for {} slots'.format(self.service.service_id, path.node_list, initial_slot, number_slots))
        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_slots'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                                                                        initial_slot:initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                                                    initial_slot:initial_slot + number_slots] = self.service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service)
        self._update_network_stats()
        self.service.route = path
        self.service.initial_slot = initial_slot
        self.service.number_slots = number_slots

        self.services_accepted += 1
        self.services_accepted_since_reset += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.bit_rate_provisioned_since_reset += self.service.bit_rate

    def _release_path(self, service: Service):
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_slots'][
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        self.topology.graph['running_services'].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            cur_throughtput = 0.
            last_throughput = self.topology.graph['throughput']
            for service in self.topology.graph["running_services"]:
                cur_throughtput += service.bit_rate
            utilization = ((last_throughput * last_update) + (cur_throughtput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = utilization

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :])) / self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_fragmentation = self.topology[node1][node2]['fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_fragmentation = 0.
            cur_compactness = 0.
            if np.sum(slot_allocation) > 0:
                blocks = np.split(slot_allocation, np.where(np.diff(slot_allocation) != 0)[0] + 1)
                max_empty = 0
                for block in blocks:
                    if np.all(block == 1):
                        max_empty = max(max_empty, len(block))
                cur_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                lambdas = np.where(slot_allocation == 0)
                if len(lambdas) > 1:
                    lambda_min = np.min(lambdas)
                    lambda_max = np.max(lambdas)
                    # alloc = slot_allocation[lambda_min:lambda_max]
                    blocks = np.split(slot_allocation[lambda_min:lambda_max],
                                      np.where(np.diff(slot_allocation[lambda_min:lambda_max]) != 0)[0] + 1)
                    k = 0
                    for block in blocks:
                        if np.all(block == 1):
                            k += 1
                    # number of blocks of free slots between first and last slot used
                    if k > 0:
                        cur_compactness = ((lambda_max - lambda_min + 1) / len(lambdas)) * (1 / k)
                    else:
                        cur_compactness = 1.
                else:
                    cur_compactness = 1.

            fragmentation = ((last_fragmentation * last_update) + (cur_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['fragmentation'] = fragmentation

            link_compactness = ((last_compactness * last_update) + (cur_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6476152

            # TODO: implement fragmentation
        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        bit_rate = self.rng.randint(self.bit_rate_lower_bound, self.bit_rate_higher_bound)

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

        self.service = Service(self.services_processed_since_reset, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)
        self._new_service = True

    def _get_path_slot_id(self, action: int) -> (int, int):
        path = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return path, initial_slot

    def get_number_slots(self, path: Path) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        return math.ceil(self.service.bit_rate / path.best_modulation['capacity']) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph['available_slots'][
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      initial_slot:initial_slot + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, path: Path):
        available_slots = functools.reduce(np.multiply,
            self.topology.graph["available_slots"][[self.topology[path.node_list[i]][path.node_list[i + 1]]['id']
                                                    for i in range(len(path.node_list) - 1)], :])
        return available_slots


def shortest_path_first_fit(env: RMSAEnv) -> int:
    num_slots = env.get_number_slots(env.k_shortest_paths[env.service.source, env.service.destination][0])
    for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
        if env.is_path_free(env.k_shortest_paths[env.service.source, env.service.destination][0], initial_slot, num_slots):
            return initial_slot
    return env.topology.graph['k_paths'] * env.topology.graph['num_spectrum_resources']


def shortest_available_path_first_fit(env: RMSAEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                if idp != 0:
                    print('test')
                return idp * env.topology.graph['num_spectrum_resources'] + initial_slot
    return env.topology.graph['k_paths'] * env.topology.graph['num_spectrum_resources']


def least_loaded_path_first_fit(env: RMSAEnv) -> int:
    max_free_slots = 0
    action = env.topology.graph['k_paths'] * env.topology.graph['num_spectrum_resources']
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
            if env.is_path_free(path, initial_slot, num_slots):
                if idp != 0:
                    print('test')
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = idp * env.topology.graph['num_spectrum_resources'] + initial_slot
                    max_free_slots = free_slots
                break # breaks the loop for the initial slot
    return action


class SimpleMatrixObservation(gym.ObservationWrapper):

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        shape = self.env.topology.number_of_nodes() * 2 \
                + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(self.env.service.source_id, self.env.service.destination_id)
        max_node = max(self.env.service.source_id, self.env.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate((source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
                            .reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):

    def __init__(self, env: RMSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths - 1:
            num_slots = self.env.get_number_slots(self.env.k_shortest_paths[self.env.service.source,
                                                                            self.env.service.destination][action])
            for initial_slot in range(0, self.env.topology.graph['num_spectrum_resources'] - num_slots):
                if self.env.is_path_free(self.env.k_shortest_paths[self.env.service.source,
                                                                   self.env.service.destination][action],
                                        initial_slot, num_slots):
                    return action * self.env.topology.graph['num_spectrum_resources'] + initial_slot
                else:
                    return self.env.topology.graph['k_paths'] * self.env.topology.graph['num_spectrum_resources']
        else:
            return self.env.topology.graph['k_paths'] * self.env.topology.graph['num_spectrum_resources']

    def step(self, action):
        return self.env.step(self.action(action))
