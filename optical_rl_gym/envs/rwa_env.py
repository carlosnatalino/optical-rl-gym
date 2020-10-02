import gym

import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt

from optical_rl_gym.utils import Service
from .optical_network_env import OpticalNetworkEnv


class RWAEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=80,
                 num_service_classes=1,
                 node_request_probabilities=None,
                 allow_rejection=True,
                 k_paths=5,
                 seed=None, reset=True):
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths)

        self.num_service_classes = num_service_classes

        self.reject_action = 1 if allow_rejection else 0

        self.actions_output = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_actions_output = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.actions_taken = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_actions_taken = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.action_space = gym.spaces.Discrete(self.k_paths + self.reject_action)
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rwaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action: int):
        self.actions_output[action] += 1
        if (self.service.service_class == 0 and action == 0) \
                or (self.service.service_class != 0 and action < len(self.k_shortest_paths[self.service.source, self.service.destination])):
            if self._is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][action], self.service.number_slots):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][action])
                self.service.accepted = True
                self.services_accepted += 1
                self.episode_services_accepted += 1

                self.actions_taken[action] += 1
                self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths] += 1

        self.services_processed += 1
        self.episode_services_processed += 1

        self.topology.graph['services'].append(self.service)

        reward = self.reward()
        info = {
            'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
            'episode_service_blocking_rate': (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
        }

        self._new_service = False
        self._next_service()

        return self.observation(), reward, self.episode_services_processed == self.episode_length, info

    def reset(self, only_counters=True):
        self.episode_actions_output = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_actions_output = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_actions_taken = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_actions_taken = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        if only_counters:
            return self.observation()

        super().reset()

        # saving statistics
        self.actions_output = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.actions_taken = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human', close=False):
        fig = plt.figure(figsize=(18, 12))

        plt.subplot(1, 4, 1)
        slots_representation = np.zeros((self.topology.number_of_edges(), self.num_spectrum_resources))
        for idl, link in enumerate(self.topology.edges()):
            slots_representation[idl, 0:self.topology.graph['available_spectrum'][self.topology[link[0]][link[1]]['index']]] = 1.
        plt.pcolor(slots_representation, cmap=plt.cm.Greys, edgecolors='gray', linewidth=.01)
        plt.xlabel('slot')
        plt.ylabel('edge')

        plt.subplot(1, 4, 2)
        source_destination_representation = np.zeros((self.topology.number_of_nodes(), 2))
        source_destination_representation[self.service.source_id, 0] = 1
        source_destination_representation[self.service.destination_id, 1] = 1
        plt.pcolor(source_destination_representation, cmap=plt.cm.Greys, edgecolors='none', linewidth=.01)
        # plt.xlabel('edge')
        plt.ylabel('node')
        plt.xticks([0.5, 1.5], ['src', 'dst'], rotation=90)

        plt.subplot(1, 4, 3)
        class_representation = np.zeros((self.num_service_classes, 1))
        class_representation[self.service.service_class, 0] = 1
        plt.pcolor(class_representation, cmap=plt.cm.Greys, edgecolors='none', linewidth=.01)
        # plt.xlabel('edge')
        plt.ylabel('class')
        # plt.xticks([0.5, 1.5], ['src', 'dst'], rotation=90)

        plt.subplot(1, 4, 4)
        paths = np.zeros((self.k_paths, self.topology.number_of_edges()))
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            for i in range(len(path.node_list) - 1):
                paths[idp, self.topology[path.node_list[i]][path.node_list[i + 1]]['index']] = 1
        plt.pcolor(paths, cmap=plt.cm.Greys, edgecolors='none', linewidth=.01)
        plt.xlabel('path')
        plt.ylabel('node')

        plt.tight_layout()
        plt.show()

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the look

        self.service = Service(self.episode_services_processed, src, src_id, destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, number_slots=1)
        self._new_service = True

    def observation(self):
        return {'topology': self.topology,
                'service': self.service}

    def _provision_path(self, path):
        # usage
        if not self._is_path_free(path, self.service.number_slots):
            raise ValueError("Path {} has not enough capacity".format(path.node_list, path))

        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_spectrum'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index']] -= self.service.number_slots
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service.service_id)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service.service_id)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service.service_id)
        self._update_network_stats()
        self.service.route = path

    def _release_path(self, service):
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_spectrum'][self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index']] += service.number_slots
            try:
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service.service_id)
            except:
                self.logger.warning('error')
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        try:
            self.topology.graph['running_services'].remove(service.service_id)
        except:
            self.logger.warning('error')

    def _update_network_stats(self):
        """
        Implement here any network-wide statistics
        :return:
        """
        pass
        # last_update = self.topology.graph['last_update']
        # time_diff = self.current_time - last_update
        # if self.current_time > 0:
        #     for service in self.topology.graph["running_services"]:
        #         cur_throughtput += service.bit_rate
        #     utilization = ((last_throughput * last_update) + (cur_throughtput * time_diff)) / self.current_time
        #     self.topology.graph['throughput'] = utilization

    def _update_link_stats(self, node1, node2):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - self.topology.graph['available_spectrum'][self.topology[node1][node2]['index']]) / self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

        self.topology[node1][node2]['last_update'] = self.current_time

    def _is_path_free(self, path, number_slots):
        return is_path_free(self.topology, path, number_slots)


def is_path_free(topology, path, number_slots):
    if number_slots > topology.graph['num_spectrum_resources']:
        return False
    for i in range(len(path.node_list) - 1):
        if topology.graph['available_spectrum'][topology[path.node_list[i]][path.node_list[i + 1]]['index']] < number_slots:
            return False
    return True


def get_path_capacity(topology, path):
    capacity = np.finfo(0.0).max
    for i in range(len(path.node_list) - 1):
        capacity = min(capacity, topology.graph['available_spectrum'][topology[path.node_list[i]][path.node_list[i + 1]]['index']])
    return capacity


def shortest_path(env: RWAEnv) -> int:
    if is_path_free(env.topology, env.topology.graph['ksp'][env.service.source, env.service.destination][0], env.service.number_slots):
        return 0
    return env.topology.graph['k_paths']


def shortest_available_path(env: RWAEnv) -> int:
    if env.service.service_class == 0:
        # high-priority services only accept the shortest path
        return 0
    best_hops = np.finfo(0.0).max
    decision = env.topology.graph['k_paths']
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        if path.hops < best_hops and is_path_free(env.topology, path, env.service.number_slots):
            best_hops = path.hops
            decision = idp
    return decision


def least_loaded_path(env: RWAEnv) -> int:
    if env.service.service_class == 0:
        return 0
    best_load = np.finfo(0.0).min
    decision = env.topology.graph['k_paths']
    for idp, path in enumerate(env.topology.graph['ksp'][env.service.source, env.service.destination]):
        cap = get_path_capacity(env.topology, path)
        if cap > best_load:
            best_load = cap
            decision = idp
    return decision


class MatrixObservationWithPaths(gym.ObservationWrapper):

    def __init__(self, env: RWAEnv):
        super().__init__(env)
        shape = self.env.topology.number_of_edges() * self.env.num_spectrum_resources * (self.env.k_paths + 1) + 1
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))

    def observation(self, observation):
        observation = np.zeros((self.env.topology.number_of_edges(), self.env.num_spectrum_resources * (self.env.k_paths + 1)))
        for edge_indices in self.env.topology.edges():
            edge_id = self.env.topology.edges[edge_indices]['index']
            used_resources = self.env.num_spectrum_resources - self.env.topology.graph['available_spectrum'][edge_id]
            observation[edge_id, 0:used_resources] = 1
        for idp, path in enumerate(self.env.topology.graph['ksp'][self.env.service.source, self.env.service.destination]):
            start_index = (idp + 1) * self.env.num_spectrum_resources
            for i in range(len(path.node_list) - 1):
                edge_id = self.env.topology.edges[path.node_list[i], path.node_list[i+1]]['index']
                resulting_used_resources = self.env.num_spectrum_resources - self.env.topology.graph['available_spectrum'][edge_id] + 1
                observation[edge_id, start_index:start_index + resulting_used_resources] = 1
            if self.env.service.service_class == 0:
                # high-priority service only accepts the shortest path
                break
        service_class = np.zeros((1, 1))
        service_class[0, 0] = self.env.service.service_class
        return np.concatenate((observation.reshape((1, np.prod(observation.shape))), service_class), axis=1)
