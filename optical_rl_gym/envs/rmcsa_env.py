import gym
import copy
import math
import heapq
import logging
import functools
import numpy as np

from collections import defaultdict

from optical_rl_gym.utils import Service, Route
from .optical_network_env import OpticalNetworkEnv


class RMCSAEnv(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate',
                    'bit_rate_blocking_rate', 'episode_bit_rate_blocking_rate']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=100,
                 num_spatial_resources=3,  # number of cores - 3, 7, 11, 22

                 modulation_formats=np.array([
                     {  # BPSK
                         'mod_factor': 1,
                         'snr_min': 4.2,
                         'inband_xt': -14
                     },  # BPSK
                     {  # QPSK
                         'mod_factor': 2,
                         'snr_min': 7.2,
                         'inband_xt': -17
                     },  # QPSK
                     {  # 16QAM
                         'mod_factor': 4,
                         'snr_min': 13.9,
                         'inband_xt': -23
                     },  # 16QAM
                     {  # 64QAM
                         'mod_factor': 6,
                         'snr_min': 19.8,
                         'inband_xt': -29
                     },  # 64QAM
                 ]),
                 worst_xt=None,

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
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.utilization = []
        self.core_utilization = defaultdict(list)

        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound

        self.num_spatial_resources = num_spatial_resources  # number of cores

        self.modulation_formats = modulation_formats

        if worst_xt is None:
            self.worst_xt = _worst_xt_by_core(num_spatial_resources)
        else:
            self.worst_xt = worst_xt

        # Adding a 4db penalty margin as most operators do (4 dB in our case) to both ASE and XT limits values
        for format in self.modulation_formats:
            format['inband_xt'] += 4
        self.worst_xt += 4

        self.spectrum_slots_allocation = np.full((self.num_spatial_resources, self.topology.number_of_edges(),
                                                  self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1,
                                        self.num_spatial_resources + 1,
                                       self.num_spectrum_resources + 1),
                                       dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + 1,
                                                self.num_spatial_resources + 1,
                                                self.num_spectrum_resources + 1),
                                               dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1,
                                       self.num_spatial_resources + 1,
                                      self.num_spectrum_resources + 1),
                                      dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1,
                                               self.num_spatial_resources + 1,
                                               self.num_spectrum_resources + 1),
                                              dtype=int)
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action,
                                                      len(self.modulation_formats),
                                                      self.num_spatial_resources + self.reject_action,
                                                      self.num_spectrum_resources + self.reject_action))
        self.observation_space = gym.spaces.Dict({'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)
        })
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmcsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action: [int]):
        # Compute statistics for analysis
        route, modulation, core, initial_slot = action[0], action[1], action[2], action[3]

        # ???
        self.actions_output[route, core, initial_slot] += 1

        # Check if the decision that was passed is valid
        if route < self.k_paths and \
                core < self.num_spatial_resources and \
                initial_slot < self.num_spectrum_resources:

            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][route])

            self.logger.debug('{} processing action {} route {} and initial slot {} for {} slots'.format(self.service.service_id, action, route, initial_slot, slots))
            if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][route], core,
                                 initial_slot, slots):

                # The length of the path that was chosen by the agent
                path_length = self.k_shortest_paths[self.service.source, self.service.destination][route].length

                # Note for future: Wether or not this goes before or after is_path_free depends on which is faster
                # Check  that the chosen path length is viable considering crosstalk
                if self._crosstalk_is_acceptable(modulation, path_length):

                    # Set the resources that were free to "used"
                    self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][route], core,
                                         initial_slot, slots)
                    # More statistics
                    self.service.accepted = True
                    self.actions_taken[route, core, initial_slot] += 1

                    # Schedule the event for the resources to leave the network (after amount of time)
                    self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths, self.num_spatial_resources, self.num_spectrum_resources] += 1

        # More statistics
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.service.bit_rate
        self.episode_bit_rate_requested += self.service.bit_rate

        self.topology.graph['services'].append(self.service)

        # Get the value of the action
        reward = self.reward()
        # Summarize computed statistics
        info = {
                   'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
                   'episode_service_blocking_rate': (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
                   'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
                   'episode_bit_rate_blocking_rate': (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / self.episode_bit_rate_requested
               }

        self._new_service = False
        self._next_service()
        return self.observation(), reward, self.episode_services_processed == self.episode_length, info


    def _crosstalk_is_acceptable(self, current_modulation, path_length) -> bool:
        """
        Checks that the crosstalk for the given modulation is within the maximum calculated for that format
        """

        modulation = self.modulation_formats[current_modulation]


        average_power=1 # Average power used (in mW - milliWatts)
        nf_db=5.5  # noise factor of the amplifiers [dB]
        nf = 10.0 ** (nf_db / 10.0)

        amp_spam=100  # distance between amplifiers [km]
        amp_gain_db = 20  # gain on the amplifiers [dB]
        amp_gain = 10.0 ** (amp_gain_db / 10.0)

        lambda_=1550  # wavelength [nm]
        h = 6.626068e-34  # Plank's constant
        f_hz = 2.99e8 / (lambda_ * 1e-9)  # signal frequency [Hz]

        # we consider +2dB as the margin to make sure the channel works
        SNR_min_calc = 10 ** ((modulation['snr_min'] + 2) / 10)  # +2 to to compensate oscilation and convert

        lmax_snr = (average_power * amp_spam) / (SNR_min_calc * h * f_hz * amp_gain * nf * (
                    self.service.bit_rate / modulation['mod_factor']) * 1e9)  # eq. (1)
        lmax_snr = lmax_snr / 1000  # convert to km

        lmax_xt = 10 ** ((modulation['inband_xt'] - self.worst_xt - 4) / 10)  # Eq. (2) and -4 for penalty margin

        if path_length < lmax_xt and path_length < lmax_snr:
            return True
        else:
            return False

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                                self.num_spectrum_resources + self.reject_action),
                                               dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + self.reject_action,
                                               self.num_spectrum_resources + self.reject_action),
                                              dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones((self.num_spatial_resources, self.topology.number_of_edges(),
                                                          self.num_spectrum_resources), dtype=int)

        self.spectrum_slots_allocation = np.full((self.num_spatial_resources, self.topology.number_of_edges(),
                                                  self.num_spectrum_resources),
                                                 fill_value=-1, dtype=np.int)

        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['external_fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    def _provision_path(self, route: Route, core: int, initial_slot: int, number_slots: int):
        if not self.is_path_free(route, core, initial_slot, number_slots):
            raise ValueError("Route {} has not enough capacity on slots {}-{}".format(route.node_list, route, initial_slot,
                                                                                     initial_slot + number_slots))

        self.logger.debug('{} assigning route {} on initial slot {} for {} slots'.format(self.service.service_id,
                                                                                        route.node_list, initial_slot, number_slots))
        for i in range(len(route.node_list) - 1):
            self.topology.graph['available_slots'][core, self.topology[route.node_list[i]][route.node_list[i + 1]]['index'],
                                                                        initial_slot:initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[core, self.topology[route.node_list[i]][route.node_list[i + 1]]['index'],
                                                    initial_slot:initial_slot + number_slots] = self.service.service_id
            self.topology[route.node_list[i]][route.node_list[i + 1]]['services'].append(self.service)
            self.topology[route.node_list[i]][route.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(core, route.node_list[i], route.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service)
        self.service.route = route
        self.service.initial_slot = initial_slot
        self.service.number_slots = number_slots
        self.service.core = core
        self._update_network_stats(core)

        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.episode_bit_rate_provisioned += self.service.bit_rate

    def _release_path(self, service: Service):
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_slots'][service.core,
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                                            service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.spectrum_slots_allocation[service.core,
                    self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                                            service.initial_slot:service.initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.core, service.route.node_list[i], service.route.node_list[i + 1])
        self.topology.graph['running_services'].remove(service)

    def _update_network_stats(self, core: int):
        """
        Update network stats is used to create metrics for "throughput" & "compactness".

        :param core: number of cores
        """
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']

            last_compactness = self.topology.graph['compactness']

            cur_throughput = 0.

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = ((last_throughput * last_update) + (cur_throughput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = throughput

            compactness = ((last_compactness * last_update) + (self._get_network_compactness(core) * time_diff)) / \
                              self.current_time
            self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, core: int, node1: str, node2: str):

        """ Creates metrics for:
        Individual node "utilization", overall "core_utilization", "external fragmentation", and "link_compactness".

        :param core : number of cores,
        :param node1: number of node1 within the node_list
        :param node2: number of node2 within the node_list
        """

        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']

        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']
            cur_util = (self.num_spectrum_resources - np.sum(
                self.topology.graph['available_slots'][core, self.topology[node1][node2]['index'], :])) / \
                       self.num_spectrum_resources
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization
            # Adds each node utilization value to an array
            self.utilization.append(utilization)
            # Adds each node utilization value to the core key within a dictionary
            self.core_utilization[core].append(utilization)

            slot_allocation = self.topology.graph['available_slots'][core, self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_external_fragmentation = 0.
            cur_link_compactness = 0.
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMCSAEnv.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMCSAEnv.rle(
                        slot_allocation[lambda_min:lambda_max])
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = ((lambda_max - lambda_min) / np.sum(1 - slot_allocation)) * (
                                    1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.
                else:
                    cur_link_compactness = 1.

            external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation

            link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness


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

        self.service = Service(self.episode_services_processed, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)
        self._new_service = True

    def _get_route_slot_id(self, action: int) -> (int, int):
        """
        Decodes the single action index into the path index and the slot index to be used.

        :param action: the single action index
        :return: route index and initial slot index encoded in the action
        """
        route = int(action / self.num_spectrum_resources)
        initial_slot = action % self.num_spectrum_resources
        return route, initial_slot

    def get_number_slots(self, route: Route) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        We assume that the most efficient modulation format (within path.best_modulation) will be used.
        """
        return math.ceil(self.service.bit_rate / route.best_modulation['capacity']) + 1

    def is_path_free(self, path: Route, core: int, initial_slot: int, number_slots: int) -> bool:
        """
        Method that determines if the path is free for the core, path, and initial_slot.

        :param core: Number of cores currently being used
        :param path: Index of K shortest paths
        :param initial_slot: The current frequency slot being used <-carlos pls double check
        :param number_slots: The total number of slots

        :return: True/False
        :rtype: bool
        """
        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph['available_slots'][
                      core,
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      initial_slot:initial_slot + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, route: Route):
        available_slots = functools.reduce(np.multiply,
                                           self.topology.graph["available_slots"][[self.topology[route.node_list[i]][route.node_list[i + 1]]['id']
                                                                                   for i in range(len(route.node_list) - 1)], :])
        return available_slots

    def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def _get_network_compactness(self, core):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_routes = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_routes += service.number_slots * service.route.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = \
                RMCSAEnv.rle(self.topology.graph['available_slots'][core, self.topology[n1][n2]['index'], :])
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMCSAEnv.rle(
                    self.topology.graph['available_slots'][core, self.topology[n1][n2]['index'], lambda_min:lambda_max])
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_routes) * (self.topology.number_of_edges() /
                                                                           sum_unused_spectrum_blocks)
        else:
            cur_spectrum_compactness = 1.

        return cur_spectrum_compactness

def _worst_xt_by_core(cores) -> float:
    """
    Assigns a default worst crosstalk value based on the number of cores
    """
    worst_crosstalks_by_core = {7: -84.7, 12: -61.9, 19: -54.8}  # Cores: Crosstalk in dB
    worst_xt = worst_crosstalks_by_core.get(cores)  # Worst aggregate intercore XT
    return worst_xt

def shortest_available_path_first_core_first_fit(env: RMCSAEnv) -> int:
    """
    Algorithm for determining the shortest available first core first fit path

    :param env: OpenAI Gym object containing RMCSA environment
    :return: Cores, paths, and number of spectrum resources
    """
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        num_slots = env.get_number_slots(path)
        # Iteration of core
        for core in range(env.num_spatial_resources):
            for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
                if env.is_path_free(path, core, initial_slot, num_slots):
                    return [idp, core, initial_slot]
    return [env.topology.graph['k_paths'], env.num_spatial_resources, env.topology.graph['num_spectrum_resources']]


class SimpleMatrixObservation(gym.ObservationWrapper):

    def __init__(self, env: RMCSAEnv):
        super().__init__(env)
        shape = self.env.topology.number_of_nodes() * 2 \
                + self.env.topology.number_of_edges() * self.env.num_spectrum_resources * self.env.num_spatial_resources
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

    def __init__(self, env: RMCSAEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(self.env.k_paths + self.env.reject_action)
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(self.env.k_shortest_paths[self.env.service.source,
                                                                            self.env.service.destination][action])
            for initial_slot in range(0, self.env.topology.graph['num_spectrum_resources'] - num_slots):
                if self.env.is_path_free(self.env.k_shortest_paths[self.env.service.source,
                                                                   self.env.service.destination][action],
                                         initial_slot, num_slots):
                    return [action, initial_slot]
        return [self.env.topology.graph['k_paths'], self.env.topology.graph['num_spectrum_resources']]

    def step(self, action):
        return self.env.step(self.action(action))
