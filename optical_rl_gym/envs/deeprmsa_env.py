import gym
import numpy as np

from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRMSAEnv(RMSAEnv):

    def __init__(self, topology=None, j=1,
                 episode_length=1000,
                 mean_service_holding_time=25.0,
                 mean_service_inter_arrival_time=10.,
                 num_spectrum_resources=100,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False):
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=mean_service_holding_time * mean_service_inter_arrival_time,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths,
                         allow_rejection=allow_rejection,
                         reset=False)

        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)

    def step(self, action: int):
        self.actions_output[action] += 1
        if action < self.k_paths * self.j:  # action is for assigning a path
            path, block = self._get_path_block_id(action)

            initial_indices, lengths = self.get_available_blocks(path)

            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])

            # logging.debug('{} processing action {} path {} and initial slot {} for {} slots'.format(service.service_id, action, path, initial_slot, slots))
            if block < len(initial_indices) \
                    and lengths[block] >= slots \
                    and self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                 initial_indices[block], slots):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                     initial_indices[block], slots)
                self.service.accepted = True
                self.actions_taken[action] += 1
                self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False

        if not self.service.accepted:
            self.actions_taken[self.k_paths * self.j + self.reject_action - 1] += 1

        self.services_processed += 1.
        self.services_processed_since_reset += 1.
        self.bit_rate_requested += self.service.bit_rate
        self.bit_rate_requested_since_reset += self.service.bit_rate

        self.topology.graph['services'].append(self.service)

        reward = self.reward()
        info = {
                   'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
                   'service_blocking_rate_since_reset': (self.services_processed_since_reset - self.services_accepted_since_reset) / float(self.services_processed_since_reset),
                   'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
                   'bit_rate_blocking_rate_since_reset': (self.bit_rate_requested_since_reset - self.bit_rate_provisioned_since_reset) / self.bit_rate_requested_since_reset
               }

        self._new_service = False
        self._next_service()
        return self.observation(), reward, self.services_processed_since_reset == self.episode_length, info

    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.service.source_id, self.service.destination_id)
        max_node = max(self.service.source_id, self.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths, 2 * self.j + 3), fill_value=-1.)
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            available_slots = self.get_available_slots(path)
            num_slots = self.get_number_slots(path)
            initial_indices, lengths = self.get_available_blocks(idp)

            for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                # initial slot index
                spectrum_obs[idp, idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

                # number of contiguous FS available
                spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp, self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

            idx, values, lengths = DeepRMSAEnv.rle(available_slots)

            av_indices = np.argwhere(values == 1) # getting indices which have value 1
            spectrum_obs[idp, self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number available FSs
            spectrum_obs[idp, self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100

        return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
            .reshape(self.observation_space.shape)

    def reward(self):
        return 1 if self.service.accepted else -1

    def reset(self, only_counters=True):
        self.bit_rate_requested_since_reset = 0
        self.bit_rate_provisioned_since_reset = 0
        self.services_processed_since_reset = 0
        self.services_accepted_since_reset = 0
        self.actions_output_since_reset = np.zeros(self.k_paths + self.reject_action, dtype=int)
        self.actions_taken_since_reset = np.zeros(self.k_paths + self.reject_action, dtype=int)

        if only_counters:
            return self.observation()

        OpticalNetworkEnv.reset(self)

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

    def _get_path_block_id(self, action: int) -> (int, int):
        path = action // self.k_paths
        block = action % self.k_paths
        return path, block

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

    def get_available_blocks(self, path):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path])

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path])

        # getting the blocks
        initial_indices, values, lengths = DeepRMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[:self.j]

        return initial_indices[final_indices], lengths[final_indices]


def shortest_path_first_fit(env: RMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, lengths = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: RMSAEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        initial_indices, lengths = env.get_available_blocks(idp)
        if len(initial_indices) > 0: # if there are available slots
            return idp * env.k_paths # this path uses the first one
    return env.k_paths * env.j
