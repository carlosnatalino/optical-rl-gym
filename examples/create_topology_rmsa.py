import optical_rl_gym
from optical_rl_gym.utils import Path

from itertools import islice
import numpy as np
import networkx as nx
import math
import pickle
from xml.dom.minidom import parse
import xml.dom.minidom

import matplotlib.pyplot as plt

from graph_utils import read_sndlib_topology, read_txt_file, get_k_shortest_paths, get_path_weight


def get_modulation_format(length, modulations):
    for i in range(len(modulations) - 1):
        if length > modulations[i + 1]['maximum_length'] and length <= modulations[i]['maximum_length']:
            #             print(length, i, modulations[i]['modulation'])
            return modulations[i]
    # if length <= modulations[-1]['maximum_length']:
        #         print(length, len(modulations) - 1, modulations[len(modulations) - 1]['modulation'])
    return modulations[len(modulations) - 1]


def get_topology(file_name, topology_name, modulations, k_paths=5):
    k_shortest_paths = {}
    if file_name.endswith('.xml'):
        topology = read_sndlib_topology(file_name)
    elif file_name.endswith('.txt'):
        topology = read_txt_file(file_name)
    else:
        raise ValueError('Supplied topology is unknown')
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight='length')
                print(n1, n2, len(paths))
                lengths = [get_path_weight(topology, path, weight='length') for path in paths]
                selected_modulations = [get_modulation_format(length, modulations) for length in lengths]
                objs = []
                for path, length, modulation in zip(paths, lengths, selected_modulations):
                    objs.append(Path(idp, path, length, best_modulation=modulation))
                    print('\t', idp, length, modulation, path)
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph['name'] = topology_name
    topology.graph['ksp'] = k_shortest_paths
    topology.graph['modulations'] = modulations
    topology.graph['k_paths'] = k_paths
    topology.graph['node_indices'] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph['node_indices'].append(node)
        topology.nodes[node]['index'] = idx
    return topology


# defining the EON parameters
# definitions according to : https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
modulations = list()
# modulation: string description
# capacity: Gbps
# maximum_distance: km
modulations.append({'modulation': 'BPSK', 'capacity': 12.5, 'maximum_length': 100000})
modulations.append({'modulation': 'QPSK', 'capacity': 25., 'maximum_length': 2000})
modulations.append({'modulation': '8QAM', 'capacity': 37.5, 'maximum_length': 1250})
modulations.append({'modulation': '16QAM', 'capacity': 50., 'maximum_length': 625})


# other setup:
# modulations.append({'modulation': 'BPSK', 'capacity': 12.5, 'maximum_length': 4000})
# modulations.append({'modulation': 'QPSK', 'capacity': 25., 'maximum_length': 2000})
# modulations.append({'modulation': '8QAM', 'capacity': 37.5, 'maximum_length': 1000})
# modulations.append({'modulation': '16QAM', 'capacity': 50., 'maximum_length': 500})
# modulations.append({'modulation': '32QAM', 'capacity': 62.5, 'maximum_length': 250})
# modulations.append({'modulation': '64QAM', 'capacity': 75., 'maximum_length': 125})

k_paths = 5

# The paper uses K=5 and J=1
topology = get_topology('./topologies/nsfnet_chen.txt', 'NSFNET', modulations, k_paths=k_paths)

with open(f'./topologies/nsfnet_chen_eon_{k_paths}-paths.h5', 'wb') as f:
    pickle.dump(topology, f)

print('done for', topology)