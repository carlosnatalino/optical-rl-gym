from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import db2lin
from gnpy.core.info import create_input_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_equipment, network_from_json
from networkx import dijkstra_path, shortest_simple_paths, neighbors
from examples.graph_utils import get_k_shortest_paths
import numpy as np


def topology_to_json(topology):
    """ Load a NetworkX topology and transform it into GNPy JSON """
    data = {
        "elements": [],
        "connections": []
    }

    for i, j in enumerate(topology.nodes):
        data["elements"].append({"uid": j,
                                 "metadata": {
                                     "location": {
                                        "city": "",
                                        "region": "",
                                        "latitude": i,
                                        "longitude": i
                                     }
                                 },
                                 "type": "Transceiver"})
    for node in topology.adj:
        for connected_node in topology.adj[node]:
            data["elements"].append({"uid": f"Fiber ({node} \u2192 {connected_node})",
                                     "type": "Fiber",
                                     "type_variety": "SSMF",
                                     "params": {
                                         "length": topology.adj[node][connected_node]['length'],
                                         "length_units": "km",
                                         "loss_coef": 0.2,
                                         "con_in": 1.00,
                                         "con_out": 1.00
                                     }})
            data["connections"].append({"from_node": node,
                                       "to_node": f"Fiber ({node} \u2192 {connected_node})"})
            data["connections"].append({"from_node": f"Fiber ({node} \u2192 {connected_node})",
                                       "to_node": connected_node})
    return data


def propagation(input_power, con_in, con_out, network, sim_path, eqpt):
    """ Create network topology from JSON and outputs SNR based on inputs """
    build_network(network, eqpt, 0, 20)

    # parametrize the network elements with the con losses and adapt gain
    # (assumes all spans are identical)
    for e in network.nodes():
        if isinstance(e, Fiber):
            loss = e.params.loss_coef * e.params.length
            e.params.con_in = con_in
            e.params.con_out = con_out
        if isinstance(e, Edfa):
            e.operational.gain_target = loss + con_in + con_out

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}
    fibers = {n.uid: n for n in network.nodes() if isinstance(n, Fiber)}
    edfas = {n.uid: n for n in network.nodes() if isinstance(n, Edfa)}

    p = input_power
    # p = db2lin(p) * 1e-3
    # values from GNPy test_propagation.py
    spacing = 50e9  # THz
    si = create_input_spectral_information(191.3e12, 191.3e12 + 79 * spacing, 0.15, 32e9, p, spacing)

    # source_node = next(transceivers[uid] for uid in transceivers if uid == sim_path[0])
    # sink = next(transceivers[uid] for uid in transceivers if uid == sim_path[-1])
    # path = get_k_shortest_paths(network, source_node, sink, k_paths)[0]

    path = []

    for index, node in enumerate(sim_path):
        path.append(transceivers[node])
        if index + 1 < len(sim_path):
            fiber_str = f"Fiber ({node} \u2192 {sim_path[index+1]})"
            for uid in fibers:
                if uid[0:len(fiber_str)] == fiber_str:
                    path.append(fibers[uid])
                    edfa = f"Edfa0_{uid}"
                    fiber_neighbors = [n.uid for n in neighbors(network, fibers[uid])]
                    # print([i for i in fiber_neighbors if 'Edfa' in i])
                    if edfa in edfas and edfa in fiber_neighbors:
                        path.append(edfas[edfa])
    for el in path:
        si = el(si)

    # print([i.uid for i in path])

    return path[-1].snr
