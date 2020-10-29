from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.utils import db2lin
from gnpy.core.info import create_input_spectral_information
from gnpy.core.network import build_network
from gnpy.tools.json_io import load_equipment, network_from_json
from networkx import dijkstra_path
from numpy import mean
from random import randint


def topology_to_json(topology):
    """ Currently loads a NetworkX topology from file and transforms it into GNPy JSON """
    data = {
        "elements": [],
        "connections": []
    }

    for i in topology.nodes:
        data["elements"].append({"uid": i,
                                 "metadata": {
                                     "location": {
                                        "city": "",
                                        "region": "",
                                        "latitude": randint(0, 100),
                                        "longitude": randint(0, 100)
                                     }
                                 },
                                 "type": "Transceiver"})
    for node in topology.adj:
        for connected_node in topology.adj[node]:
            data["elements"].append({"uid": f"Fiber ({node} \u2192 {connected_node})",
                                     # dummy data that works with GNPy's test eqpt_config.json
                                     "metadata": {
                                         "location": {
                                             "latitude": randint(0, 100),
                                             "longitude": randint(0, 100)
                                            }
                                         },
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


def propagation(input_power, con_in, con_out, source, dest, topology, eqpt):
    """ Create network topology from JSON and outputs SNR based on inputs """
    equipment = load_equipment(eqpt)
    json_data = topology_to_json(topology)
    network = network_from_json(json_data, equipment)
    build_network(network, equipment, 0, 20)

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

    p = input_power
    p = db2lin(p) * 1e-3
    # values from GNPy test_propagation.py
    spacing = 50e9  # THz
    si = create_input_spectral_information(191.3e12, 191.3e12 + 79 * spacing, 0.15, 32e9, p, spacing)

    source_node = next(transceivers[uid] for uid in transceivers if uid == source)
    sink = next(transceivers[uid] for uid in transceivers if uid == dest)
    path = dijkstra_path(network, source_node, sink)
    for el in path:
        si = el(si)

    # print(f'pw: {input_power} conn in: {con_in} con out: {con_out}',
    #       f'OSNR@0.1nm: {round(mean(sink.osnr_ase_01nm),2)}',
    #       f'SNR@bandwitdth: {round(mean(sink.snr),2)}')

    return sink.snr
