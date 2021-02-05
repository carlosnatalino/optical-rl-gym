try:
    from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
    from gnpy.core.utils import db2lin, lin2db, automatic_nch
    from gnpy.core.info import create_input_spectral_information
    from gnpy.core.network import build_network
    from gnpy.tools.json_io import load_equipment, network_from_json
except:
    pass
from networkx import neighbors


def topology_to_json(topology):
    """ Load a NetworkX topology and transform it into GNPy JSON
        topology: NetworkX topology """
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


def load_files(gnpy_topology):
    """ Load GNPy equipment Library and create network from topology"""
    eqpt_library = load_equipment('../examples/default_equipment_data/eqpt_config.json')
    gnpy_network = network_from_json(topology_to_json(gnpy_topology), eqpt_library)

    return eqpt_library, gnpy_network


def propagation(input_power, network, sim_path, initial_slot, num_slots, eqpt):
    """ Calculate and output SNR based on inputs
        input_power: Power in decibels
        network: Network created from GNPy topology
        sim_path: list of nodes service will travel through
        num_slots: number of slots in service
        eqpt: equipment library for GNPy """

    # Values to create Spectral Information object
    spacing = 12.5e9
    min_freq = 195942783006536 + initial_slot * spacing
    max_freq = min_freq + (num_slots - 1) * spacing
    p = input_power
    p = db2lin(p) * 1e-3
    si = create_input_spectral_information(min_freq, max_freq, 0.15, 32e9, p, spacing)

    p_total_db = input_power + lin2db(automatic_nch(min_freq, max_freq, spacing))
    build_network(network, eqpt, input_power, p_total_db)

    # Store network elements
    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}
    fibers = {n.uid: n for n in network.nodes() if isinstance(n, Fiber)}
    edfas = {n.uid: n for n in network.nodes() if isinstance(n, Edfa)}

    # Recreate path in the GNPy network using node list from simulator
    path = []

    for index, node in enumerate(sim_path):
        # add transceiver to path
        path.append(transceivers[node])
        # add fiber connecting transceivers to path, unless source transceiver is last in path
        if index + 1 < len(sim_path):
            fiber_str = f"Fiber ({node} \u2192 {sim_path[index+1]})"
            for uid in fibers:
                # add all fibers to path even if they are split up
                if uid[0:len(fiber_str)] == fiber_str:
                    path.append(fibers[uid])
                    # add amplifier to path, if necessary
                    edfa = f"Edfa0_{uid}"
                    fiber_neighbors = [n.uid for n in neighbors(network, fibers[uid])]
                    if edfa in edfas and edfa in fiber_neighbors:
                        path.append(edfas[edfa])

    # Calculate effects of physical layer impairments
    for el in path:
        si = el(si)

    destination_node = path[-1]

    return destination_node.snr
