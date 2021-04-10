try:
    from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
    from gnpy.core.utils import db2lin, lin2db, automatic_nch
    from gnpy.core.info import create_input_spectral_information, Channel, Power
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
        data["elements"].append({"uid": j,
                                 "metadata": {
                                     "location": {
                                        "city": "",
                                        "region": "",
                                        "latitude": i,
                                        "longitude": i
                                     }
                                 },
                                 "type": "Roadm"})
        # data["elements"].append({"uid": j,
        #                          "metadata": {
        #                              "location": {
        #                                 "city": "",
        #                                 "region": "",
        #                                 "latitude": i,
        #                                 "longitude": i
        #                              }
        #                          },
        #                          "type_variety": "CienaDB_medium_gain",
        #                          "type": "Edfa"})

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


def propagation(input_power, network, sim_path, initial_slot, num_slots, eqpt, slots_allocation, service, topology):
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
    roadms = {n.uid: n for n in network.nodes() if isinstance(n, Roadm)}
    fibers = {n.uid: n for n in network.nodes() if isinstance(n, Fiber)}
    edfas = {n.uid: n for n in network.nodes() if isinstance(n, Edfa)}

    # Recreate path in the GNPy network using node list from simulator
    path = []

    for index, node in enumerate(sim_path):
        # add roadm to path
        if index == len(sim_path) - 1 or index == 0:
            path.append(transceivers[node])
        else:
            path.append(roadms[node])

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
                    # if edfa in edfas and edfa in fiber_neighbors:
                    #     path.append(edfas[edfa])

    current_node = 0
    sim_path_si = {}

    print("potential service => " + str(sim_path))

    # Calculate effects of physical layer impairments
    for el in path:
        if isinstance(el, Roadm) or isinstance(el, Transceiver):

            # print("")
            # print("@" + sim_path[current_node])

            sim_path_si[sim_path[current_node]] = si.carriers

            if current_node < len(sim_path) - 1:
                adjacent_services = {}

                for rs in topology.graph['running_services']:
                    for i in range(len(sim_path)):
                        if sim_path[i] in rs.route.node_list:
                            rs_in = rs.route.node_list.index(sim_path[i])

                            if rs_in + 1 < len(rs.route.node_list) and i + 1 < len(sim_path) and rs.route.node_list[rs_in + 1] == sim_path[i + 1] and rs.source == sim_path[current_node]:
                                # print("rs_in = " + str(rs_in) + ", match => " + str(rs.route.node_list) + ", adding " + str(rs.number_slots) + "slots")
                                adjacent_services[rs.service_id] = rs

                carriers = list(si.carriers[0:num_slots])

                # print("initially...")
                # print("len(carriers) = " + str(len(carriers)))
                # print([c.channel_number for c in list(carriers)])

                for sid, s in adjacent_services.items():
                    ref = s.power_values[sim_path[current_node]]
                    l = len(carriers)

                    for i in range(len(ref)):
                        carriers.append(
                            Channel(
                                channel_number=l + i + 1, 
                                frequency=(min_freq + spacing * ref[i].channel_number), 
                                baud_rate=32e9, 
                                roll_off=0.15, 
                                power=ref[i].power,
                                chromatic_dispersion=ref[i].chromatic_dispersion,
                                pmd=ref[i].pmd
                            )
                        )

                si = si._replace(carriers=carriers)

            current_node += 1

        # print([ch.power.nli for ch in si.carriers])

        si = el(si)

    destination_node = path[-1]

    print("FINAL VALUE " + str(destination_node.snr))

    return [destination_node.snr, sim_path_si]
