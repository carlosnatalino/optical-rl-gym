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
        data["elements"].append({"uid": j,
                                 "metadata": {
                                     "location": {
                                        "city": "",
                                        "region": "",
                                        "latitude": i,
                                        "longitude": i
                                     }
                                 },
                                 "type_variety": "CienaDB_medium_gain",
                                 "type": "Edfa"})

    for node in topology.adj:
        for connected_node in topology.adj[node]:
            data["elements"].append({"uid": f"Fiber ({node} \u2192 {connected_node})",
                                     "type": "Fiber",
                                     "type_variety": "SSMF",
                                     "params": {
                                         "length": topology.adj[node][connected_node]['length'] / 10.0,
                                         "length_units": "km",
                                         "loss_coef": 0.2,
                                         "con_in": 1.00,
                                         "con_out": 1.00
                                     }})
            data["connections"].append({"from_node": node,
                                       "to_node": f"Fiber ({node} \u2192 {connected_node})"})
            data["connections"].append({"from_node": f"Fiber ({node} \u2192 {connected_node})",
                                       "to_node": connected_node})

    # print(topology.degree)
    # print(data['elements'])
    # print(type(topology))
    # print(dir(topology))
    return data


def load_files(gnpy_topology):
    """ Load GNPy equipment Library and create network from topology"""
    eqpt_library = load_equipment('../examples/default_equipment_data/eqpt_config.json')
    gnpy_network = network_from_json(topology_to_json(gnpy_topology), eqpt_library)

    return eqpt_library, gnpy_network


def propagation(input_power, network, sim_path, initial_slot, num_slots, eqpt, running_services, service, topology):
    """ Calculate and output SNR based on inputs
        input_power: Power in decibels
        network: Network created from GNPy topology
        sim_path: list of nodes service will travel through
        num_slots: number of slots in service
        eqpt: equipment library for GNPy """

    # print("initial_slot" + str(initial_slot))
    # print("num_slots: " + str(num_slots))

    # print("INPUT POWER => " + str(input_power))

    # Values to create Spectral Information object
    spacing = 12.5e9
    min_freq = 195942783006536 + initial_slot * spacing
    max_freq = min_freq + (num_slots - 1) * spacing
    p = input_power
    p = db2lin(p) * 1e-3
    # print("INPUT POWER => " + str(input_power))
    # print("INPUT POWER => " + str(p))
    si = create_input_spectral_information(min_freq, max_freq, 0.15, 32e9, p, spacing)

    print(sim_path)

    # for service in running_services:
    #     print(service.route.node_list)

    '''
    si.carriers=[
            Channel(f, (f_min + spacing * f),
                    baud_rate, roll_off, Power(power, 0, 0), 0, 0) for f in range(1, nb_channel + 1)
        ]
    '''

    p_total_db = input_power + lin2db(automatic_nch(min_freq, max_freq, spacing))
    # print(p_total_db, end=':')
    build_network(network, eqpt, input_power, p_total_db)

    # Store network elements
    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}
    roadms = {n.uid: n for n in network.nodes() if isinstance(n, Roadm)}
    fibers = {n.uid: n for n in network.nodes() if isinstance(n, Fiber)}
    edfas = {n.uid: n for n in network.nodes() if isinstance(n, Edfa)}

    # Recreate path in the GNPy network using node list from simulator
    path = []

    # print(fibers)

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
                    if edfa in edfas and edfa in fiber_neighbors:
                        path.append(edfas[edfa])

    current_node = 0
    # print(path[-1])
    # Calculate effects of physical layer impairments
    for el in path:
        if isinstance(el, Roadm) or isinstance(el, Transceiver):
            if current_node < len(sim_path) - 1:
                count = 0

                print(running_services[topology[sim_path[current_node]][sim_path[current_node + 1]]['index']])
                print("CARRIERS BEFORE => " + str(len(si.carriers)))

                for sid in running_services[topology[sim_path[current_node]][sim_path[current_node + 1]]['index']]:
                    # print(str(service.service_id) + " == " + str(sid) + " ? " + str(sid != -1 and sid != service.service_id))
                    if sid != -1 and sid != service.service_id:
                        count += 1

            
                print("OTHER SVCS ON THIS PATH => " + str(count))

                # print(si.carriers)

                si = si._replace(carriers=[
                    Channel(f, (min_freq + spacing * f),
                            32e9, 0.15, Power(p, 0, 0), 0, 0) for f in range(1, num_slots + count + 1)
                ])

                # print(si.carriers)
                print("CARRIERS AFTER => " + str(len(si.carriers)))
            current_node += 1

            # running_services[service.service_id, initial_slot:initial_slot + num_slots]
        # print(el)
        # print("Previous SI => " + str(si))
        # print(si.carriers)
        # print("ABOUT TO GO THROUGH => " + str(el))
        si = el(si)
        # print("SI => " + str(si))

    # print(list(path))
    # for el in path:
        # if isinstance(el, Edfa):
        # print(el)


    # print(path[-1]) 
    # print("NLI for each channel: " + str([c.power.nli for c in si.carriers]))
    destination_node = path[-1]

    # print("")

    # print(destination_node.snr, end=":")

    # print("Destination node SNR: " + str(destination_node.snr))
    return destination_node.snr
