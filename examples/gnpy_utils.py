import json


def topology_to_json(file):
    data = {
        "elements": [],
        "connections": []
    }
    with open(file, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
        for line_num, line in enumerate(lines):
            if line_num == 0:
                num_nodes = int(line)
                for i in range(1, num_nodes + 1):
                    data["elements"].append({"uid": i,
                                             # "metadata": {
                                                 # "location": {
                                                 #    "city": "",
                                                 #    "region": "",
                                                 #    "latitude": 0,
                                                 #    "longitude": 0
                                                 # }
                                             # },
                                             "type": "Transceiver"})
            elif line_num == 1:
                pass
            else:
                begin, end, length = [int(part) for part in line.split()]
                data["elements"].append({"uid": f"Fiber ({begin} \u2192 {end})",
                                         # dummy data that works with GNPy's test eqpt_config.json
                                         # "metadata": {
                                         #     "location": {
                                         #         "latitude": 0.0,
                                         #         "longitude": 0.0
                                         #        }
                                         #     },
                                         "type": "Fiber",
                                         "type_variety": "SSMF",
                                         "params": {
                                             "length": length,
                                             "length_units": "km",
                                             "loss_coef": 0.2,
                                             "con_in": 1.00,
                                             "con_out": 1.00
                                         }})
                data["connections"].append({"from_node": begin,
                                           "to_node": f"Fiber ({begin} \u2192 {end})"})
                data["connections"].append({"from_node": f"Fiber ({begin} \u2192 {end})",
                                           "to_node": end})
    return data


if __name__ == "__main__":
    filename = "./topologies/nsfnet_chen.txt"
    output = topology_to_json(filename)
    with open('mock_network.json', 'w') as outfile:
        json.dump(output, outfile, indent=1, separators=(',', ': '))
        print("Dumped.")
