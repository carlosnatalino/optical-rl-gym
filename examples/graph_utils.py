import math
import numpy as np
import networkx as nx
from itertools import islice
from xml.dom.minidom import parse
import xml.dom.minidom

def get_k_shortest_paths(G, source, target, k, weight=None):
    '''
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    '''
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='weight'):
    return np.sum([graph[path[i]][path[i+1]][weight] for i in range(len(path) - 1)])


def calculate_geographical_distance(latlong1, latlong2):
    R = 6373.0

    lat1 = math.radians(latlong1[0])
    lon1 = math.radians(latlong1[1])
    lat2 = math.radians(latlong2[0])
    lon2 = math.radians(latlong2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    length = R * c
    return length


def read_sndlib_topology(file):
    graph = nx.Graph()

    with open(file) as file:
        tree = xml.dom.minidom.parse(file)
        document = tree.documentElement

        graph.graph['coordinatesType'] = document.getElementsByTagName('nodes')[0].getAttribute('coordinatesType')

        nodes = document.getElementsByTagName('node')
        for node in nodes:
            x = node.getElementsByTagName('x')[0]
            y = node.getElementsByTagName('y')[0]
            # print(node['id'], x.string, y.string)
            graph.add_node(node.getAttribute('id'), pos=((float(x.childNodes[0].data), float(y.childNodes[0].data))))
        # print('Total nodes: ', graph.number_of_nodes())
        links = document.getElementsByTagName('link')
        for idx, link in enumerate(links):
            source = link.getElementsByTagName('source')[0]
            target = link.getElementsByTagName('target')[0]

            if graph.graph['coordinatesType'] == 'geographical':
                length = np.around(calculate_geographical_distance(graph.nodes[source.childNodes[0].data]['pos'], graph.nodes[target.childNodes[0].data]['pos']), 3)
            else:
                latlong1 = graph.nodes[source.childNodes[0].data]['pos']
                latlong2 = graph.nodes[target.childNodes[0].data]['pos']
                length = np.around(math.sqrt((latlong1[0] - latlong2[0]) ** 2 + (latlong1[1] - latlong2[1]) ** 2), 3)

            weight = 1.0
            graph.add_edge(source.childNodes[0].data, target.childNodes[0].data,
                           id=link.getAttribute('id'), weight=weight, length=length, index=idx)

    return graph


def read_txt_file(file):
    graph = nx.Graph()
    num_nodes = 0
    num_links = 0
    id_link = 0
    with open(file, 'r') as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith('#')]
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                for id in range(1, num_nodes+1):
                    graph.add_node(str(id), name=str(id))
            elif idx == 1:
                num_links = int(line)
            elif len(line) > 1:
                info = line.replace('\n', '').split(' ')
                graph.add_edge(info[0], info[1], id=id_link, index=id_link, weight=1, length=int(info[2]))
                id_link += 1

    return graph


