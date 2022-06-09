import os
import inspect
import networkx as nx
import numpy as np
from os.path import abspath
import inspect
import os


def generate_graph(symmetrize = None):
    list_node = []
    list_edge = []

    filename = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), './edge-table.csv')

    f = open(filename, 'r')

    f.readline()

    for line in f.readlines():
        linedata = line.strip().split(',')

        if linedata[0] not in list_node:
            list_node.append(linedata[0])
        if linedata[1] not in list_node:
            list_node.append(linedata[1])

        list_edge.append((linedata[0], linedata[1]))

    f.close()

    num_node = len(list_node)
    label_to_index = {}

    for index in range(num_node):
        label_to_index[list_node[index]] = index

    A = np.zeros((num_node, num_node))

    for (u, v) in list_edge:
        A[label_to_index[u], label_to_index[v]] = 1

    if symmetrize != None:
        A = symmetrize(A)

    G = nx.Graph(A)

    for i in range(len(list_node)):
        G.nodes[i]['name'] = list_node[i]

    return G
