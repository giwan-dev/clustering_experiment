import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_graph(G, cluster_label):
    plt.figure(figsize=(12, 6))
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G),
                     node_color=[n[1][cluster_label] for n in G.nodes(data=True)],
                     cmap=plt.cm.rainbow,
                     node_size=10,
                     with_labels=False,
                     width=0.1
                     )
    plt.axis('off')
    plt.show()


def save_labels_in_csv(G, name, cluster_label):
    labels_file = open("./labels_%s.csv" % name, "w")
    labels_file.write("Id,%s Cluster\n" % name)
    for node in G.nodes(data=True):
        labels_file.write("%s,%s\n" %
                          (node[1]["name"], node[1][cluster_label]))
    labels_file.close()


def print_graph_normalized_cut(G: nx.Graph, cluster_label):
    labels_unique = np.unique([node[1][cluster_label]
                              for node in G.nodes(data=True)])

    cut = np.zeros(len(labels_unique))
    vol = np.zeros(len(labels_unique))

    for start, end, _ in G.edges(data=True):
        start_cluster = G.nodes[start][cluster_label]
        end_cluster = G.nodes[end][cluster_label]

        vol[start_cluster] += 1

        if start_cluster != end_cluster:
            cut[start_cluster] += 1

    for i in range(len(vol)):
        if vol[i] == 0:
            vol[i] = 1

    normalized_cut = cut / vol
    graph_normalized_cut = np.sum(normalized_cut)

    print("Graph normalized cut of %s: %s" %
          (cluster_label,  graph_normalized_cut))
