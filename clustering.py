import multiprocessing
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sknetwork.clustering import Louvain


def node2vec_embedding(G):
    cpu_cores = multiprocessing.cpu_count()

    node2vec = Node2Vec(
        graph=G,
        weight_key=None,
        p=0.1,
        q=1,
        dimensions=64,
        walk_length=20,
        num_walks=cpu_cores * 200,
        workers=cpu_cores,
    )

    model = node2vec.fit(window=5)

    return model.wv.vectors, model.wv.index_to_key


def kmeans_clustering(G, vectors, indices, cluster_label, cluster_count = 60):
    kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(vectors)

    for n, label in zip(indices, kmeans.labels_):
        G.nodes[int(n)][cluster_label] = label
