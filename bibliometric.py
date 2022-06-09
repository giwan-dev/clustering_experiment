import numpy as np

import utils
import load_data
import clustering


def symmetrize(A):
    A = A + np.identity(len(A))
    U = np.matmul(A, A.T) + np.matmul(A.T, A)
    np.fill_diagonal(U, 0)
    return U


G = load_data.generate_graph(symmetrize=symmetrize)

vectors, indices = clustering.node2vec_embedding(G)

clustering.kmeans_clustering(G, vectors, indices, "label")
utils.print_graph_normalized_cut(G, "label")
utils.save_labels_in_csv(G, "bibliometric", "label")
