import numpy as np

import utils
import load_data
import clustering


def symmetrize(A):
    def nonzero_average(A):
        sum = 0
        count = 0

        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] != 0:
                    sum += A[i][j]
                    count += 1

        return sum / count

    def multiply_matrix(*args):
        sum = np.identity(len(args[0]))

        for arg in args:
            sum = np.matmul(sum, arg)

        return sum

    D_out = np.zeros((len(A), len(A)))
    np.fill_diagonal(D_out, np.sum(A, axis=1))
    D_in = np.zeros((len(A), len(A)))
    np.fill_diagonal(D_in, np.sum(A, axis=0))

    # remove zero
    for i in range(len(A)):
        if D_out[i][i] == 0:
            D_out[i][i] = 0.000000001
        if D_in[i][i] == 0:
            D_in[i][i] = 0.000000001

    B = multiply_matrix(
        np.sqrt(np.linalg.inv(D_out)),
        A,
        np.sqrt(np.linalg.inv(D_in)),
        A.T,
        np.sqrt(np.linalg.inv(D_out)),
    )

    C = multiply_matrix(
        np.sqrt(np.linalg.inv(D_in)),
        A.T,
        np.sqrt(np.linalg.inv(D_out)),
        A,
        np.sqrt(np.linalg.inv(D_in)),
    )

    U = B + C

    threshold = nonzero_average(U)

    for i in range(len(U)):
        for j in range(len(U)):
            if U[i][j] > threshold:
                U[i][j] = 1
            else:
                U[i][j] = 0

    np.fill_diagonal(U, 0)

    return U


G = load_data.generate_graph(symmetrize=symmetrize)

vectors, indices = clustering.node2vec_embedding(G)

clustering.kmeans_clustering(G, vectors, indices, "label", cluster_count=60)
utils.print_graph_normalized_cut(G, "label")
utils.save_labels_in_csv(G, "degree_discounted", "label")
