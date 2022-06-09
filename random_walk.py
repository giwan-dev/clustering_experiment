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
    out_link_p = np.zeros(len(A))

    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] != 0:
                out_link_p[i] += 1

    for i in range(len(out_link_p)):
        if out_link_p[i] != 0:
            out_link_p[i] = 1 / out_link_p[i]

    P = np.zeros((len(A), len(A)))

    for i in range(len(P)):
        for j in range(len(P)):
            if A[i][j] != 0:
                P[i][j] = out_link_p[i]

    alpha = 0.00001
    e = np.array([np.ones(len(A))]).T

    def power_iteration(P, x, count):
        if count == 0:
            return x

        return power_iteration(P, alpha * np.matmul(P.T, x) + (1 - alpha) * e, count - 1)

    pi = power_iteration(P, (1 - alpha) * e, 10)

    pi_diagonal = np.zeros((len(A), len(A)))
    np.fill_diagonal(pi_diagonal, pi.T)

    U = (np.matmul(pi_diagonal, P) + np.matmul(P.T, pi_diagonal)) / 2

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

clustering.kmeans_clustering(G, vectors, indices, "label")
utils.print_graph_normalized_cut(G, "label")
utils.save_labels_in_csv(G, "random_walk", "label")
