import utils
import load_data
import clustering


def symmetrize(A):
    return A + A.T


G = load_data.generate_graph(symmetrize=symmetrize)

vectors, indices = clustering.node2vec_embedding(G)

clustering.kmeans_clustering(G, vectors, indices, "label")
utils.print_graph_normalized_cut(G, "label")
utils.save_labels_in_csv(G, "simple_symmetrization", "label")
