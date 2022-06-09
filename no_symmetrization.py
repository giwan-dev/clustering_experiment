import utils
import load_data
import clustering

G = load_data.generate_graph()

vectors, indices = clustering.node2vec_embedding(G)

clustering.kmeans_clustering(G, vectors, indices, "label", cluster_count=60)
utils.print_graph_normalized_cut(G, "label")
utils.save_labels_in_csv(G, "no_symmetrization", "label")
