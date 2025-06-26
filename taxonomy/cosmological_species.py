import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# ================================================================
# COSMOLOGICAL SPECIES CLASSIFIER
# ================================================================

class CosmologicalSpeciesClassifier:
    def __init__(self, genome_sequences):
        self.genomes = genome_sequences

    def genome_to_vector(self, genome):
        """
        Convert symbolic genome string to numerical vector.
        """
        symbol_map = {'.': 0, 'a': 1, 'b': 2, 'c': 3, 'X': 4, 'E': 0}
        return np.array([symbol_map[s] for s in genome])

    def build_distance_matrix(self):
        vectors = [self.genome_to_vector(g) for g in self.genomes]
        vectors = np.array(vectors)
        num = len(vectors)
        dist_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                dist_matrix[i, j] = np.sum(np.abs(vectors[i] - vectors[j]))
        return dist_matrix

    def cluster_species(self, num_species=4):
        dist_matrix = self.build_distance_matrix()
        clustering = AgglomerativeClustering(n_clusters=num_species, affinity='precomputed', linkage='average')
        labels = clustering.fit_predict(dist_matrix)
        return labels

    def visualize_species(self, num_species=4):
        dist_matrix = self.build_distance_matrix()
        clustering = AgglomerativeClustering(n_clusters=num_species, affinity='precomputed', linkage='average')
        labels = clustering.fit_predict(dist_matrix)

        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.scatter(i, 0, color=plt.cm.tab10(label), s=100, label=f'Species {label}' if i == 0 else "")
        plt.title("Cosmological Species Classification")
        plt.yticks([])
        plt.legend()
        plt.show()

    def plot_dendrogram(self):
        vectors = [self.genome_to_vector(g) for g in self.genomes]
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(vectors, method='ward')
        plt.figure(figsize=(12, 6))
        dendrogram(Z)
        plt.title("Cosmological Species Evolution Tree")
        plt.show()
