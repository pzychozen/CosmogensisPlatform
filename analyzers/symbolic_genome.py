import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# SYMBOLIC GENOME CORE: Recursive Universe Genome Encoder
# ================================================================

class SymbolicGenome:
    def __init__(self, memory_lattice):
        self.memory = memory_lattice

    def encode_snapshot(self, centroids, grid_size=128, bin_resolution=8):
        """
        Compress centroids into discrete symbolic bins across space.
        """
        if len(centroids) == 0:
            return 'E'  # Empty state

        bins = np.zeros((bin_resolution, bin_resolution), dtype=int)

        for c in centroids:
            x_bin = int(c[0] / grid_size * bin_resolution)
            y_bin = int(c[1] / grid_size * bin_resolution)
            bins[x_bin, y_bin] += 1

        # Convert to symbolic string
        symbols = ''
        for row in bins:
            for count in row:
                if count == 0:
                    symbols += '.'
                elif count == 1:
                    symbols += 'a'
                elif count == 2:
                    symbols += 'b'
                elif count == 3:
                    symbols += 'c'
                else:
                    symbols += 'X'
        return symbols

    def build_genome_sequence(self):
        genome_sequence = []
        for snapshot in self.memory.memory:
            centroids = snapshot['centroids']
            code = self.encode_snapshot(centroids)
            genome_sequence.append(code)
        return genome_sequence

    def visualize_genome_similarity(self):
        genome_sequence = self.build_genome_sequence()
        num_steps = len(genome_sequence)
        distance_matrix = np.zeros((num_steps, num_steps))

        # Simple Hamming distance between symbolic codes
        for i in range(num_steps):
            for j in range(num_steps):
                dist = sum(c1 != c2 for c1, c2 in zip(genome_sequence[i], genome_sequence[j]))
                distance_matrix[i, j] = dist

        plt.figure(figsize=(8,8))
        plt.imshow(distance_matrix, cmap='magma')
        plt.title("Symbolic Genome Evolution Distance Map")
        plt.xlabel("Memory Step")
        plt.ylabel("Memory Step")
        plt.colorbar(label='Genome Distance')
        plt.show()
