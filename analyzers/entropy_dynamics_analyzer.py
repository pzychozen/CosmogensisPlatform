import numpy as np
import matplotlib.pyplot as plt

class EntropyDynamicsAnalyzer:
    def __init__(self, memory_lattice, grid_size=128, bins=50):
        self.memory = memory_lattice
        self.grid_size = grid_size
        self.bins = bins

    def compute_entropy(self, centroids):
        if len(centroids) == 0:
            return 0

        # Build 2D histogram of attractor positions
        H, xedges, yedges = np.histogram2d(
            centroids[:,0], centroids[:,1], bins=self.bins, 
            range=[[0, self.grid_size], [0, self.grid_size]]
        )
        H = H / H.sum()  # normalize to probability

        # Compute Shannon entropy (ignoring zero bins)
        H_nonzero = H[H > 0]
        entropy = -np.sum(H_nonzero * np.log2(H_nonzero))
        return entropy

    def analyze_full_memory(self):
        entropies = []
        iterations = []
        for snapshot in self.memory.memory:
            iteration = snapshot['iteration']
            centroids = snapshot['centroids']
            entropy = self.compute_entropy(centroids)
            entropies.append(entropy)
            iterations.append(iteration)
            print(f"Iteration {iteration} — Entropy: {entropy:.4f}")
        return iterations, entropies

    def visualize_entropy(self, iterations, entropies):
        plt.figure(figsize=(10,6))
        plt.plot(iterations, entropies, marker='o', color='orange')
        plt.xlabel("Iteration")
        plt.ylabel("Shannon Entropy (bits)")
        plt.title("Recursive Entropy Dynamics Evolution")
        plt.grid()
        plt.show()
