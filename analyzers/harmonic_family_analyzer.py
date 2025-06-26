import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class HarmonicFamilyAnalyzer:
    def __init__(self, memory_lattice, grid_size=128, shell_width=5):
        self.memory = memory_lattice
        self.grid_center = np.array([grid_size // 2, grid_size // 2])
        self.shell_width = shell_width

    def analyze_snapshot(self, centroids):
        if len(centroids) == 0:
            return None

        distances = np.linalg.norm(centroids - self.grid_center, axis=1)
        hist, bin_edges = np.histogram(distances, bins=np.arange(0, distances.max() + self.shell_width, self.shell_width))

        peaks, _ = find_peaks(hist)
        harmonic_shells = bin_edges[peaks]
        return harmonic_shells, hist, bin_edges

    def analyze_full_memory(self):
        all_results = []
        for snapshot in self.memory.memory:
            iteration = snapshot['iteration']
            centroids = snapshot['centroids']
            result = self.analyze_snapshot(centroids)
            if result is not None:
                harmonic_shells, hist, bin_edges = result
                print(f"Iteration {iteration} — Harmonic Shells at: {harmonic_shells}")
                all_results.append((iteration, harmonic_shells, hist, bin_edges))
        return all_results

    def visualize_harmonics(self, all_results):
        plt.figure(figsize=(10,6))
        for iteration, shells, hist, bins in all_results:
            plt.plot(bins[:-1], hist, label=f"Iter {iteration}")
            plt.scatter(shells, hist[np.searchsorted(bins[:-1], shells)], color='red')
        plt.xlabel("Radius")
        plt.ylabel("Attractor Count per Shell")
        plt.title("Recursive Harmonic Family Evolution")
        plt.legend()
        plt.grid()
        plt.show()
