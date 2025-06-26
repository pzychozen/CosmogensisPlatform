import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class MassSpectrumAnalyzer:
    def __init__(self, memory_lattice, grid_size=128):
        self.memory = memory_lattice
        self.grid_center = np.array([grid_size // 2, grid_size // 2])
        self.shell_width = 5  # shell thickness

    def analyze_snapshot(self, centroids):
        if len(centroids) == 0:
            return []

        distances = distance.cdist([self.grid_center], centroids)[0]
        max_radius = distances.max()
        num_shells = int(np.ceil(max_radius / self.shell_width))

        spectrum = np.zeros(num_shells, dtype=int)
        for d in distances:
            shell_idx = int(d / self.shell_width)
            spectrum[shell_idx] += 1

        return spectrum

    def analyze_full_memory(self):
        all_spectra = []
        for snapshot in self.memory.memory:
            iteration = snapshot['iteration']
            centroids = snapshot['centroids']
            spectrum = self.analyze_snapshot(centroids)
            all_spectra.append((iteration, spectrum))
            print(f"Iteration {iteration} — Shells: {len(spectrum)}")

        return all_spectra

    def visualize_spectrum(self, all_spectra):
        plt.figure(figsize=(10,6))

        for iteration, spectrum in all_spectra:
            plt.plot(range(len(spectrum)), spectrum, marker='o', label=f"Iter {iteration}")

        plt.xlabel("Shell Index")
        plt.ylabel("Number of Attractors")
        plt.title("Recursive Mass Spectrum Evolution")
        plt.legend()
        plt.grid()
        plt.show()
