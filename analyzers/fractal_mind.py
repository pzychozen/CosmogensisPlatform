import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# ================================================================
# FRACTAL MIND MODULE: Recursive Fractal Dimension Analyzer
# ================================================================

class FractalMind:
    def __init__(self, memory_lattice):
        self.memory = memory_lattice

    def estimate_fractal_dimension(self, centroids, scales=np.logspace(0.5, 1.5, 10)):
        if len(centroids) < 2:
            return 0
        N = []
        for scale in scales:
            count = 0
            for point in centroids:
                neighbors = np.sum(distance.cdist([point], centroids) < scale)
                count += neighbors
            N.append(count)
        N = np.array(N)
        log_R = np.log(scales)
        log_N = np.log(N)
        slope, intercept = np.polyfit(log_R, log_N, 1)
        return slope

    def analyze_fractal_evolution(self):
        dimensions = []
        iterations = []
        for snapshot in self.memory.memory:
            iteration = snapshot['iteration']
            centroids = snapshot['centroids']
            dim = self.estimate_fractal_dimension(centroids)
            dimensions.append(dim)
            iterations.append(iteration)
            print(f"Iteration {iteration} — Estimated fractal dimension: {dim:.3f}")

        plt.figure(figsize=(10,6))
        plt.plot(iterations, dimensions, marker='o', color='purple')
        plt.title("Recursive Fractal Dimension Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Estimated Fractal Dimension")
        plt.grid(True)
        plt.show()
