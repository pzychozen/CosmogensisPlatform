import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Memory Lattice Storage System (Modular Future-Proof Version)
# ================================================================

class MemoryLattice:
    def __init__(self):
        self.memory = []

    def store(self, iteration, centroids, pairs):
        if centroids is None or pairs is None:
            print(f"Iteration {iteration}: No attractors found.")
            return
        
        self.memory.append({
            'iteration': iteration,
            'centroids': centroids,
            'pairs': pairs
        })
        print(f"Iteration {iteration}: Stored {len(centroids)} nodes, {len(pairs)} connections.")

    def visualize_step(self, step, grid_size=128):
        if step >= len(self.memory):
            print("Step out of range.")
            return
        
        snapshot = self.memory[step]
        centroids = snapshot['centroids']
        pairs = snapshot['pairs']

        plt.figure(figsize=(8,8))
        plt.imshow(np.zeros((grid_size, grid_size)), cmap='Greys', alpha=0)
        plt.scatter(centroids[:,1], centroids[:,0], color='cyan', s=40)
        
        for i, j in pairs:
            p1 = centroids[i]
            p2 = centroids[j]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], color='lime', linewidth=1)
        
        plt.title(f"Memory Lattice — Iteration {snapshot['iteration']}")
        plt.show()

    def animate(self, grid_size=128, pause_time=0.1):
        for step, snapshot in enumerate(self.memory):
            centroids = snapshot['centroids']
            pairs = snapshot['pairs']

            plt.figure(figsize=(8,8))
            plt.imshow(np.zeros((grid_size, grid_size)), cmap='Greys', alpha=0)
            plt.scatter(centroids[:,1], centroids[:,0], color='cyan', s=40)

            for i, j in pairs:
                p1 = centroids[i]
                p2 = centroids[j]
                plt.plot([p1[1], p2[1]], [p1[0], p2[0]], color='lime', linewidth=1)
            
            plt.title(f"Memory Lattice Evolution — Step {step} — Iteration {snapshot['iteration']}")
            plt.pause(pause_time)
            plt.close()
