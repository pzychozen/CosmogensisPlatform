import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import json
import os

class MemoryLattice:
    def __init__(self):
        self.memory = []  # Stores centroids and pairs for each iteration
        self.states = []  # Stores state information (optional)
        self.last_centroids = None  # Keeps track of last centroids for reference

    def step_count(self):
        return len(self.memory)

    def store(self, centroids, pairs, iteration):
        # Check if centroids are valid (must be a 2D numpy array with at least two points)
        if (
            centroids is None or not isinstance(centroids, np.ndarray)
            or centroids.ndim != 2 or centroids.shape[0] < 2
        ):
            print(f"[WARNING] Skipping invalid data at iteration {iteration} — centroids: {type(centroids)}, shape: {getattr(centroids, 'shape', None)}")
            return

        # Ensure 'pairs' is a valid iterable (list or set)
        if not isinstance(pairs, (set, list)):
            print(f"[ERROR] Invalid pairs at iteration {iteration}: {type(pairs)}")
            return

        # Check if pairs are valid and not empty
        if len(pairs) == 0:
            print(f"[WARNING] Empty pairs at iteration {iteration}")
            return

        # Store the data in memory
        self.memory.append({
            "iteration": iteration,
            "centroids": centroids,
            "pairs": pairs
        })
        self.last_centroids = centroids

        # Debugging: Print iteration info
        print(f"Iteration {iteration}: Stored {len(centroids)} nodes, {len(pairs)} connections.")

    def compute_motion_vectors(self):
        motions = []
        for i in range(1, len(self.memory)):
            prev = self.memory[i - 1].get("centroids")
            curr = self.memory[i].get("centroids")

            if (
                not isinstance(prev, np.ndarray) or not isinstance(curr, np.ndarray)
                or prev.ndim != 2 or curr.ndim != 2
                or len(prev) != len(curr)
            ):
                continue

            try:
                # Calculate motion vectors using KDTree for nearest neighbor matching
                tree = KDTree(prev)
                dists, indices = tree.query(curr)
                motion = curr - prev[indices]
                motions.append({
                    "step": i,
                    "vectors": motion,
                    "magnitudes": np.linalg.norm(motion, axis=1),
                    "average_magnitude": np.mean(np.linalg.norm(motion, axis=1))
                })
            except Exception as e:
                print(f"[ERROR] Motion vector computation failed at step {i}: {e}")
                continue

        return motions

    def export_fingerprint(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        export_data = []
        for snapshot in self.memory:
            centroids = snapshot['centroids']
            pairs = snapshot['pairs']

            if (
                not isinstance(centroids, np.ndarray) or centroids.ndim != 2
                or centroids.shape[0] == 0 or pairs is None or len(pairs) == 0
            ):
                print(f"[WARNING] Invalid data at iteration {snapshot['iteration']} — Skipping export.")
                continue

            export_data.append({
                'iteration': snapshot['iteration'],
                'nodes': centroids.tolist(),
                'edges': list(map(list, pairs))
            })

        if not export_data:
            print(f"[WARNING] No valid memory entries stored. Skipping export.")
            return

        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Exported {len(export_data)} symbolic fingerprint steps to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to export fingerprint to {filename}: {e}")
            return

        motions = self.compute_motion_vectors()
        print(f"📈 Computed motion vectors for {len(motions)} steps.")
        for motion in motions:
            step = motion['step']
            avg_mag = motion.get('average_magnitude', None)
            if avg_mag is not None:
                print(f"Iteration {self.memory[step]['iteration']} — avg motion magnitude: {avg_mag:.3f}")

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
