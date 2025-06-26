import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import json
import os

class MemoryLattice:
    def __init__(self):
        self.memory = []
        self.last_centroids = None

    def store(self, centroids, pairs, iteration):
        if (
            centroids is None or not isinstance(centroids, np.ndarray)
            or centroids.ndim != 2 or centroids.shape[0] < 2
        ):
            print(f"[WARNING] Skipping invalid data at iteration {iteration} — centroids: {type(centroids)}, shape: {getattr(centroids, 'shape', None)}")
            return

        self.memory.append({
            "iteration": iteration,
            "centroids": centroids,
            "pairs": pairs
        })
        self.last_centroids = centroids

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
                continue

            export_data.append({
                'iteration': snapshot['iteration'],
                'nodes': centroids.tolist(),
                'edges': list(map(list, pairs))
            })

        if not export_data:
            print(f"[WARNING] No valid memory entries stored. Skipping export.")
            return

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Exported {len(export_data)} symbolic fingerprint steps to {filename}")

        motions = self.compute_motion_vectors()
        print(f"📈 Computed motion vectors for {len(motions)} steps.")
        for motion in motions:
            step = motion['step']
            avg_mag = motion.get('average_magnitude', None)
            if avg_mag is not None:
                print(f"Iteration {self.memory[step]['iteration']} — avg motion magnitude: {avg_mag:.3f}")
