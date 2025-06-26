from .memory_lattice import MemoryLattice


class LatticeManager:
    def __init__(self, extractor, memory_lattice=None):
        self.extractor = extractor
        self.memory = memory_lattice or MemoryLattice()

    def process(self, T, iteration):
        centroids, pairs = self.extractor(T)

        if isinstance(centroids, (list, tuple, set)):
            centroids = list(centroids)
        if not hasattr(centroids, "__len__") or len(centroids) == 0:
            print(f"[WARNING] No valid centroids detected at iteration {iteration}. Skipping.")
            return None, None

        if pairs is None or not hasattr(pairs, "__iter__"):
            print(f"[WARNING] Invalid or empty connection pairs at iteration {iteration}. Skipping.")
            return None, None

        try:
            self.memory.store(centroids, pairs, iteration)
        except Exception as e:
            print(f"[ERROR] Failed to store lattice data at iteration {iteration}: {e}")

        return centroids, pairs

    def node_count(self):
        if self.memory.last_centroids is not None:
            return len(self.memory.last_centroids)
        return 0