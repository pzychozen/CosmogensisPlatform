import sqlite3
import numpy as np
import json
from collections import Counter
from database.genome_database import RecursiveGenomeDatabase
from law_discovery.recursive_law_learner import RecursiveLawLearner

class RecursiveAdaptiveOptimizer:
    def __init__(self, db_path="data_storage/genome_database.db"):
        self.db_path = db_path
        self.db = RecursiveGenomeDatabase(db_path)

    def compute_genome_diversity(self):
        raw = self.db.fetch_all_genomes()
        all_sequences = [json.loads(row[4]) for row in raw]
        unique_sequences = {tuple(seq) for seq in all_sequences}
        diversity_ratio = len(unique_sequences) / max(len(all_sequences), 1)
        return diversity_ratio

    def compute_law_compression_score(self, chunk_size=4):
        learner = RecursiveLawLearner(self.db_path)
        common_patterns = learner.discover_laws(chunk_size)
        total_counts = sum([count for _, count in common_patterns])
        compression_score = len(common_patterns) / max(total_counts, 1)
        learner.close()
        return compression_score

    def compute_entropy_stability(self, recent_window=100):
        raw = self.db.fetch_all_genomes()[-recent_window:]
        entropies = []
        for row in raw:
            genome_seq = json.loads(row[4])
            hist, _ = np.histogram(genome_seq, bins=20, range=(0, 200))
            hist = hist / np.sum(hist)
            H = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(H)
        if len(entropies) == 0:
            return 0
        entropy_stability = np.std(entropies)
        return entropy_stability

    def propose_parameter_adjustments(self):
        diversity = self.compute_genome_diversity()
        compression = self.compute_law_compression_score()
        entropy_stability = self.compute_entropy_stability()

        print("\n--- Recursive Optimization Diagnostics ---")
        print(f"Genome Diversity Ratio: {diversity:.3f}")
        print(f"Law Compression Score: {compression:.3f}")
        print(f"Entropy Stability: {entropy_stability:.3f}")

        adjustments = {}

        if diversity < 0.5:
            adjustments['mutation_rate'] = +0.05  # encourage diversity
        if compression > 0.1:
            adjustments['fusion_strength'] = -0.1  # encourage stabilization
        if entropy_stability > 0.5:
            adjustments['time_feedback'] = -0.005  # dampen oscillation

        print("Proposed Adjustments:", adjustments)
        return adjustments

    def close(self):
        self.db.close()
