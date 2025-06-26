import sqlite3
import json
import os
import itertools
import numpy as np
from symbolic_memory.memory_core import RecursiveSymbolicMemoryCore

class MetaSymbolicKnowledgeCompressor:
    def __init__(self, memory_db_path="data_storage/symbolic_memory.db"):
        self.memory = RecursiveSymbolicMemoryCore(memory_db_path)

    def fetch_all_laws(self):
        raw = self.memory.fetch_all_laws()
        law_patterns = []
        for row in raw:
            pattern = json.loads(row[2])
            law_patterns.append(pattern)
        return law_patterns

    def compute_similarity(self, pattern1, pattern2):
        """Very simple similarity score based on shared tokens"""
        set1, set2 = set(pattern1), set(pattern2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0
        return len(intersection) / len(union)

    def cluster_laws(self, similarity_threshold=0.5):
        laws = self.fetch_all_laws()
        clusters = []

        for law in laws:
            found_cluster = False
            for cluster in clusters:
                if any(self.compute_similarity(law, existing) >= similarity_threshold for existing in cluster):
                    cluster.append(law)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([law])
        return clusters

    def synthesize_meta_laws(self, clusters):
        meta_laws = []
        for cluster in clusters:
            flattened = list(itertools.chain(*cluster))
            merged = list(set(flattened))
            merged.sort()
            meta_laws.append(merged)
        return meta_laws

    def compress(self, similarity_threshold=0.5):
        print("\n=== Meta-Symbolic Knowledge Compression ===")
        clusters = self.cluster_laws(similarity_threshold)
        meta_laws = self.synthesize_meta_laws(clusters)
        print(f"Discovered {len(meta_laws)} compressed meta-laws from {len(clusters)} clusters.")
        for idx, meta in enumerate(meta_laws):
            print(f"Meta-Law {idx+1}: {meta}")
        return meta_laws

    def store_meta_laws(self, meta_laws):
        for meta in meta_laws:
            self.memory.store_law(
                law_pattern=meta,
                source_genomes=9999,  # Marked as meta-layer
                confidence=2.0  # Higher confidence level for synthesized laws
            )

    def close(self):
        self.memory.close()
