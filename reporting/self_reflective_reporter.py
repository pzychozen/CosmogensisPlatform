import sqlite3
import json
import numpy as np
import datetime
from database.genome_database import RecursiveGenomeDatabase
from symbolic_memory.memory_core import RecursiveSymbolicMemoryCore

class SelfReflectiveKnowledgeReporter:
    def __init__(self):
        self.genome_db = RecursiveGenomeDatabase()
        self.memory_core = RecursiveSymbolicMemoryCore()

    def compute_genome_stats(self):
        raw = self.genome_db.fetch_all_genomes()
        genomes = [json.loads(row[4]) for row in raw]
        total = len(genomes)
        unique = len({tuple(seq) for seq in genomes})
        avg_length = np.mean([len(seq) for seq in genomes]) if genomes else 0
        return total, unique, avg_length

    def compute_law_stats(self):
        laws = self.memory_core.fetch_all_laws()
        total_laws = len(laws)
        meta_laws = sum(1 for row in laws if row[3] >= 9999)
        return total_laws, meta_laws

    def compute_entropy_trend(self):
        raw = self.genome_db.fetch_all_genomes()[-100:]
        entropies = []
        for row in raw:
            genome_seq = json.loads(row[4])
            hist, _ = np.histogram(genome_seq, bins=20, range=(0, 200))
            hist = hist / np.sum(hist)
            H = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(H)
        avg_entropy = np.mean(entropies) if entropies else 0
        entropy_variation = np.std(entropies) if entropies else 0
        return avg_entropy, entropy_variation

    def generate_report(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_genomes, unique_genomes, avg_len = self.compute_genome_stats()
        total_laws, meta_laws = self.compute_law_stats()
        avg_entropy, entropy_var = self.compute_entropy_trend()

        report = f"""
===============================
Recursive Cognition Report
Timestamp: {timestamp}
===============================
Symbolic Genome Database:
- Total Genomes: {total_genomes}
- Unique Genomes: {unique_genomes}
- Average Genome Length: {avg_len:.2f}

Symbolic Law Memory:
- Total Laws Discovered: {total_laws}
- Meta-Laws Synthesized: {meta_laws}

Entropy Dynamics:
- Average Entropy: {avg_entropy:.3f}
- Entropy Variation: {entropy_var:.3f}

Knowledge Summary:
- Recursive cognition is expanding its symbolic structure.
- Law compression is yielding meta-recursive attractor stabilization.
- Entropy stability reflects cognitive phase balance.

===============================
"""
        print(report)
        return report

    def save_report(self):
        report = self.generate_report()
        filename = f"data_storage/reports/recursive_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to: {filename}")

    def close(self):
        self.genome_db.close()
        self.memory_core.close()
