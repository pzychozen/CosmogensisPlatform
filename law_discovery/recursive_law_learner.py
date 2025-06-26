import sqlite3
import json
from collections import Counter
import re
import matplotlib.pyplot as plt

class RecursiveLawLearner:
    def __init__(self, db_path="data_storage/genome_database.db"):
        self.conn = sqlite3.connect(db_path)

    def fetch_genomes(self):
        query = "SELECT genome FROM genomes"
        cursor = self.conn.execute(query)
        genomes = []
        for row in cursor:
            genome_seq = json.loads(row[0])
            genomes.append(genome_seq)
        return genomes

    def compress_sequence(self, sequence, chunk_size=4):
        """Break sequence into chunks"""
        chunks = []
        for i in range(0, len(sequence), chunk_size):
            chunk = tuple(sequence[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def discover_laws(self, chunk_size=4):
        genomes = self.fetch_genomes()
        chunk_counter = Counter()

        for genome in genomes:
            chunks = self.compress_sequence(genome, chunk_size=chunk_size)
            chunk_counter.update(chunks)

        # Filter for most common patterns
        most_common = chunk_counter.most_common(20)
        print("\nTop Recurring Symbolic Genome Patterns:")
        for pattern, count in most_common:
            print(f"{pattern} — {count} occurrences")

        return most_common

    def visualize_law_distribution(self, common_patterns):
        counts = [count for _, count in common_patterns]
        labels = [str(pattern) for pattern, _ in common_patterns]

        plt.figure(figsize=(10,6))
        plt.barh(labels, counts, color='green')
        plt.xlabel("Occurrences")
        plt.title("Symbolic Law Pattern Frequencies")
        plt.tight_layout()
        plt.show()

    def close(self):
        self.conn.close()