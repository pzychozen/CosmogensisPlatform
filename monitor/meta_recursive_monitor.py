import time
import threading
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from database.genome_database import RecursiveGenomeDatabase
from law_discovery.recursive_law_learner import RecursiveLawLearner

class MetaRecursiveSelfMonitor:
    def __init__(self, refresh_interval=10):
        self.db = RecursiveGenomeDatabase()
        self.learner = RecursiveLawLearner()
        self.refresh_interval = refresh_interval
        self.running = True

    def compute_metrics(self):
        # Genome diversity
        raw = self.db.fetch_all_genomes()
        all_sequences = [json.loads(row[4]) for row in raw]
        unique_sequences = {tuple(seq) for seq in all_sequences}
        diversity_ratio = len(unique_sequences) / max(len(all_sequences), 1)

        # Law compression
        common_patterns = self.learner.discover_laws(chunk_size=4)
        compression_score = len(common_patterns) / max(sum([c for _, c in common_patterns]), 1)

        # Entropy stabilization
        entropies = []
        for genome in all_sequences[-100:]:
            hist, _ = np.histogram(genome, bins=20, range=(0, 200))
            hist = hist / np.sum(hist)
            H = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(H)
        entropy_stability = np.std(entropies) if entropies else 0

        return diversity_ratio, compression_score, entropy_stability

    def start_monitor(self):
        threading.Thread(target=self.run_loop).start()

    def run_loop(self):
        diversity_data, compression_data, entropy_data, time_steps = [], [], [], []

        while self.running:
            diversity, compression, entropy = self.compute_metrics()
            t = len(time_steps) + 1
            time_steps.append(t)
            diversity_data.append(diversity)
            compression_data.append(compression)
            entropy_data.append(entropy)

            print(f"[t={t}] Diversity: {diversity:.3f} | Compression: {compression:.3f} | Entropy: {entropy:.3f}")

            fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

            fig.add_trace(go.Scatter(x=time_steps, y=diversity_data, mode='lines+markers', name='Diversity'), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_steps, y=compression_data, mode='lines+markers', name='Compression'), row=2, col=1)
            fig.add_trace(go.Scatter(x=time_steps, y=entropy_data, mode='lines+markers', name='Entropy Stability'), row=3, col=1)

            fig.update_layout(height=800, title_text="Meta-Recursive Self-Reflection Monitor", template='plotly_dark')
            fig.show()

            time.sleep(self.refresh_interval)

    def stop(self):
        self.running = False
        self.db.close()
        self.learner.close()
