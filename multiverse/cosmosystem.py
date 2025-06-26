# cosmosystem.py

import os
import pickle
import random

from core_engine.recursive_universe import RecursiveUniverse
from memory_lattice import MemoryLattice
from lattice_extractor import extract_lattice
from symbolic_genome import SymbolicGenome
from cosmological_species import CosmologicalSpeciesClassifier

# ================================================================
# COSMOLOGICAL OPERATING SYSTEM CORE
# ================================================================

class CosmologicalOperatingSystem:
    def __init__(self, storage_dir="cosmo_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.universe_counter = 0
        self.universe_records = {}  # universe_id: params

    def random_params(self):
        return {
            'alpha': random.uniform(0.5, 1.5),
            'beta': random.uniform(0.1, 0.3),
            'gamma': random.uniform(0.01, 0.1),
            'delta': random.uniform(0.005, 0.02),
            'eta': random.uniform(0.5, 1.2),
            'epsilon': random.uniform(0.001, 0.005),
            'lambda_amp': random.uniform(0.1, 0.5),
            'alpha_phase': random.uniform(0.5, 1.5),
            'time_coupling': random.uniform(0.002, 0.01),
            'time_feedback': random.uniform(0.005, 0.02),
            'fusion_strength': random.uniform(0.1, 0.5),
            'tunneling_rate': random.uniform(0.001, 0.01),
            'Q_max': 100
        }

    def generate_universe(self, steps=500, params=None):
        universe_id = f"universe_{self.universe_counter}"
        if params is None:
            params = self.random_params()

        print(f"Generating {universe_id}...")
        universe = RecursiveUniverse(grid_size=128, params=params)

        for _ in range(steps):
            universe.step()

        # Save state
        self.save_universe(universe_id, universe)
        self.universe_records[universe_id] = params
        self.universe_counter += 1

    def save_universe(self, universe_id, universe_obj):
        path = os.path.join(self.storage_dir, f"{universe_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(universe_obj.memory, f)
        print(f"Universe {universe_id} saved.")

    def load_universe(self, universe_id):
        path = os.path.join(self.storage_dir, f"{universe_id}.pkl")
        with open(path, "rb") as f:
            memory = pickle.load(f)
        return memory

    def analyze_species_classification(self):
        # Gather all stored universes
        all_genomes = []
        for universe_id in self.universe_records:
            memory = self.load_universe(universe_id)
            genome_module = SymbolicGenome(memory)
            genome_sequence = genome_module.build_genome_sequence()
            all_genomes.extend(genome_sequence)

        classifier = CosmologicalSpeciesClassifier(all_genomes)
        classifier.visualize_species(num_species=5)
        classifier.plot_dendrogram()
