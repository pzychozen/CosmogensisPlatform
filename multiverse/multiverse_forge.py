import numpy as np
import random

# ================================================================
# MULTIVERSE FORGE: Randomized Recursive Universe Factory
# ================================================================

class MultiverseForge:
    def __init__(self, universe_class, genome_class, species_class, num_universes=10, steps_per_universe=500):
        self.universe_class = universe_class
        self.genome_class = genome_class
        self.species_class = species_class
        self.num_universes = num_universes
        self.steps_per_universe = steps_per_universe
        self.genome_bank = []

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

    def run_forge(self):
        for n in range(self.num_universes):
            print(f"\nGenerating Universe {n+1}...")
            params = self.random_params()
            universe = self.universe_class(grid_size=128, params=params)

            for _ in range(self.steps_per_universe):
                universe.step()

            genome_module = self.genome_class(universe.memory)
            genome_sequence = genome_module.build_genome_sequence()
            self.genome_bank.append(genome_sequence)

        print("\nAll universes generated.")

    def classify_multiverse(self):
        # Flatten genomes into one big sequence list
        all_genomes = [gen for universe in self.genome_bank for gen in universe]

        species_classifier = self.species_class(all_genomes)
        species_classifier.visualize_species(num_species=5)
        species_classifier.plot_dendrogram()
