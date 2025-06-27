# evolutionary_breeder.py

import numpy as np
import random
import copy

from core_engine.recursive_universe import UnifiedRecursiveUniverse
from symbolic_genome import SymbolicGenome
from cosmological_species import CosmologicalSpeciesClassifier

# ================================================================
# EVOLUTIONARY BREEDER CORE
# ================================================================

class EvolutionaryBreeder:
    def __init__(self, cosmo_system):
        self.cosmo = cosmo_system  # Access to COS persistent universe bank

    def select_parents(self, num_parents=2):
        return random.sample(list(self.cosmo.universe_records.items()), num_parents)

    def crossover_params(self, parent_params):
        keys = parent_params[0].keys()
        child_params = {}
        for key in keys:
            values = [p[key] for p in parent_params]
            child_params[key] = random.choice(values)  # Simple gene mixing
            # Introduce small mutation
            mutation = 1 + random.uniform(-0.05, 0.05)
            child_params[key] *= mutation
        return child_params

    def breed_universe(self, steps=500):
        # Select parents from existing universe pool
        selected = self.select_parents()
        parent_ids, parent_params = zip(*selected)

        print(f"\nBreeding from parents: {parent_ids}")

        # Extract parent parameter dicts
        parent_param_dicts = []
        for pid in parent_ids:
            parent_param_dicts.append(self.cosmo.universe_records[pid])

        # Perform crossover + mutation
        child_params = self.crossover_params(parent_param_dicts)

        # Generate child universe
        self.cosmo.generate_universe(steps=steps, params=child_params)

    def full_breeding_cycle(self, generations=5, children_per_generation=5, steps_per_universe=500):
        for gen in range(generations):
            print(f"\n=== GENERATION {gen+1} ===")
            for _ in range(children_per_generation):
                self.breed_universe(steps=steps_per_universe)

        # After evolution — analyze full species pool
        self.cosmo.analyze_species_classification()
