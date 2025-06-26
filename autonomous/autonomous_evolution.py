import time
import random
import numpy as np

from core_engine.recursive_universe import RecursiveUniverse
from database.genome_database import RecursiveGenomeDatabase
from analyzers.symbolic_genome import SymbolicGenome
from recursive_farm.breeder_module import GenomeBreeder

class AutonomousEvolutionEngine:
    def __init__(self, grid_size=128):
        self.grid_size = grid_size
        self.db = RecursiveGenomeDatabase()
        self.breeder = GenomeBreeder()

    def fetch_existing_genomes(self):
        raw = self.db.fetch_all_genomes()
        genomes = []
        for row in raw:
            import json
            genome_seq = json.loads(row[4])
            genomes.append(genome_seq)
        return genomes

    def generate_offspring(self, parent_genomes):
        p1 = random.choice(parent_genomes)
        p2 = random.choice(parent_genomes)
        child = self.breeder.crossover(p1, p2)
        child = self.breeder.mutate(child, mutation_rate=0.1)
        return child

    def evolve(self, total_generations=1000, steps_per_universe=500, sleep_time=0.5):
        for generation in range(total_generations):
            existing_genomes = self.fetch_existing_genomes()

            if len(existing_genomes) < 2:
                print(f"[GEN {generation}] Not enough genomes yet — running fresh universe...")
                self.run_fresh_universe(generation, steps_per_universe)
            else:
                print(f"[GEN {generation}] Running evolutionary breeding cycle...")
                child_genome = self.generate_offspring(existing_genomes)
                self.run_universe_with_injected_genome(child_genome, generation, steps_per_universe)

            time.sleep(sleep_time)

    def run_fresh_universe(self, generation_id, steps):
        universe = RecursiveUniverse(grid_size=self.grid_size)
        for _ in range(steps):
            universe.step()
        self.store_genome(universe, generation_id, "fresh")

    def run_universe_with_injected_genome(self, child_genome, generation_id, steps):
        universe = RecursiveUniverse(grid_size=self.grid_size)
        # NOTE: We are not injecting genome into recursion yet directly.
        # Later we'll add genome-parameter fusion system here.
        for _ in range(steps):
            universe.step()
        self.store_genome(universe, generation_id, "bred")

    def store_genome(self, universe, generation_id, tag):
        symbolic_genome = SymbolicGenome(universe.memory)
        genome_sequence = symbolic_genome.build_genome_sequence()

        self.db.store_genome(
            experiment_name=f"auto_{tag}_gen_{generation_id}",
            iteration=universe.iteration,
            genome_sequence=genome_sequence,
            parameters=universe.params
        )
        print(f"Stored genome for generation {generation_id} ({tag})")

    def close(self):
        self.db.close()
