import random
import sqlite3
import json
from database.genome_database import RecursiveGenomeDatabase
from recursive_farm.breeder_module import GenomeBreeder
from recursive_farm.universe_worker import run_universe_run
from recursive_farm.farm_manager import RecursiveEvolutionaryFarm

class RecursiveEvolutionaryFarm:
    def __init__(self):
        self.db = RecursiveGenomeDatabase()
        self.breeder = GenomeBreeder()

    def fetch_existing_genomes(self):
        raw = self.db.fetch_all_genomes()
        genomes = []
        for row in raw:
            genome_seq = json.loads(row[4])
            genomes.append(genome_seq)
        return genomes

    def generate_offspring(self, parent_genomes):
        p1 = random.choice(parent_genomes)
        p2 = random.choice(parent_genomes)
        child = self.breeder.crossover(p1, p2)
        child = self.breeder.mutate(child)
        return child

    def run_farm_generation(self, population_size=10, steps_per_universe=500):
        existing_genomes = self.fetch_existing_genomes()

        if len(existing_genomes) < 2:
            print("Not enough genomes yet. Running fresh universes...")
            for run_id in range(population_size):
                run_universe_run(farm_run_id=run_id, steps=steps_per_universe)
            return

        print("Breeding new generation...")
        for run_id in range(population_size):
            child_genome = self.generate_offspring(existing_genomes)
            # TODO: allow injecting child genome into new universe parameters
            # For now: run fresh universe (full integration comes next)
            run_universe_run(farm_run_id=f"gen_{run_id}", steps=steps_per_universe)

        print("Generation complete.")

    def close(self):
        self.db.close()


if __name__ == "__main__":
    farm = RecursiveEvolutionaryFarm()
    farm.run_farm_generation(population_size=10, steps_per_universe=500)
    farm.close()