from core_engine.recursive_universe import RecursiveUniverse
from database.genome_database import RecursiveGenomeDatabase
from analyzers.symbolic_genome import SymbolicGenome
from recursive_farm.farm_manager import RecursiveEvolutionaryFarm
from law_discovery.recursive_law_learner import RecursiveLawLearner
from dashboard.recursive_dashboard import RecursiveDashboard

class IntegratedRecursiveControl:
    def __init__(self):
        self.db = RecursiveGenomeDatabase()
        self.farm = RecursiveEvolutionaryFarm()
        self.learner = RecursiveLawLearner()
        # Note: dashboard remains separate for now (it’s event-loop driven)
        self.menu()

    def menu(self):
        while True:
            print("\n=== INTEGRATED RECURSIVE COGNITION SYSTEM ===")
            print("1️⃣  Run Single Universe Simulation")
            print("2️⃣  Run Evolutionary Farm Generation")
            print("3️⃣  Extract Symbolic Genomes Now")
            print("4️⃣  Run Recursive Law Learner")
            print("5️⃣  Display Genome Database Stats")
            print("6️⃣  Exit")

            choice = input("Select option: ")

            if choice == "1":
                self.run_universe()
            elif choice == "2":
                self.run_farm()
            elif choice == "3":
                self.extract_genomes()
            elif choice == "4":
                self.run_law_learning()
            elif choice == "5":
                self.show_stats()
            elif choice == "6":
                print("Shutting down...")
                break
            else:
                print("Invalid option.")

    def run_universe(self, steps=500):
        print("Running Universe Simulation...")
        universe = RecursiveUniverse(grid_size=128)
        for _ in range(steps):
            universe.step()
        centroids, pairs = universe.memory.memory[-1]['centroids'], universe.memory.memory[-1]['pairs']

        symbolic_genome = SymbolicGenome(universe.memory)
        genome_sequence = symbolic_genome.build_genome_sequence()

        self.db.store_genome(
            experiment_name="integrated_run",
            iteration=universe.iteration,
            genome_sequence=genome_sequence,
            parameters=universe.params
        )
        print("Universe run complete and genome stored.")

    def run_farm(self):
        print("Running farm cycle...")
        self.farm.run_farm_generation(population_size=10, steps_per_universe=500)
        print("Farm generation complete.")

    def extract_genomes(self):
        genomes = self.db.fetch_all_genomes()
        print(f"Total genomes stored: {len(genomes)}")
        if len(genomes) > 0:
            print("Last genome example:")
            import json
            print(json.loads(genomes[-1][4]))

    def run_law_learning(self):
        print("Extracting recursive symbolic laws...")
        patterns = self.learner.discover_laws(chunk_size=4)
        self.learner.visualize_law_distribution(patterns)

    def show_stats(self):
        genomes = self.db.fetch_all_genomes()
        print(f"Total genomes stored: {len(genomes)}")

    def close(self):
        self.db.close()
        self.learner.close()