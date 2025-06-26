# cosmogenesis_controller.py

from lattice_extractor import extract_lattice
from memory_lattice import MemoryLattice
from fractal_mind import FractalMind
from self_similarity_core import SelfSimilarityCore
from symbolic_genome import SymbolicGenome
from cosmological_species import CosmologicalSpeciesClassifier
from multiverse_forge import MultiverseForge

class CosmogenesisController:
    def run_multiverse(self, num_universes=5, steps_per_universe=100):
        from core_engine.RUG import RecursiveUniverse  # <- Lazy import avoids circular import

        for i in range(num_universes):
            universe = RecursiveUniverse()
            for _ in range(steps_per_universe):
                universe.step()

    def run_single_universe(self, grid_size=128, params=None, steps=500):
        print("Starting single universe simulation...")

        universe = RecursiveUniverse(grid_size=grid_size, params=params)

        for _ in range(steps):
            universe.step()

        print("Universe simulation complete.")

        # Trigger full analysis stack
        self.analyze_universe(universe)

    def analyze_universe(self, universe):
        print("\nRunning Fractal Mind analysis...")
        fractal = FractalMind(universe.memory)
        fractal.analyze_fractal_evolution()

        print("\nRunning Self-Similarity analysis...")
        similarity = SelfSimilarityCore(universe.memory)
        similarity.visualize_similarity_matrix()

        print("\nEncoding Symbolic Genome...")
        genome = SymbolicGenome(universe.memory)
        genome_sequence = genome.build_genome_sequence()

        print("\nClassifying Species (single universe)...")
        classifier = CosmologicalSpeciesClassifier(genome_sequence)
        classifier.visualize_species(num_species=3)
        classifier.plot_dendrogram()

    def run_multiverse(self, num_universes=10, steps_per_universe=500):
        print("\nLaunching Multiverse Forge...")
        forge = MultiverseForge(
            universe_class=RecursiveUniverse,
            genome_class=SymbolicGenome,
            species_class=CosmologicalSpeciesClassifier,
            num_universes=num_universes,
            steps_per_universe=steps_per_universe
        )
        forge.run_forge()
        forge.classify_multiverse()
