from core_engine.recursive_universe import UnifiedRecursiveUniverse
from extractors.lattice_extractor import extract_lattice
from extractors.memory_lattice import MemoryLattice
from analyzers.fractal_mind import FractalMind
from analyzers.symbolic_genome import SymbolicGenome
from taxonomy.cosmological_species import CosmologicalSpeciesClassifier
from multiverse.multiverse_forge import MultiverseForge
from multiverse.cosmosystem import CosmologicalOperatingSystem
from evolution.evolutionary_breeder import EvolutionaryBreeder
from ai_optimizer.ai_optimizer import AICosmogenesisOptimizer
from analyzers.mass_spectrum_analyzer import MassSpectrumAnalyzer
from analyzers.harmonic_family_analyzer import HarmonicFamilyAnalyzer
from analyzers.entropy_dynamics_analyzer import EntropyDynamicsAnalyzer
from database.genome_database import RecursiveGenomeDatabase
from analyzers.symbolic_genome import SymbolicGenome


# ================================================================
# Phase 1 — Single Universe Test Run
# ================================================================
print("Running Single Universe Simulation...")
universe = UnifiedRecursiveUniverse(grid_size=128)
for _ in range(1000):
    universe.step()

# Analysis
fractal = FractalMind(universe.memory)
fractal.analyze_fractal_evolution()

genome = SymbolicGenome(universe.memory)
genome_sequence = genome.build_genome_sequence()
genome.visualize_genome_similarity()

species = CosmologicalSpeciesClassifier(genome_sequence)
species.visualize_species(num_species=3)
species.plot_dendrogram()

# ================================================================
# Phase 2 — Persistent Operating System Storage
# ================================================================
cosmo = CosmologicalOperatingSystem()
for _ in range(5):
    cosmo.generate_universe(steps=500)
cosmo.analyze_species_classification()

# ================================================================
# Phase 3 — Multiverse Forge Expansion
# ================================================================
forge = MultiverseForge(
    universe_class=UnifiedRecursiveUniverse,
    genome_class=SymbolicGenome,
    species_class=CosmologicalSpeciesClassifier,
    num_universes=5,
    steps_per_universe=500
)
forge.run_forge()
forge.classify_multiverse()

# ================================================================
# Phase 4 — Evolutionary Breeder
# ================================================================
breeder = EvolutionaryBreeder(cosmo)
breeder.full_breeding_cycle(generations=3, children_per_generation=3, steps_per_universe=500)

# ================================================================
# Phase 5 — AI Optimizer (Experimental)
# ================================================================
ai_opt = AICosmogenesisOptimizer(population_size=5, generations=3)
ai_opt.run()

# ================================================================
# Phase 6 — quantized recursive shell data
# ================================================================
mass_analyzer = MassSpectrumAnalyzer(universe.memory)
spectra = mass_analyzer.analyze_full_memory()
mass_analyzer.visualize_spectrum(spectra)

# ================================================================
# Phase 7 — recursive symbolic resonance extraction
# ================================================================
harmonic_analyzer = HarmonicFamilyAnalyzer(universe.memory)
harmonic_results = harmonic_analyzer.analyze_full_memory()
harmonic_analyzer.visualize_harmonics(harmonic_results)

# ================================================================
# Phase 8 — symbolic stability fields across recursion depth
# ================================================================
entropy_analyzer = EntropyDynamicsAnalyzer(universe.memory)
iterations, entropies = entropy_analyzer.analyze_full_memory()
entropy_analyzer.visualize_entropy(iterations, entropies)

# ================================================================
# Phase 9 — full persistent genome storage
# ================================================================
db = RecursiveGenomeDatabase()

symbolic_genome = SymbolicGenome(universe.memory)
genome_sequence = symbolic_genome.build_genome_sequence()

# Store into database
db.store_genome(
    experiment_name="test_run_01",
    iteration=universe.iteration,
    genome_sequence=genome_sequence,
    parameters=universe.params  # store full parameter config
)
db.close()

# ================================================================
# Phase 10 — symbolic recursive compression analysis
# ================================================================
law_learner = RecursiveLawLearner()

common_patterns = law_learner.discover_laws(chunk_size=4)
law_learner.visualize_law_distribution(common_patterns)

law_learner.close()