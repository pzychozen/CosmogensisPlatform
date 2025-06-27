import numpy as np
from core_engine.recursive_universe import UnifiedRecursiveUniverse
from extractors.lattice_extractor import extract_lattice
from analyzers.symbolic_genome import SymbolicGenome
from database.genome_database import RecursiveGenomeDatabase

def run_universe_run(farm_run_id, steps=500):
    universe = UnifiedRecursiveUniverse(grid_size=128)
    db = RecursiveGenomeDatabase()

    for _ in range(steps):
        universe.step()

    centroids, pairs = extract_lattice(universe.T)
    symbolic_genome = SymbolicGenome(universe.memory)
    genome_sequence = symbolic_genome.build_genome_sequence()

    db.store_genome(
        experiment_name=f"farm_{farm_run_id}",
        iteration=universe.iteration,
        genome_sequence=genome_sequence,
        parameters=universe.params
    )
    db.close()

    return genome_sequence
