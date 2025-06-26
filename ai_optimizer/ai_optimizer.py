import random
import copy

from core_engine.recursive_universe import RecursiveUniverse
from analyzers.fractal_mind import FractalMind
from analyzers.symbolic_genome import SymbolicGenome

class AICosmogenesisOptimizer:
    def __init__(self, population_size=10, generations=5):
        self.population_size = population_size
        self.generations = generations
        self.population = []

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

    def initial_population(self):
        for _ in range(self.population_size):
            self.population.append(self.random_params())

    def mutate(self, params):
        child = copy.deepcopy(params)
        for key in child:
            if random.random() < 0.3:
                mutation = 1 + random.uniform(-0.1, 0.1)
                child[key] *= mutation
        return child

    def compute_fitness(self, memory_lattice):
        fractal = FractalMind(memory_lattice)
        dimension = fractal.estimate_fractal_dimension(memory_lattice.memory[-1]['centroids'])
        genome = SymbolicGenome(memory_lattice)
        genome_sequence = genome.build_genome_sequence()
        unique_symbols = len(set("".join(genome_sequence)))
        last_snapshot = memory_lattice.memory[-1]
        num_attractors = len(last_snapshot['centroids'])
        fitness = (dimension * 2.0) + (unique_symbols * 1.5) + (num_attractors * 0.5)
        return fitness

    def evaluate_fitness(self, params):
        universe = RecursiveUniverse(grid_size=128, params=params)
        for _ in range(500):
            universe.step()
        fitness = self.compute_fitness(universe.memory)
        return fitness

    def run(self):
        self.initial_population()

        for generation in range(self.generations):
            print(f"\n=== AI Optimization Generation {generation+1} ===")
            scored_population = []
            for params in self.population:
                fitness = self.evaluate_fitness(params)
                scored_population.append((fitness, params))
                print(f"Fitness: {fitness:.3f}")

            scored_population.sort(reverse=True, key=lambda x: x[0])
            self.population = [p for f, p in scored_population[:self.population_size//2]]

            children = []
            for parent in self.population:
                child = self.mutate(parent)
                children.append(child)

            self.population += children

        print("\nOptimization Complete. Best parameters found:")
        best_params = self.population[0]
        print(best_params)
        return best_params
