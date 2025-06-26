import random
import numpy as np

class GenomeBreeder:
    def __init__(self, genome_type='symbolic', mutation_strategy=None):
        self.genome_type = genome_type
        self.mutation_strategy = mutation_strategy or self.default_mutation

    def crossover(self, genome1, genome2, crossover_rate=0.5):
        length = min(len(genome1), len(genome2))
        child = []
        for i in range(length):
            if random.random() < crossover_rate:
                child.append(genome1[i])
            else:
                child.append(genome2[i])
        return child

    def mutate(self, genome, mutation_rate=0.1):
        return self.mutation_strategy(genome, mutation_rate)

    def default_mutation(self, genome, mutation_rate):
        mutated = []
        for gene in genome:
            if isinstance(gene, str):
                if random.random() < mutation_rate:
                    mutated.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
                else:
                    mutated.append(gene)
            else:
                if random.random() < mutation_rate:
                    mutated.append(gene + np.random.randint(-2, 3))
                else:
                    mutated.append(gene)
        return mutated
