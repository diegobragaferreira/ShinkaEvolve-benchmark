# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from deap import base, creator, tools, algorithms
import time


def calculate_c1(sequence):
    """Calculate the autocorrelation constant C₁ for a given sequence."""
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return float('inf')

    # Compute convolution (auto-correlation)
    conv = signal.convolve(sequence, sequence, mode='full')
    # Get the maximum value (excluding the zero padding)
    max_conv = np.max(conv[len(sequence)-1:])

    # Calculate C₁
    sum_sq = sum(x**2 for x in sequence)
    numerator = 2 * len(sequence) * max_conv
    denominator = sum_sq

    if denominator == 0:
        return float('inf')

    return numerator / denominator


def evaluate_individual(individual):
    """Evaluate fitness of an individual (sequence)."""
    # Convert individual to a valid sequence
    sequence = [max(0, min(1000, x)) for x in individual]
    c1 = calculate_c1(sequence)
    # We want to minimize C₁, so we return its inverse as fitness
    if c1 == float('inf'):
        return (0,)  # Invalid solution
    return (1/c1,)


def create_individual(n_steps):
    """Create a random individual with specified number of steps."""
    return [random.uniform(0, 1000) for _ in range(n_steps)]


def mutate_individual(individual, indpb=0.1):
    """Mutate an individual by adding Gaussian noise."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0, 100)
            individual[i] = max(0, individual[i])  # Ensure non-negative
    return individual,


def crossover_individuals(ind1, ind2):
    """Crossover two individuals."""
    cxpoint = random.randint(1, min(len(ind1), len(ind2))-1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


def search_for_best_sequence():
    """Search for the best coefficient sequence using evolutionary algorithm."""
    # Set up the evolutionary algorithm
    random.seed(42)
    np.random.seed(42)

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, n_steps=random.randint(100, 1000))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evolution parameters
    population_size = 50
    generations = 100
    crossover_prob = 0.8
    mutation_prob = 0.2

    # Generate initial population
    pop = toolbox.population(n=population_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Main evolutionary loop
    for gen in range(generations):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new one
        pop[:] = offspring

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")