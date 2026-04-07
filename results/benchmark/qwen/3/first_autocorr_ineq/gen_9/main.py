# EVOLVE-BLOCK-START

import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy import signal
import time

# Global constants for optimization
MAX_SEQUENCE_LENGTH = 1000
MIN_SEQUENCE_LENGTH = 50
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
BENCHMARK_THRESHOLD = 1.5031

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def calculate_c1(sequence):
    """
    Calculate C1 for a given sequence.
    C1 = 2n * max(convolution) / (sum(sequence))^2
    """
    if len(sequence) == 0:
        return float('inf')

    sequence = np.array(sequence)
    total_sum = np.sum(sequence)

    if total_sum < 0.01:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv_result = signal.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv_result)

    n = len(sequence)
    c1 = (2 * n * max_conv) / (total_sum ** 2)
    return c1

def evaluate_individual(individual):
    """
    Evaluate fitness of an individual (sequence).
    Returns (1/C1,) - we maximize 1/C1 to minimize C1.
    """
    c1_value = calculate_c1(individual)
    if c1_value == float('inf'):
        return (0,)  # Invalid solution
    return (1.0 / c1_value,)

def create_individual():
    """Create a random individual (sequence)"""
    length = random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
    # Create sequence with values in [0, 1000]
    individual = [random.uniform(0, 1000) for _ in range(length)]
    return individual

def mutate_individual(individual):
    """Mutate an individual"""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Apply small perturbation
            individual[i] *= random.uniform(0.9, 1.1)
            individual[i] = max(0, min(1000, individual[i]))
    return individual,

def crossover_individuals(ind1, ind2):
    """Crossover two individuals"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def search_for_best_sequence():
    """Main optimization function using evolutionary algorithm"""

    # Setup DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution
    start_time = time.time()
    try:
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                           ngen=GENERATIONS, stats=stats, verbose=False)
    except Exception as e:
        print(f"Evolution error: {e}")

    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

def get_performance_metrics(sequence):
    """Calculate performance metrics for the solution"""
    c1 = calculate_c1(sequence)
    inv_c1 = 1.0 / c1 if c1 != float('inf') else 0
    benchmark_ratio = c1 / BENCHMARK_THRESHOLD if c1 != float('inf') else 0
    eval_time = 0  # Placeholder - actual timing would be more complex

    return {
        'inv_c1': inv_c1,
        'benchmark_ratio': benchmark_ratio,
        'eval_time': eval_time,
        'c1': c1
    }

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")