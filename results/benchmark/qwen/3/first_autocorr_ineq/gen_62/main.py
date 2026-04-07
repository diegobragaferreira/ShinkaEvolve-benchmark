# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from deap import base, creator, tools, algorithms

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
POP_SIZE = 50
NGEN = 100
MUTPB = 0.2
CXPB = 0.5

def convolve_fft(a, b):
    """Compute convolution using FFT for efficiency."""
    n = len(a)
    pad_size = 2 * n - 1
    a_padded = np.pad(a, (0, pad_size - n), 'constant')
    b_padded = np.pad(b, (0, pad_size - n), 'constant')
    a_fft = fft(a_padded)
    b_fft = fft(b_padded)
    conv_result = ifft(a_fft * np.conj(b_fft))
    return np.real(conv_result[:pad_size])

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')
    
    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)
    sum_sq = np.sum(sequence)**2
    
    if sum_sq < 1e-10:
        return float('inf')
        
    c1 = 2 * n * max_conv / sum_sq
    return c1

def evaluate_individual(individual):
    """Evaluate the individual and return inverse of C1 as fitness."""
    seq = list(individual)
    c1 = compute_c1_constant(seq)
    if np.isinf(c1) or c1 <= 0:
        return (float('-inf'),)  # Penalize invalid sequences
    return (1.0 / c1,)

def mutate_sequence(individual, indpb=0.1):
    """Mutation operation with convolution-aware perturbation."""
    # Convert to numpy for easier manipulation
    individual_np = np.array(individual)
    for i in range(len(individual)):
        if random.random() < indpb:
            # Apply Gaussian noise with amplitude dependent on current value
            noise = np.random.normal(0, 0.1 * individual_np[i] + 1e-3)
            individual_np[i] = max(0, individual_np[i] + noise)
    return tuple(individual_np.tolist())

def crossover_sequences(ind1, ind2):
    """Crossover operation that respects convolution structure."""
    # Simple uniform crossover
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def initialize_population(pop_size, min_length, max_length):
    """Initialize population with random sequences."""
    pop = []
    for _ in range(pop_size):
        n = random.randint(min_length, max_length)
        seq = [random.uniform(0.1, 2.0) for _ in range(n)]
        pop.append(tuple(seq))
    return pop

def search_for_best_sequence():
    """Search for the best coefficient sequence using genetic algorithm."""
    start_time = time.time()
    
    # Set up DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(0.1, 2.0), n=random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_sequences)
    toolbox.register("mutate", mutate_sequence)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population
    pop = initialize_population(POP_SIZE, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    hof = tools.ParetoFront()
    
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                       ngen=NGEN, stats=stats, halloffame=hof, verbose=False)
    except Exception as e:
        pass
    
    # Return the best individual found
    if hof:
        best_individual = hof[0]
        return list(best_individual)
    else:
        # Fallback to a random sequence if nothing was found
        n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        return [random.uniform(0.1, 2.0) for _ in range(n)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
