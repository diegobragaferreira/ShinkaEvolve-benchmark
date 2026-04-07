# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
from deap import base, creator, tools, algorithms
import random
import time
from functools import partial
import warnings
import math

warnings.filterwarnings('ignore')

def convolve_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Use scipy's fftconvolve for better numerical stability
    conv = fftconvolve(seq, seq, mode='full')
    return conv[:2*n - 1]

def compute_c1_value(seq):
    """Compute the C1 constant from the sequence."""
    n = len(seq)
    if n == 0:
        return float('inf')

    # Use FFT for efficiency when possible
    if n > 100:
        conv = convolve_fft(seq)
    else:
        conv = np.convolve(seq, seq, mode='full')

    max_conv = np.max(conv)
    sum_seq = np.sum(seq)

    if sum_seq < 1e-10:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def evaluate_individual(individual):
    """Evaluate the fitness of an individual (sequence)."""
    # Clip values to [0, 1000]
    seq = np.clip(individual, 0, 1000)

    # Compute C1
    c1 = compute_c1_value(seq)

    # Return negative inverse C1 as fitness (maximize inverse C1)
    if c1 < 1e-10:
        return (float('inf'),)  # Penalize invalid solutions

    inv_c1 = 1.0 / c1
    return (inv_c1,)

def create_structured_individual(n):
    """Create a structured individual to boost exploration."""
    individual = []
    # Mix of geometric decay and random elements
    decay_factor = 0.85
    for i in range(n):
        if i % 3 == 0:
            # Sparse structure with decay
            individual.append(1000 * (decay_factor ** (i // 3)))
        else:
            # Random elements
            individual.append(random.uniform(0, 1000))
    return individual

def create_random_individual(n):
    """Create a purely random individual."""
    return [random.uniform(0, 1000) for _ in range(n)]

def create_individual(n):
    """Create a new individual with given length."""
    # Hybrid approach: 60% structured, 40% random
    if random.random() < 0.6:
        return create_structured_individual(n)
    else:
        return create_random_individual(n)

def mutate_individual(individual, indpb, mut_strength, generation, max_generations):
    """Mutate an individual with adaptive parameters."""
    # Adaptive mutation rate that decreases with generations
    adaptive_indpb = indpb * (1.0 - generation / max_generations)

    for i in range(len(individual)):
        if random.random() < adaptive_indpb:
            # Gaussian mutation with adaptive strength
            individual[i] += random.gauss(0, mut_strength)
            individual[i] = max(0, individual[i])  # Ensure non-negativity
    return individual,

def crossover_individuals(ind1, ind2, cxpb):
    """Crossover two individuals."""
    if random.random() < cxpb:
        # Uniform crossover
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def gradient_ascent_update(seq, max_iter=5):
    """Apply gradient ascent to improve sequence."""
    try:
        seq = np.array(seq, dtype=float)
        n = len(seq)
        if n < 1:
            return seq.tolist()

        # Simple gradient estimation
        def compute_c1_and_grad(s):
            conv = convolve_fft(s)
            max_conv = np.max(conv)
            sum_s = np.sum(s)

            if sum_s < 1e-10:
                return float('inf'), np.zeros_like(s)
                
            c1 = 2 * n * max_conv / (sum_s ** 2)
            
            # Estimate gradient numerically
            eps = 1e-6
            grad = np.zeros_like(s)
            for i in range(n):
                s_eps = s.copy()
                s_eps[i] += eps
                c1_plus = compute_c1_value(s_eps)
                grad[i] = (c1_plus - c1) / eps
                
            return c1, grad
            
        # Perform gradient ascent
        for _ in range(max_iter):
            _, grad = compute_c1_and_grad(seq)
            seq -= 0.01 * grad  # Fixed learning rate for simplicity
            seq = np.maximum(seq, 0)  # Ensure non-negativity
            
        return seq.tolist()
    except Exception as e:
        return seq.tolist()

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimize the sequence using adaptive hybrid method with gradient refinement."""
    try:
        # Determine sequence length
        n = len(sequence)
        if n < 10:
            n = 100  # Default small size
        elif n > 1000:
            n = 1000  # Cap maximum size

        # Set up evolutionary algorithm parameters
        pop_size = max(40, min(200, n // 2))
        gen_limit = max(20, min(70, n // 5))

        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", crossover_individuals)
        toolbox.register("mutate", mutate_individual, indpb=0.1, mut_strength=60, generation=0, max_generations=gen_limit)
        toolbox.register("select", tools.selTournament, tournsize=max(3, int(np.log2(n)) + 2))

        # Create initial population
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution loop with gradient refinement
        for generation in range(gen_limit):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    child1, child2 = toolbox.mate(child1, child2)
                toolbox.mutate(child1, generation, gen_limit)
                toolbox.mutate(child2, generation, gen_limit)
                del child1.fitness.values
                del child2.fitness.values

            # Evaluate new individuals and refine with gradient ascent
            new_individuals = [ind for ind in offspring if not ind.fitness.valid]
            refined_individuals = []
            
            for ind in new_individuals:
                # Apply gradient ascent refinement
                refined = gradient_ascent_update(ind)
                refined_individuals.append(refined)
                
            # Evaluate refined individuals
            fitnesses = list(map(toolbox.evaluate, refined_individuals))
            for ind, fit in zip(refined_individuals, fitnesses):
                ind.fitness.values = fit

            # Replace old population
            population[:] = refined_individuals

        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        return best_ind

    except Exception as e:
        print(f"Error in adaptive optimization: {e}")
        # Return mutated version of input if evolution fails
        return [(x + random.uniform(-100, 100)) for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initialize with a random sequence of moderate size
    n = random.randint(100, 1000)
    sequence = [random.uniform(0, 1000) for _ in range(n)]

    # Apply hybrid optimization
    optimized_sequence = get_good_direction_to_move_into(sequence)

    # Ensure minimum sum constraint
    if sum(optimized_sequence) < 0.01:
        optimized_sequence = [x + random.uniform(0, 1) for x in optimized_sequence]

    return optimized_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")