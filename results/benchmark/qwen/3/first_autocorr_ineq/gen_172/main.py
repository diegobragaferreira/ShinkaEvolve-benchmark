# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from deap import base, creator, tools, algorithms
import random
import time
from functools import partial
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# Global history to store top performers for historical sampling
top_performers = deque(maxlen=10)

def convolve_fft(seq):
    """Compute convolution using FFT for better performance with enhanced numerical stability."""
    n = len(seq)

    # Use scipy's fftconvolve for better numerical stability and handling of edge cases
    from scipy.signal import fftconvolve

    # Pre-pad to ensure proper linear convolution behavior
    padded_len = 2 * n - 1
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant', constant_values=0)
    conv = fftconvolve(padded_seq, padded_seq, mode='full')

    # Return only the linear convolution part (should be same length as padded_len)
    result = conv[:padded_len]

    # Apply additional numerical stabilization for very large sequences
    if n > 1000:
        # For large sequences, apply a small amount of numerical damping
        # to prevent floating point artifacts
        result = np.clip(result, 0, np.max(result) * 10)

    return result

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

def create_individual(n, use_historical=False):
    """Create a new individual with given length."""
    # Hybrid initialization strategy with historical sampling
    individual = []

    if use_historical and len(top_performers) > 0 and random.random() < 0.3:
        # Use historical best performer with slight variation
        historical = random.choice(list(top_performers))
        for i in range(n):
            if i < len(historical):
                # Add small random variation
                variation = random.gauss(0, 0.1 * historical[i] if historical[i] != 0 else 1)
                individual.append(max(0, historical[i] + variation))
            else:
                individual.append(random.uniform(0, 1000))
    elif n < 50:
        # For small sequences, use a structured pattern
        for i in range(n):
            individual.append(random.uniform(0, 1000))
    else:
        # For larger sequences, use a combination of patterns
        # First half: geometric decay
        decay_factor = 0.9
        for i in range(n//2):
            individual.append(1000 * (decay_factor ** i))
        # Second half: random values
        for i in range(n//2, n):
            individual.append(random.uniform(0, 1000))
    return individual

def mutate_individual(individual, indpb, mut_strength, generation=None):
    """Mutate an individual with advanced adaptive mutation."""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Use different mutation strategies based on generation and position
            if generation is not None and generation > 20:
                # Later generations: use Cauchy for broader exploration
                mutation = random.gauss(0, mut_strength * (1.0 + abs(individual[i]) / 1000.0))
            else:
                # Earlier generations: use Gaussian for fine-tuning
                mutation = random.gauss(0, mut_strength * (1.0 + abs(individual[i]) / 1000.0))

            individual[i] += mutation
            individual[i] = max(0, individual[i])  # Ensure non-negativity
    return individual,

def crossover_individuals(ind1, ind2, cxpb):
    """Crossover two individuals with adaptive crossover."""
    if random.random() < cxpb:
        # Adaptive crossover with preference for maintaining structure in later gens
        for i in range(len(ind1)):
            if random.random() < 0.7:  # 70% chance to maintain structure
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimize the sequence using enhanced evolutionary algorithm with adaptive parameters."""
    try:
        # Determine sequence length with more dynamic selection
        n = len(sequence)
        if n < 10:
            n = random.choice([50, 100, 128])  # Prefer smaller, manageable sizes
        elif n > 1000:
            n = random.choice([500, 600, 700, 800, 900, 1000])  # Focus on large sequences but varied
        else:
            # Randomly select from common sizes for better coverage
            common_sizes = [50, 100, 128, 200, 256, 300, 400, 500, 512, 600, 700, 800, 900, 1000]
            n = random.choice(common_sizes)

        # Adaptive parameters based on sequence size and complexity
        pop_size = max(30, min(200, n // 2 + 10))  # Larger populations for larger sequences
        gen_limit = max(20, min(100, n // 5))  # More generations for larger sequences
        mut_strength = 50.0 / np.sqrt(n)  # Reduced mutation strength for finer adjustments
        cxpb = 0.7  # Increased crossover probability for more recombination

        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # Use the enhanced initialization logic
        toolbox.register("individual", create_individual, n, use_historical=True)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", crossover_individuals)
        toolbox.register("mutate", mutate_individual, indpb=0.1, mut_strength=mut_strength)  # Slightly higher mutation rate
        toolbox.register("select", tools.selTournament, tournsize=max(3, int(np.log2(n)) + 1))

        # Create initial population with diverse initialization
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution loop with more sophisticated early stopping criteria
        last_best_fitness = float('-inf')
        stagnation_count = 0
        max_stagnation = 15  # Slightly longer stagnation period to allow for complex convergence
        improvement_threshold = 1e-5  # Minimum improvement required

        for generation in range(gen_limit):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    child1, child2 = toolbox.mate(child1, child2)
                # Pass generation info to mutation function
                toolbox.mutate(child1, generation=generation)
                toolbox.mutate(child2, generation=generation)
                del child1.fitness.values
                del child2.fitness.values

            # Evaluate new individuals
            new_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, new_individuals))
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit

            # Replace old population
            population[:] = offspring

            # Check for improvement with a more sensitive measure
            best_fitness = max(ind.fitness.values[0] for ind in population)
            if best_fitness > last_best_fitness + improvement_threshold:
                last_best_fitness = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Better early stopping condition
            if stagnation_count >= max_stagnation:
                break  # Early stopping

        # Get best individual
        best_ind = tools.selBest(population, 1)[0]

        # Save top performers for historical sampling
        if len(top_performers) < top_performers.maxlen:
            top_performers.append(best_ind)
        else:
            # Replace oldest if better
            if any(fitness > top_performers[0].fitness.values[0] for fitness in [ind.fitness.values[0] for ind in population]):
                # Remove oldest and append new best
                top_performers.popleft()
                top_performers.append(best_ind)

        return best_ind

    except Exception as e:
        print(f"Error in enhanced evolutionary optimization: {e}")
        # Return mutated version of input if evolution fails
        return [(x + random.uniform(-100, 100)) for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Sample sequence lengths strategically to boost optimization success
    candidates = [50, 100, 128, 200, 256, 300, 400, 500, 512, 600, 700, 800, 900, 1000]
    n = random.choice(candidates)
    sequence = [random.uniform(0, 1000) for _ in range(n)]

    # Apply enhanced evolutionary optimization
    optimized_sequence = get_good_direction_to_move_into(sequence)

    # Ensure minimum sum constraint
    if sum(optimized_sequence) < 0.01:
        optimized_sequence = [x + random.uniform(0, 1) for x in optimized_sequence]

    return optimized_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")