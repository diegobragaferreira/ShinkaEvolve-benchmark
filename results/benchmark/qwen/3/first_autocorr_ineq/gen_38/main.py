# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from deap import base, creator, tools, algorithms
import random
import time
from functools import partial
import warnings

warnings.filterwarnings('ignore')

def convolve_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Zero-pad to avoid circular convolution effects
    padded_len = 2 * n - 1
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant')
    conv = np.fft.irfft(np.fft.rfft(padded_seq) ** 2)
    return conv[:padded_len]

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

def create_individual(n):
    """Create a new individual with given length."""
    # Create a diverse sequence with some structure to encourage exploration
    individual = []
    for i in range(n):
        # Mix of random values and structured elements
        if i % 3 == 0:
            individual.append(random.uniform(0, 500))  # Sparse structure
        else:
            individual.append(random.uniform(0, 1000))
    return individual

def mutate_individual(individual, indpb, mut_strength):
    """Mutate an individual with adaptive mutation strength."""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Adaptive mutation based on current value
            adaptive_strength = mut_strength * (1.0 + abs(individual[i]) / 1000.0)
            individual[i] += random.gauss(0, adaptive_strength)
            individual[i] = max(0, individual[i])  # Ensure non-negativity
    return individual,

def crossover_individuals(ind1, ind2, cxpb):
    """Crossover two individuals with uniform crossover."""
    if random.random() < cxpb:
        # Uniform crossover with some preference for maintaining structure
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimize the sequence using evolutionary algorithm with adaptive parameters."""
    try:
        # Determine sequence length
        n = len(sequence)
        if n < 10:
            n = 100  # Default small size
        elif n > 1000:
            n = 1000  # Cap maximum size
            
        # Adaptive parameters based on sequence size
        pop_size = max(20, min(200, n // 2))
        gen_limit = max(10, min(100, n // 5))
        mut_strength = 100.0 / np.sqrt(n)  # Decreasing mutation strength with sequence size
        
        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", crossover_individuals)
        toolbox.register("mutate", mutate_individual, indpb=0.1, mut_strength=mut_strength)
        toolbox.register("select", tools.selTournament, tournsize=max(3, int(np.log2(n))))  # Adaptive tournament size
        
        # Create initial population
        population = toolbox.population(n=pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            
        # Evolution loop with early stopping based on improvement
        last_best_fitness = float('-inf')
        stagnation_count = 0
        max_stagnation = 10  # Early stopping after 10 generations without improvement
        
        for generation in range(gen_limit):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    child1, child2 = toolbox.mate(child1, child2)
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                del child1.fitness.values
                del child2.fitness.values
                
            # Evaluate new individuals
            new_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, new_individuals))
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit
                
            # Replace old population
            population[:] = offspring
            
            # Check for improvement
            best_fitness = max(ind.fitness.values[0] for ind in population)
            if best_fitness > last_best_fitness:
                last_best_fitness = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if stagnation_count >= max_stagnation:
                break  # Early stopping
            
        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        return best_ind
        
    except Exception as e:
        print(f"Error in evolutionary optimization: {e}")
        # Return mutated version of input if evolution fails
        return [(x + random.uniform(-100, 100)) for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initialize with a diverse random sequence
    n = random.randint(100, 1000)
    sequence = [random.uniform(0, 1000) for _ in range(n)]
    
    # Apply evolutionary optimization
    optimized_sequence = get_good_direction_to_move_into(sequence)
    
    # Ensure minimum sum constraint
    if sum(optimized_sequence) < 0.01:
        optimized_sequence = [x + random.uniform(0, 1) for x in optimized_sequence]
        
    return optimized_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
