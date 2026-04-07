# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial
from deap import base, creator, tools, algorithms
import math

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the evaluation function for a single sequence
def evaluate_sequence(sequence):
    """
    Evaluate a sequence and return its performance metric (1/C₁).
    
    Args:
        sequence: List of non-negative real numbers
        
    Returns:
        float: Performance metric (1/C₁) - higher is better
    """
    try:
        # Convert to numpy array
        a = np.array(sequence)
        sum_a = np.sum(a)
        
        # Avoid division by zero or negligible sums
        if sum_a < 1e-10:
            return 0.0
            
        # Compute autoconvolution using FFT for efficiency
        b = fftconvolve(a, a, mode='full')
        b = b[len(a)-1:2*len(a)-1]  # Convolution part
        
        max_b = np.max(b)
        
        # Compute C₁ = 2n * max(b) / (sum(a))^2
        n = len(a)
        c1 = 2 * n * max_b / (sum_a ** 2)
        
        # Return inverse for maximization
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
        
        return inv_c1
    except Exception as e:
        return 0.0

# Parallel evaluation of a batch of sequences
def evaluate_population_parallel(population, chunk_size=10):
    """
    Evaluate a list of sequences in parallel.
    
    Args:
        population: List of sequences to evaluate
        chunk_size: Number of sequences per worker
        
    Returns:
        List of performance metrics (1/C₁) for each sequence
    """
    # Split the population into chunks
    chunks = [population[i:i+chunk_size] for i in range(0, len(population), chunk_size)]
    
    # Use multiprocessing to evaluate chunks in parallel
    with mp.Pool() as pool:
        results = pool.map(evaluate_sequences_chunk, chunks)
    
    # Flatten results
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results

# Helper function to evaluate a chunk of sequences
def evaluate_sequences_chunk(chunk):
    return [evaluate_sequence(seq) for seq in chunk]

# Initialize a random valid sequence
def generate_random_valid_sequence():
    """Generate a random valid sequence with length between 100 and 500."""
    length = random.randint(100, 500)
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    
    # Ensure sum is meaningful
    if sum(sequence) < 0.01:
        sequence[random.randint(0, length-1)] += 0.01
    
    return sequence

# Generate initial population
def generate_initial_population(pop_size):
    """Generate an initial population of random valid sequences."""
    return [generate_random_valid_sequence() for _ in range(pop_size)]

# Genetic algorithm main loop
def optimize_step_function_evolutionary():
    """Optimize the step function using an evolutionary algorithm."""
    # Configuration
    pop_size = 50
    ngen = 50
    cxpb = 0.7  # Crossover probability
    mutpb = 0.2  # Mutation probability
    
    # Initialize population
    population = generate_initial_population(pop_size)
    
    # Evaluate initial population
    fitnesses = evaluate_population_parallel(population)
    
    # Update individual fitnesses
    for i, fit in enumerate(fitnesses):
        population[i].fitness.values = (fit,)
    
    # Begin evolution
    for gen in range(ngen):
        # Select the next generation individuals
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(creator.Individual, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tools.cxUniform(child1, child2, 0.5)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(mutant, mu=0, sigma=100, indpb=0.1)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = evaluate_population_parallel(invalid_ind)
        
        for i, fit in enumerate(fitnesses):
            invalid_ind[i].fitness.values = (fit,)
            
        # Replace the old population with the new one
        population[:] = offspring
        
        # Print statistics
        fits = [ind.fitness.values[0] for ind in population]
        if fits:
            best_fit = max(fits)
            avg_fit = sum(fits) / len(fits)
            print(f"Gen {gen}: Best = {best_fit:.6f}, Avg = {avg_fit:.6f}")
    
    # Find and return the best individual
    best_ind = tools.selBest(population, 1)[0]
    return best_ind

# Main function to call the optimization
def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()
    timeout = 170  # Leave 10 seconds for cleanup
    
    try:
        # Run evolutionary optimization
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to a basic sequence if nothing worked
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
