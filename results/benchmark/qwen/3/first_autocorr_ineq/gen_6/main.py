# EVOLVE-BLOCK-START

import numpy as np
import multiprocessing as mp
from functools import partial
from scipy import signal
import random
import time

np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Computes the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute autoconvolution using FFT for efficiency
    conv = signal.fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    
    # Sum of squares over 2n
    sum_sq = np.sum(np.array(sequence)**2)
    n = len(sequence)
    
    if sum_sq == 0:
        return float('inf')

    # Calculate C1
    c1 = (2 * n * max_conv) / sum_sq
    
    return c1

def inv_c1(sequence):
    """Computes the inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generates a random step function sequence."""
    if length is None:
        length = np.random.randint(min_length, max_length)
    
    # Generate random heights between 0 and 1000
    sequence = np.random.uniform(0, 1000, size=length)
    return sequence

def mutate_sequence(sequence, rate=0.1):
    """Mutates a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if np.random.random() < rate:
            mutated[i] = np.random.uniform(0, 1000)
    return mutated

def local_search(sequence, iterations=50):
    """Performs simple local search around the sequence."""
    current = sequence.copy()
    current_inv_c1 = inv_c1(current)
    
    for _ in range(iterations):
        candidate = mutate_sequence(current, rate=0.2)
        candidate_inv_c1 = inv_c1(candidate)
        
        if candidate_inv_c1 > current_inv_c1:
            current = candidate
            current_inv_c1 = candidate_inv_c1
            
    return current

def evaluate_candidate(args):
    """Evaluate a single candidate sequence."""
    sequence, _ = args
    try:
        return inv_c1(sequence)
    except:
        return 0

def parallel_evaluate(candidates, num_processes=4):
    """Evaluate candidates in parallel."""
    func = partial(evaluate_candidate)
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(func, candidates)
    return results

def search_for_best_sequence(max_time=180):
    """Search for the best sequence using a hybrid optimization approach."""
    start_time = time.time()
    best_inv_c1 = 0
    best_sequence = None
    
    # Multi-start approach with various strategies
    strategies = [
        ("random", lambda: generate_random_sequence()),
        ("sparse", lambda: generate_random_sequence() * (np.random.random() < 0.3)),
        ("dense", lambda: generate_random_sequence() * (np.random.random() > 0.7)),
    ]
    
    # Initial population
    pop_size = 200
    population = []
    
    # Generate diverse initial sequences
    for _ in range(pop_size):
        strategy_name, generator = random.choice(strategies)
        seq = generator()
        # Ensure sum is not too small
        if np.sum(seq) < 0.01:
            seq = seq * 100
        # Clip to avoid extreme values
        seq = np.clip(seq, 0, 1000)
        population.append(seq)
    
    # Evaluate initial population
    candidates = [(seq, i) for i, seq in enumerate(population)]
    fitness_scores = parallel_evaluate(candidates)
    
    # Track best
    best_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_idx]
    if best_fitness > best_inv_c1:
        best_inv_c1 = best_fitness
        best_sequence = population[best_idx].copy()
    
    # Main optimization loop
    iteration = 0
    while time.time() - start_time < max_time - 5:  # Leave 5s for final cleanup
        iteration += 1
        
        # Create offspring through mutation and crossover
        offspring = []
        for i in range(pop_size // 2):
            parent1 = population[np.random.randint(0, len(population))]
            parent2 = population[np.random.randint(0, len(population))]
            
            # Simple crossover
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
            # Mutate children
            child1 = mutate_sequence(child1, rate=0.1)
            child2 = mutate_sequence(child2, rate=0.1)
            
            # Local search on offspring
            child1 = local_search(child1, iterations=20)
            child2 = local_search(child2, iterations=20)
            
            offspring.extend([child1, child2])
        
        # Add some random sequences
        for _ in range(pop_size // 10):
            new_seq = generate_random_sequence()
            if np.sum(new_seq) < 0.01:
                new_seq = new_seq * 100
            new_seq = np.clip(new_seq, 0, 1000)
            offspring.append(new_seq)
        
        # Evaluate offspring
        candidates = [(seq, i) for i, seq in enumerate(offspring)]
        fitness_scores = parallel_evaluate(candidates)
        
        # Select best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size]
        population = [offspring[i] for i in sorted_indices]
        
        # Update best
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_inv_c1:
            best_inv_c1 = fitness_scores[best_idx]
            best_sequence = population[best_idx].copy()
            
        # Occasionally reinitialize
        if iteration % 10 == 0:
            for i in range(pop_size // 5):
                if np.random.random() < 0.3:
                    population[i] = generate_random_sequence()
    
    # Final local search on best found
    if best_sequence is not None:
        best_sequence = local_search(best_sequence, iterations=100)
        best_inv_c1 = inv_c1(best_sequence)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
