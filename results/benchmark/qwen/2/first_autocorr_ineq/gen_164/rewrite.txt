# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    # Ensure sequence is numpy array
    a = np.array(sequence)

    # Skip if sum is too small to avoid numerical issues
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')

    # Compute convolution using FFT for better performance
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]

    # Find maximum in the convolution
    max_conv = np.max(conv)

    # Calculate C1 = 2n * max(b) / (sum(a))^2
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)

    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0  # Penalty for invalid sequences
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_random_sequence(length=None, min_length=100, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)

    # Generate random heights between 0 and 1000
    sequence = [random.uniform(0, 1000) for _ in range(length)]

    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0

    return sequence

def generate_structured_sequence(length=None):
    """Generate a structured sequence to provide better starting points."""
    if length is None:
        length = random.randint(100, 1000)
    
    # Create a sequence with a peak in the middle
    sequence = []
    midpoint = length // 2
    for i in range(length):
        distance_from_center = abs(i - midpoint)
        # Gaussian-like shape centered at midpoint
        value = max(0, 1000 * np.exp(-0.5 * (distance_from_center / (length / 6)) ** 2))
        sequence.append(value)
    
    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
        
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.3, generation=0):
    """Mutate a sequence by randomly modifying elements."""
    new_sequence = sequence.copy()
    
    # Adapt mutation rate over generations
    adapted_mutation_rate = mutation_rate * (0.9 ** generation)

    for i in range(len(new_sequence)):
        if random.random() < adapted_mutation_rate:
            # Apply random mutation to the element
            new_sequence[i] *= random.uniform(1 - max_mutation, 1 + max_mutation)
            # Clip to valid range [0, 1000]
            new_sequence[i] = max(0, min(1000, new_sequence[i]))

    # Ensure at least one element is positive
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])

    return new_sequence

def crossover_sequences(seq1, seq2):
    """Perform uniform crossover between two sequences."""
    # Make sequences same length by padding with zeros or truncating
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    # Pad shorter sequence with zeros
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))

    # Perform crossover
    new_seq = []
    for i in range(max_len):
        if random.random() < 0.5:
            new_seq.append(padded_seq1[i])
        else:
            new_seq.append(padded_seq2[i])

    return new_seq

def genetic_algorithm_search(max_time_seconds=180, population_size=50, generations=100):
    """Search using genetic algorithm approach with adaptive parameters."""
    start_time = time.time()
    
    # Initialize population
    population = [generate_structured_sequence() for _ in range(population_size)]
    
    best_individual = None
    best_fitness = 0.0
    elite_size = max(1, population_size // 10)  # Top 10% as elite

    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Keep top 10% as elites
        elites = sorted_population[:elite_size]

        # Selection - tournament selection with elitism
        selected = elites.copy()  # Start with elites
        remaining_slots = population_size - elite_size

        for _ in range(remaining_slots):
            # Tournament selection of 3
            tournament_indices = random.sample(range(len(sorted_population)), 3)
            tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(sorted_population[winner_index].copy())

        # Create new population through crossover and mutation
        new_population = elites.copy()  # Preserve elites
        for i in range(elite_size, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]

            # Crossover
            child1 = crossover_sequences(parent1, parent2)
            child2 = crossover_sequences(parent2, parent1)

            # Mutation
            child1 = mutate_sequence(child1, mutation_rate=0.15, max_mutation=0.3, generation=gen)
            child2 = mutate_sequence(child2, mutation_rate=0.15, max_mutation=0.3, generation=gen)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    return (best_individual, best_fitness)

def local_improvement_search(initial_sequence, max_iter=100):
    """Improve a sequence using local search around it."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)

    for _ in range(max_iter):
        # Try mutating the current sequence with smaller changes
        mutated = mutate_sequence(current_sequence, mutation_rate=0.3, max_mutation=0.1)
        mutated_fitness = evaluate_sequence(mutated)

        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness

    return current_sequence, current_fitness

def gradient_based_refinement(sequence, iterations=50):
    """Use gradient-based approach to refine the sequence."""
    current_sequence = np.array(sequence, dtype=float)
    current_fitness = evaluate_sequence(current_sequence)
    
    # Initial learning rate
    lr = 0.01
    eps = 1e-6

    for iteration in range(iterations):
        # Estimate gradient using finite differences
        grad = np.zeros_like(current_sequence)
        for i in range(len(current_sequence)):
            # Compute numerical gradient
            delta = np.zeros_like(current_sequence)
            delta[i] = eps
            f_plus = evaluate_sequence(current_sequence + delta)
            f_minus = evaluate_sequence(current_sequence - delta)
            grad[i] = (f_plus - f_minus) / (2 * eps)

        # Update step
        new_sequence = current_sequence + lr * grad

        # Ensure non-negativity and reasonable bounds
        new_sequence = np.maximum(0, new_sequence)
        new_sequence = np.minimum(1000, new_sequence)

        # Check if update improved fitness
        new_fitness = evaluate_sequence(new_sequence)
        if new_fitness > current_fitness:
            current_sequence = new_sequence
            current_fitness = new_fitness
        else:
            # Reduce learning rate if no improvement
            lr *= 0.95
            if lr < 1e-6:
                break

    return current_sequence.tolist(), current_fitness

def hybrid_search(max_time_seconds=180):
    """Combine multiple search strategies to improve optimization."""
    start_time = time.time()
    best_sequence = None
    best_fitness = 0.0

    # Strategy 1: Genetic Algorithm with structured initialization
    if time.time() - start_time < max_time_seconds:
        ga_seq, ga_fitness = genetic_algorithm_search(
            max_time_seconds - (time.time() - start_time),
            population_size=30,
            generations=50
        )
        if ga_fitness > best_fitness:
            best_fitness = ga_fitness
            best_sequence = ga_seq

    # Strategy 2: Refinement with gradient descent
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        grad_seq, grad_fitness = gradient_based_refinement(best_sequence, 100)
        if grad_fitness > best_fitness:
            best_fitness = grad_fitness
            best_sequence = grad_seq

    # Strategy 3: Local improvement with adaptive mutation
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        local_seq, local_fitness = local_improvement_search(best_sequence, 200)
        if local_fitness > best_fitness:
            best_fitness = local_fitness
            best_sequence = local_seq

    # Strategy 4: Multiple random starts with refinement
    for attempt in range(3):
        if time.time() - start_time >= max_time_seconds:
            break
            
        # Start with a structured sequence
        initial_seq = generate_structured_sequence()
        
        # Local improvement
        improved_seq, improved_fitness = local_improvement_search(initial_seq, 100)
        if improved_fitness > best_fitness:
            best_fitness = improved_fitness
            best_sequence = improved_seq
            
        # Gradient refinement
        if best_sequence is not None:
            grad_seq, grad_fitness = gradient_based_refinement(best_sequence, 50)
            if grad_fitness > best_fitness:
                best_fitness = grad_fitness
                best_sequence = grad_seq

    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    sequence, fitness = hybrid_search(180)
    
    # If no good solution was found, fall back to basic approach
    if sequence is None:
        # Start with a few diverse sequences
        best_sequence = None
        best_inv_c1 = 0.0
        
        # Try multiple random starting points
        for attempt in range(5):
            # Random initialization
            initial_sequence = generate_random_sequence()
            
            # Local improvement
            improved_seq, improved_fitness = local_improvement_search(initial_sequence, 100)
            
            if improved_fitness > best_inv_c1:
                best_inv_c1 = improved_fitness
                best_sequence = improved_seq
                
            # Also try genetic algorithm approach
            ga_seq, ga_fitness = genetic_algorithm_search(10)  # Shorter time for GA
            if ga_fitness > best_inv_c1:
                best_inv_c1 = ga_fitness
                best_sequence = ga_seq
                
        # Final local optimization on the best found sequence
        if best_sequence is not None:
            final_seq, final_fitness = local_improvement_search(best_sequence, 500)
            if final_fitness > best_inv_c1:
                best_inv_c1 = final_fitness
                best_sequence = final_seq
                
        sequence = best_sequence if best_sequence is not None else generate_random_sequence()

    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")