# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
import math

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

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)

    # Generate random heights between 0 and 1000
    sequence = [random.uniform(0, 1000) for _ in range(length)]

    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0

    return sequence

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence by randomly modifying elements."""
    new_sequence = sequence.copy()

    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
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

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence using LP optimization."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    if sum_sequence < 0.01:
        return None
    
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        return None
    
    sum_sequence = np.sum(g_fun)
    if sum_sequence < 0.01:
        return None
    
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def genetic_algorithm_search(max_time_seconds=180):
    """Search using genetic algorithm approach with LP-based enhancements."""
    start_time = time.time()

    # Initialize population
    pop_size = 50
    population = [generate_random_sequence() for _ in range(pop_size)]

    best_individual = None
    best_fitness = 0.0

    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1

        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # Selection - tournament selection
        selected = []
        for _ in range(pop_size):
            # Tournament selection of 3
            tournament_indices = random.sample(range(pop_size), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())

        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % pop_size]

            # Crossover
            child1 = crossover_sequences(parent1, parent2)
            child2 = crossover_sequences(parent2, parent1)

            # Mutation
            child1 = mutate_sequence(child1)
            child2 = mutate_sequence(child2)

            # Enhance with LP-based direction
            enhanced_child1 = get_good_direction_to_move_into(child1)
            if enhanced_child1 is not None:
                child1 = enhanced_child1
            
            enhanced_child2 = get_good_direction_to_move_into(child2)
            if enhanced_child2 is not None:
                child2 = enhanced_child2

            new_population.extend([child1, child2])

        population = new_population[:pop_size]

        # Occasionally add some diversity
        if generation % 10 == 0:
            for i in range(0, pop_size, 5):
                if i < pop_size:
                    population[i] = generate_random_sequence()

    return (best_individual, best_fitness)

def local_improvement_search(initial_sequence, max_iter=100):
    """Improve a sequence using local search around it."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)

    for _ in range(max_iter):
        # Try mutating the current sequence
        mutated = mutate_sequence(current_sequence, mutation_rate=0.3, max_mutation=0.2)
        mutated_fitness = evaluate_sequence(mutated)

        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness

        # Try LP-based enhancement
        enhanced = get_good_direction_to_move_into(current_sequence)
        if enhanced is not None:
            enhanced_fitness = evaluate_sequence(enhanced)
            if enhanced_fitness > current_fitness:
                current_sequence = enhanced
                current_fitness = enhanced_fitness

    return current_sequence, current_fitness

def search_for_best_sequence():
    """Main search function to find the best sequence."""
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

    return best_sequence if best_sequence is not None else generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")