# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
import math

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

def estimate_gradient(sequence, eps=1e-6):
    """Estimate gradient of the inverse C1 function using finite differences."""
    n = len(sequence)
    grad = np.zeros(n)
    base_fitness = evaluate_sequence(sequence)
    
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = eps
        f_plus = evaluate_sequence(sequence + delta)
        f_minus = evaluate_sequence(sequence - delta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def adaptive_mutation(sequence, gradient, mutation_strength=0.1):
    """Apply adaptive mutation influenced by gradient information."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        # Adjust mutation strength based on gradient magnitude
        if abs(gradient[i]) > 1e-8:  # Only adjust if gradient is significant
            # Mutate more aggressively in directions with higher gradient
            adj_mutation = mutation_strength * (1 + abs(gradient[i]))
        else:
            adj_mutation = mutation_strength
            
        if random.random() < 0.1:  # 10% chance to mutate
            scale_factor = random.uniform(1 - adj_mutation, 1 + adj_mutation)
            new_sequence[i] *= scale_factor
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    
    # Ensure at least one element is positive
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])

    return new_sequence

def guided_crossover(seq1, seq2, gradient1, gradient2):
    """Perform crossover guided by gradient information."""
    # Prefer elements with higher gradient magnitudes
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    # Pad shorter sequence
    if len(seq1) < max_len:
        seq1.extend([0] * (max_len - len(seq1)))
    if len(seq2) < max_len:
        seq2.extend([0] * (max_len - len(seq2)))
    
    # Create child based on gradient information
    child = []
    for i in range(max_len):
        # Select element with higher gradient influence
        if i < min_len:
            if abs(gradient1[i]) >= abs(gradient2[i]):
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        else:
            # For extended parts, choose based on sequence lengths
            if len(seq1) > len(seq2):
                child.append(seq1[i])
            else:
                child.append(seq2[i])
    
    return child

def gradient_guided_evolution(max_time_seconds=180):
    """Evolutionary algorithm guided by gradients."""
    start_time = time.time()
    
    # Initialize population with diverse sequences
    pop_size = 50
    population = []
    for _ in range(pop_size):
        length = random.randint(100, 1000)
        individual = [random.uniform(0, 1000) for _ in range(length)]
        population.append(individual)
    
    best_individual = None
    best_fitness = 0.0

    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        
        # Evaluate fitness for all individuals
        fitness_scores = [evaluate_sequence(ind) for ind in population]
        
        # Track best individual
        for i, fitness in enumerate(fitness_scores):
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = population[i].copy()
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top 20% as elites
        elite_count = max(1, pop_size // 5)
        elites = sorted_population[:elite_count]
        
        # Estimate gradients for elites
        elite_gradients = []
        for i in range(elite_count):
            try:
                grad = estimate_gradient(np.array(sorted_population[i]))
                elite_gradients.append(grad)
            except:
                elite_gradients.append(np.zeros(len(sorted_population[i])))
        
        # Create new population
        new_population = elites.copy()
        
        # Generate offspring using gradient-guided operations
        while len(new_population) < pop_size:
            # Select parents using tournament selection
            parent1_idx = random.choice(range(elite_count))
            parent2_idx = random.choice(range(elite_count))
            
            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]
            
            # Estimate gradients for parents
            grad1 = elite_gradients[parent1_idx]
            grad2 = elite_gradients[parent2_idx]
            
            # Guided crossover
            child = guided_crossover(parent1, parent2, grad1, grad2)
            
            # Adaptive mutation
            child = adaptive_mutation(child, (grad1 + grad2) / 2)
            
            # Ensure minimum sum
            if np.sum(child) < 0.01:
                child[0] = 0.1
                
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Occasionally introduce diversity
        if generation % 20 == 0:
            for i in range(0, pop_size, 7):
                if i < pop_size:
                    population[i] = generate_random_sequence()

    return (best_individual, best_fitness)

def multi_resolution_search(max_time_seconds=180):
    """Search using multiple sequence resolutions."""
    start_time = time.time()
    best_sequence = None
    best_fitness = 0.0
    
    # Different resolution levels
    resolutions = [100, 200, 400, 600, 800, 1000]
    
    for res in resolutions:
        if time.time() - start_time >= max_time_seconds:
            break
            
        # Generate sequences of specific resolution
        seq = generate_random_sequence(res)
        
        # Local improvement
        improved_seq, improved_fitness = local_improvement_search(seq, 50)
        
        if improved_fitness > best_fitness:
            best_fitness = improved_fitness
            best_sequence = improved_seq
            
    # Final gradient-based refinement if time allows
    if best_sequence is not None and (time.time() - start_time) < max_time_seconds:
        grad_seq, grad_fitness = gradient_based_improvement(best_sequence, 50)
        if grad_fitness > best_fitness:
            best_fitness = grad_fitness
            best_sequence = grad_seq
    
    return best_sequence, best_fitness

def local_improvement_search(initial_sequence, max_iter=100):
    """Improve a sequence using local search around it."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)

    for _ in range(max_iter):
        # Try mutating the current sequence with smaller changes
        mutated = adaptive_mutation(current_sequence, np.zeros(len(current_sequence)), 0.05)
        mutated_fitness = evaluate_sequence(mutated)

        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness

    return current_sequence, current_fitness

def gradient_based_improvement(sequence, max_iter=50):
    """Improve sequence using simple gradient estimation."""
    current_sequence = np.array(sequence, dtype=float)
    current_fitness = evaluate_sequence(current_sequence)
    step_size = 0.01
    eps = 1e-6

    for iteration in range(max_iter):
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
        new_sequence = current_sequence + step_size * grad

        # Ensure non-negativity and reasonable bounds
        new_sequence = np.maximum(0, new_sequence)
        new_sequence = np.minimum(1000, new_sequence)

        # Check if update improved fitness
        new_fitness = evaluate_sequence(new_sequence)
        if new_fitness > current_fitness:
            current_sequence = new_sequence
            current_fitness = new_fitness
        else:
            # Reduce step size if no improvement
            step_size *= 0.95
            if step_size < 1e-6:
                break

    return current_sequence.tolist(), current_fitness

def ensemble_search(max_time_seconds=180):
    """Ensemble approach combining gradient-guided evolution and multi-resolution search."""
    start_time = time.time()
    best_sequence = None
    best_fitness = 0.0

    # Strategy 1: Gradient-guided evolution
    if time.time() - start_time < max_time_seconds:
        try:
            evol_seq, evol_fitness = gradient_guided_evolution(max_time_seconds - (time.time() - start_time))
            if evol_fitness > best_fitness:
                best_fitness = evol_fitness
                best_sequence = evol_seq
        except Exception:
            pass

    # Strategy 2: Multi-resolution search
    if time.time() - start_time < max_time_seconds:
        try:
            mr_seq, mr_fitness = multi_resolution_search(max_time_seconds - (time.time() - start_time))
            if mr_fitness > best_fitness:
                best_fitness = mr_fitness
                best_sequence = mr_seq
        except Exception:
            pass

    # Strategy 3: Local refinement on best found so far
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        try:
            final_seq, final_fitness = local_improvement_search(best_sequence, 100)
            if final_fitness > best_fitness:
                best_fitness = final_fitness
                best_sequence = final_seq
        except Exception:
            pass

    # Strategy 4: Gradient refinement
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        try:
            grad_seq, grad_fitness = gradient_based_improvement(best_sequence, 50)
            if grad_fitness > best_fitness:
                best_fitness = grad_fitness
                best_sequence = grad_seq
        except Exception:
            pass

    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    # Use ensemble method for better results
    sequence, fitness = ensemble_search(180)

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
            try:
                ga_seq, ga_fitness = gradient_guided_evolution(10)  # Shorter time for GA
                if ga_fitness > best_inv_c1:
                    best_inv_c1 = ga_fitness
                    best_sequence = ga_seq
            except:
                continue

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