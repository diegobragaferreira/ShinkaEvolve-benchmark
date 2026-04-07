# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
import time
import math

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
    return sequence

def estimate_gradient(sequence, epsilon=1e-4):
    """Estimates the gradient of the objective function using finite differences."""
    n = len(sequence)
    gradient = np.zeros(n)
    base_fitness = evaluate_sequence(sequence)
    
    for i in range(n):
        perturbed = sequence.copy()
        perturbed[i] += epsilon
        perturbed_fitness = evaluate_sequence(perturbed)
        gradient[i] = (perturbed_fitness - base_fitness) / epsilon
    
    return gradient

def adaptive_gradient_step(current_sequence, gradient, step_size=0.01, clip_range=(0, 1000)):
    """Performs an adaptive gradient step."""
    new_sequence = np.array(current_sequence) - step_size * gradient
    new_sequence = np.clip(new_sequence, clip_range[0], clip_range[1])
    return new_sequence.tolist()

def adaptive_mutation(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Applies adaptive mutation to a sequence."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            new_sequence[i] *= random.uniform(1 - max_mutation, 1 + max_mutation)
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    return new_sequence

def adaptive_evolution_step(population, fitness_scores, elite_fraction=0.2):
    """Performs an adaptive evolutionary step using gradient information."""
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite_count = max(1, int(len(population) * elite_fraction))
    
    # Preserve elites
    elite_pop = [population[i] for i in sorted_indices[:elite_count]]
    
    # Calculate adaptive step size based on fitness variance
    if len(fitness_scores) > 1:
        fitness_variance = np.var(fitness_scores)
        adaptive_step_size = 0.01 * (1.0 / (1.0 + fitness_variance)) if fitness_variance > 0 else 0.01
    else:
        adaptive_step_size = 0.01
    
    # Create offspring using gradient-guided mutation and crossover
    offspring = []
    for i in range(len(population) - elite_count):
        # Select parents (roulette wheel selection)
        parent_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores)/np.sum(fitness_scores))
        parent1 = population[parent_indices[0]]
        parent2 = population[parent_indices[1]]
        
        # Crossover
        child = []
        for j in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[j])
            else:
                child.append(parent2[j])
        
        # Gradient-guided mutation
        gradient = estimate_gradient(child)
        child = adaptive_gradient_step(child, gradient, adaptive_step_size)
        
        # Add adaptive mutation
        child = adaptive_mutation(child)
        
        offspring.append(child)
    
    # Combine elites and offspring
    new_population = elite_pop + offspring
    return new_population

def adaptive_local_search(initial_sequence, max_iter=100):
    """Implements adaptive local search with gradient information."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    
    # Adaptive parameters
    step_size = 0.01
    patience = 0
    patience_limit = 10
    
    for iteration in range(max_iter):
        # Estimate gradient
        gradient = estimate_gradient(current_sequence)
        
        # Take gradient step
        new_sequence = adaptive_gradient_step(current_sequence, gradient, step_size)
        new_fitness = evaluate_sequence(new_sequence)
        
        # Accept improvement
        if new_fitness > current_fitness:
            current_sequence = new_sequence
            current_fitness = new_fitness
            patience = 0
        else:
            patience += 1
            # Reduce step size if no improvement
            if patience > patience_limit:
                step_size *= 0.9
                patience = 0
        
        # Occasionally add random perturbation
        if random.random() < 0.05:
            current_sequence = adaptive_mutation(current_sequence)
            current_fitness = evaluate_sequence(current_sequence)
    
    return current_sequence, current_fitness

def search_for_best_sequence():
    """Main search function using adaptive gradient evolution strategy."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    max_time_seconds = 180
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Strategy 1: Adaptive gradient evolution
    if time.time() - start_time < max_time_seconds:
        # Initialize population
        pop_size = 50
        population = [generate_random_sequence() for _ in range(pop_size)]
        generation = 0
        
        while time.time() - start_time < max_time_seconds and generation < 1000:
            # Evaluate fitness
            fitness_scores = [evaluate_sequence(seq) for seq in population]
            
            # Track best
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_inv_c1:
                best_inv_c1 = fitness_scores[current_best_idx]
                best_sequence = population[current_best_idx].copy()
            
            # Adaptive evolution step
            population = adaptive_evolution_step(population, fitness_scores)
            
            generation += 1
    
    # Strategy 2: Local search refinement of the best found
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        refined_seq, refined_fitness = adaptive_local_search(best_sequence, 500)
        if refined_fitness > best_inv_c1:
            best_inv_c1 = refined_fitness
            best_sequence = refined_seq
    
    # Return the best found
    if best_sequence is None:
        best_sequence = generate_random_sequence()
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")