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

def smart_mutate(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Enhanced mutation that considers convolution properties."""
    new_sequence = sequence.copy()
    n = len(sequence)
    
    # Analyze the sequence to identify potentially problematic regions
    conv = fftconvolve(sequence, sequence, mode='full')[:2*n-1]
    max_conv = np.max(conv)
    
    # Mutate with probability adjusted by convolution impact
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Adjust mutation magnitude based on sequence characteristics
            mutation_factor = 1.0
            if max_conv > 0 and i < n:  # Only adjust for relevant indices
                # Reduce mutation if this element contributes significantly to convolution max
                # This heuristic promotes stability in high-contribution areas
                contribution_ratio = sequence[i] / max(sequence) if max(sequence) > 0 else 0
                mutation_factor = 1.0 - 0.3 * contribution_ratio  # Less mutation for high-contributing elements
            
            # Apply mutation with adjusted factor
            new_sequence[i] *= random.uniform(
                max(0.5, 1 - max_mutation * mutation_factor),
                min(1.5, 1 + max_mutation * mutation_factor)
            )
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    
    # Ensure at least one element is positive
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    
    return new_sequence

def adaptive_crossover(seq1, seq2):
    """Adaptive crossover that preserves high-performing characteristics."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))
    
    # Perform crossover with preference towards better performing parents
    new_seq = []
    for i in range(max_len):
        # Probability of inheriting from seq1 increases with fitness
        fitness1 = evaluate_sequence(seq1[:min_len] + [0]*(max_len-min_len)) if len(seq1) <= max_len else 0.0
        fitness2 = evaluate_sequence(seq2[:min_len] + [0]*(max_len-min_len)) if len(seq2) <= max_len else 0.0
        
        prob = fitness1 / (fitness1 + fitness2 + 1e-10) if (fitness1 + fitness2) > 0 else 0.5
        if random.random() < prob:
            new_seq.append(padded_seq1[i])
        else:
            new_seq.append(padded_seq2[i])
    
    return new_seq

def estimate_gradient_and_improve(sequence, iterations=10):
    """Estimates a local gradient and improves the sequence."""
    current = np.array(sequence)
    n = len(current)
    best = current.copy()
    best_fitness = evaluate_sequence(list(best))
    
    # Simple finite difference estimation of direction of improvement
    eps = 1e-4
    direction = np.zeros_like(current)
    for i in range(n):
        # Perturb dimension i
        perturbed = current.copy()
        perturbed[i] += eps
        fitness_perturbed = evaluate_sequence(list(perturbed))
        
        # Estimate partial derivative
        partial_derivative = (fitness_perturbed - best_fitness) / eps
        direction[i] = partial_derivative
    
    # Update using gradient-like direction
    step_size = 0.01 * (0.99 ** iterations)  # Decay step size
    new_seq = current + step_size * direction
    new_seq = np.clip(new_seq, 0, 1000)
    
    # If new sequence is better, return it
    new_fitness = evaluate_sequence(list(new_seq))
    if new_fitness > best_fitness:
        return list(new_seq)
    
    return list(current)

def population_refinement(population, fitness_scores, elite_fraction=0.2):
    """Refine population by preserving elites and improving others."""
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite_count = int(len(population) * elite_fraction)
    
    # Preserve elites
    elite_pop = [population[i] for i in sorted_indices[:elite_count]]
    
    # Improve remaining members
    refined_pop = elite_pop[:]
    for i in range(elite_count, len(population)):
        idx = sorted_indices[i]
        member = population[idx]
        # Apply local improvement
        improved_member, _ = local_improvement_search(member, max_iter=20)
        refined_pop.append(improved_member)
    
    return refined_pop

def local_improvement_search(initial_sequence, max_iter=100):
    """Improved local search with gradient estimation and simulated annealing."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    
    # Simulated annealing parameters
    temp = 1.0
    cooling_rate = 0.95
    min_temp = 1e-4
    
    for iteration in range(max_iter):
        # Try mutating the current sequence
        mutated = smart_mutate(current_sequence, mutation_rate=0.3, max_mutation=0.2)
        mutated_fitness = evaluate_sequence(mutated)
        
        # Accept or reject based on fitness gain
        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness
        else:
            # Accept with probability based on temperature
            if random.random() < np.exp((mutated_fitness - current_fitness) / (temp + 1e-10)):
                current_sequence = mutated
                current_fitness = mutated_fitness
        
        # Update best
        if current_fitness > best_fitness:
            best_sequence = current_sequence.copy()
            best_fitness = current_fitness
        
        # Cool down temperature
        temp = max(temp * cooling_rate, min_temp)
        
    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main search function using hybrid approach."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    max_time_seconds = 180
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Strategy 1: Direct evolutionary search with adaptive techniques
    if time.time() - start_time < max_time_seconds:
        # Initialize population
        pop_size = 50
        population = [generate_random_sequence() for _ in range(pop_size)]
        
        # Evolution loop
        generation = 0
        while time.time() - start_time < max_time_seconds and generation < 1000:
            # Evaluate fitness
            fitness_scores = [evaluate_sequence(seq) for seq in population]
            
            # Track best
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_inv_c1:
                best_inv_c1 = fitness_scores[current_best_idx]
                best_sequence = population[current_best_idx].copy()
            
            # Refine population
            population = population_refinement(population, fitness_scores)
            
            # Create new generation
            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = population[i]
                parent2 = population[(i + 1) % pop_size]
                
                # Crossover
                child1 = adaptive_crossover(parent1, parent2)
                child2 = adaptive_crossover(parent2, parent1)
                
                # Mutate
                child1 = smart_mutate(child1)
                child2 = smart_mutate(child2)
                
                # Possibly enhance with gradient
                enhanced1 = estimate_gradient_and_improve(child1, generation)
                enhanced2 = estimate_gradient_and_improve(child2, generation)
                
                new_population.extend([enhanced1, enhanced2])
            
            population = new_population[:pop_size]
            generation += 1
    
    # Strategy 2: Local search refinement of the best found
    if best_sequence is not None and time.time() - start_time < max_time_seconds:
        refined_seq, refined_fitness = local_improvement_search(best_sequence, 500)
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