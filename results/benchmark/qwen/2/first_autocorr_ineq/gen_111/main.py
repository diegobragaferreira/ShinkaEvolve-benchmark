# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
import time
from collections import deque
import math

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """
    Compute C1 for a given sequence.
    C1 = 2*n*max(convolution) / (sum(sequence))^2
    We want to maximize 1/C1, which means minimizing C1.
    """
    if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
        return float('inf')

    # Use FFT-based convolution for efficiency
    convolved = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved[len(sequence)-1:])  # Only consider relevant part
    sum_seq = sum(sequence)

    # Return C1 value
    return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def compute_convolution_profile(sequence):
    """Compute the convolution profile to understand structure."""
    conv = fftconvolve(sequence, sequence, mode='full')
    return conv[len(sequence)-1:]

def evaluate_sequence_with_convolution(sequence):
    """Evaluate sequence including convolution information."""
    inv_c1 = compute_inv_c1(sequence)
    conv_profile = compute_convolution_profile(sequence)
    max_conv = np.max(conv_profile)
    total_mass = np.sum(sequence)
    
    # Return tuple of (inverse_c1, max_conv, total_mass) 
    # to inform selection based on both C1 and convolution shape
    return (inv_c1, max_conv, total_mass)

def create_initial_population(pop_size, min_length=50, max_length=500):
    """Create diverse initial population using multiple strategies."""
    population = []
    for _ in range(pop_size):
        # Random length
        n = random.randint(min_length, max_length)
        
        # Strategy 1: Exponential decay
        if random.random() < 0.4:
            decay_factor = 0.95
            seq = [1.0 * (decay_factor ** i) for i in range(n)]
        # Strategy 2: Uniform
        elif random.random() < 0.3:
            seq = [1.0] * n
        # Strategy 3: Step function
        elif random.random() < 0.2:
            half = n // 2
            seq = [1.0] * half + [0.0] * (n - half)
        # Strategy 4: Random with bias
        else:
            seq = [random.uniform(0, 100) for _ in range(n)]
            
        # Add noise and ensure minimal mass
        seq = [max(x, 0.01) for x in seq]
        noise_factor = 0.05
        seq = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in seq]
        seq = [max(x, 0.01) for x in seq]
        
        # Clip to valid range
        seq = [min(x, 1000) for x in seq]
        population.append(seq)
    
    return population

def crossover(parent1, parent2):
    """Perform crossover between two sequences."""
    n1, n2 = len(parent1), len(parent2)
    min_len = min(n1, n2)
    max_len = max(n1, n2)
    
    # Choose crossover point
    if min_len > 1:
        cx_point = random.randint(1, min_len - 1)
    else:
        cx_point = 1
    
    # Create offspring
    if n1 >= n2:
        child = parent1[:cx_point] + parent2[cx_point:]
    else:
        child = parent1[:cx_point] + parent2[cx_point:]
    
    # Ensure correct length
    if len(child) < max_len:
        child.extend([0.0] * (max_len - len(child)))
    elif len(child) > max_len:
        child = child[:max_len]
        
    return child

def mutate(sequence, mutation_rate=0.1):
    """Mutate a sequence with probability mutation_rate per element."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            change = random.uniform(-0.1, 0.1)
            mutated[i] = max(0.0, mutated[i] + change * mutated[i])
    return mutated

def select_parents(population, fitnesses, tournament_size=3):
    """Tournament selection."""
    selected = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])
    return selected

def refine_sequence_convolution_aware(sequence):
    """Refine a sequence by flattening high convolution peaks."""
    seq = np.array(sequence, dtype=float)
    conv = compute_convolution_profile(seq)
    max_conv = np.max(conv)
    
    if max_conv <= 0:
        return sequence
        
    # Identify indices contributing to high convolution
    max_indices = np.where(conv >= 0.9 * max_conv)[0]
    
    # Reduce values around high convolution indices  
    for idx in max_indices[:min(5, len(max_indices))]:
        for offset in [-2, -1, 0, 1, 2]:
            pos = idx + offset
            if 0 <= pos < len(seq):
                seq[pos] *= 0.98
    
    # Apply clipping and minimal mass enforcement
    seq = np.clip(seq, 0, 1000)
    if np.sum(seq) < 0.01:
        seq[0] = 0.1
        
    return seq.tolist()

def evolutionary_convolution_guided_search(max_generations=100, pop_size=30):
    """Main evolutionary search with convolution guidance."""
    population = create_initial_population(pop_size)
    best_sequence = None
    best_inv_c1 = 0
    generation_history = deque(maxlen=20)
    
    for generation in range(max_generations):
        # Evaluate fitness for current generation
        fitness_results = [evaluate_sequence_with_convolution(seq) for seq in population]
        fitness_scores = [result[0] for result in fitness_results]  # inverse C1
        
        # Update best
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]
        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx][:]
        
        # Store history
        generation_history.append(current_best_inv_c1)
        
        # Early stopping if no improvement
        if len(generation_history) >= 20:
            if all(x <= generation_history[-1] for x in list(generation_history)[:-5]):
                break
                
        # Selection
        parents = select_parents(population, fitness_scores)
        
        # Create new population through crossover and mutation
        new_population = []
        elite_count = pop_size // 5  # Keep top 20%
        
        # Elitism: keep best sequences
        sorted_indices = np.argsort(fitness_scores)[::-1]
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]])
            
        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.1)
            
            # Convolution-aware refinement
            child = refine_sequence_convolution_aware(child)
            
            new_population.append(child)
            
        population = new_population[:pop_size]
        
    return best_sequence if best_sequence is not None else population[0]

def search_for_best_sequence():
    """Main entry point for searching the best sequence."""
    start_time = time.time()
    
    # Perform evolutionary search
    best_sequence = evolutionary_convolution_guided_search(max_generations=80, pop_size=25)
    
    # Final refinement
    final_sequence = refine_sequence_convolution_aware(best_sequence)
    
    # Verify and return
    if compute_inv_c1(final_sequence) > compute_inv_c1(best_sequence):
        return final_sequence
    else:
        return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
