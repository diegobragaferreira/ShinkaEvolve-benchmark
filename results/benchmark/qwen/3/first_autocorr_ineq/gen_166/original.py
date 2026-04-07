# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from functools import partial
import warnings
import math

warnings.filterwarnings('ignore')

def convolve_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Use scipy's fftconvolve for better numerical stability
    conv = fftconvolve(seq, seq, mode='full')
    return conv[:2*n - 1]

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

def create_structured_individual(n):
    """Create a structured individual to boost exploration."""
    individual = []
    # Geometric decay with a bit of randomness
    decay_factor = 0.85
    for i in range(n):
        if i % 5 == 0:
            # Sparse structure with decay
            individual.append(1000 * (decay_factor ** (i // 5)))
        else:
            # Random elements
            individual.append(random.uniform(0, 1000))
    return individual

def create_random_individual(n):
    """Create a purely random individual."""
    return [random.uniform(0, 1000) for _ in range(n)]

def create_individual(n):
    """Create a new individual with given length."""
    # Hybrid approach: 60% structured, 40% random
    if random.random() < 0.6:
        return create_structured_individual(n)
    else:
        return create_random_individual(n)

def mutate_individual(individual, indpb, mut_strength, generation, max_generations):
    """Mutate an individual with adaptive parameters."""
    # Adaptive mutation rate that decreases with generations
    adaptive_indpb = indpb * (1.0 - generation / max_generations)
    
    # Adaptive mutation strength
    adaptive_mut_strength = mut_strength * (1.0 - generation / max_generations)
    
    for i in range(len(individual)):
        if random.random() < adaptive_indpb:
            # Gaussian mutation with adaptive strength
            individual[i] += random.gauss(0, adaptive_mut_strength)
            individual[i] = max(0, individual[i])  # Ensure non-negativity
    return individual,

def crossover_individuals(ind1, ind2, cxpb):
    """Crossover two individuals."""
    if random.random() < cxpb:
        # Uniform crossover
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def gradient_update(sequence, learning_rate=0.01, max_iterations=5):
    """Improve a sequence using gradient ascent."""
    try:
        seq = np.array(sequence, dtype=float)
        n = len(seq)
        if n < 1:
            return sequence
            
        # Simple gradient approximation using finite differences
        def compute_c1_and_grad(seq):
            conv = convolve_fft(seq)
            max_conv = np.max(conv)
            sum_seq = np.sum(seq)
            
            if sum_seq < 1e-10:
                return float('inf'), np.zeros_like(seq)
                
            c1 = 2 * n * max_conv / (sum_seq ** 2)
            
            # Gradient estimation (simplified)
            eps = 1e-6
            grad = np.zeros_like(seq)
            for i in range(n):
                seq_eps = seq.copy()
                seq_eps[i] += eps
                c1_plus = compute_c1_value(seq_eps)
                grad[i] = (c1_plus - c1) / eps
                
            return c1, grad
            
        # Perform gradient ascent
        for _ in range(max_iterations):
            _, grad = compute_c1_and_grad(seq)
            seq -= learning_rate * grad
            seq = np.maximum(seq, 0)  # Ensure non-negativity
            
        return seq.tolist()
    except Exception as e:
        return sequence

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimize the sequence using adaptive hybrid method with gradient updates."""
    try:
        # Determine sequence length
        n = len(sequence)
        if n < 10:
            n = 100  # Default small size
        elif n > 1000:
            n = 1000  # Cap maximum size

        # Set up evolutionary algorithm parameters
        pop_size = max(40, min(120, n // 2))
        gen_limit = max(10, min(50, n // 6))

        # Initial population
        population = []
        for _ in range(pop_size):
            ind = create_individual(n)
            population.append(ind)

        # Evaluate initial population
        fitnesses = [evaluate_individual(ind)[0] for ind in population]

        # Evolution loop with gradient refinement
        for generation in range(gen_limit):
            # Selection
            selected_indices = np.argsort(fitnesses)[::-1][:pop_size // 2]
            selected_population = [population[i] for i in selected_indices]
            
            # Generate offspring with crossover and mutation
            offspring = []
            while len(offspring) < pop_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = crossover_individuals(parent1, parent2, 0.6)
                
                # Mutate children
                child1 = mutate_individual(child1, 0.1, 50.0, generation, gen_limit)[0]
                child2 = mutate_individual(child2, 0.1, 50.0, generation, gen_limit)[0]
                
                offspring.extend([child1, child2])
            
            # Keep only required number
            offspring = offspring[:pop_size]
            
            # Gradient-based refinement
            refined_offspring = []
            for ind in offspring:
                refined = gradient_update(ind, 0.005, 3)
                refined_offspring.append(refined)
            
            # Evaluate refined offspring
            fitnesses = [evaluate_individual(ind)[0] for ind in refined_offspring]
            
            # Replace population
            population = refined_offspring[:]
            if len(population) != pop_size:
                population.extend([create_individual(n) for _ in range(pop_size - len(population))])

        # Get best individual
        best_fitness = max(fitnesses)
        best_ind = population[fitnesses.index(best_fitness)]
        return best_ind

    except Exception as e:
        print(f"Error in adaptive optimization: {e}")
        # Return mutated version of input if evolution fails
        return [(x + random.uniform(-100, 100)) for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initialize with a random sequence of moderate size
    n = random.randint(100, 1000)
    sequence = [random.uniform(0, 1000) for _ in range(n)]

    # Apply hybrid optimization
    optimized_sequence = get_good_direction_to_move_into(sequence)

    # Ensure minimum sum constraint
    if sum(optimized_sequence) < 0.01:
        optimized_sequence = [x + random.uniform(0, 1) for x in optimized_sequence]

    return optimized_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")