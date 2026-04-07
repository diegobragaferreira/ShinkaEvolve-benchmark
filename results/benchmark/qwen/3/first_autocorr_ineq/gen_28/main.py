# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Optional
import time

def compute_c1_from_convolution(conv_result: np.ndarray, sequence_length: int, sum_sequence: float) -> float:
    """Compute C1 from the convolution result."""
    max_conv = np.max(conv_result)
    if sum_sequence <= 0:
        return float('inf')
    c1 = 2 * sequence_length * max_conv / (sum_sequence ** 2)
    return c1

def inv_c1_from_convolution(conv_result: np.ndarray, sequence_length: int, sum_sequence: float) -> float:
    """Compute inverse C1 from the convolution result."""
    max_conv = np.max(conv_result)
    if max_conv <= 0 or sum_sequence <= 0:
        return 0.0
    inv_c1 = (sum_sequence ** 2) / (2 * sequence_length * max_conv)
    return inv_c1

def fft_convolve_sequence(sequence: np.ndarray) -> np.ndarray:
    """Compute convolution using FFT for efficiency."""
    # Pad to avoid circular convolution effects
    n = len(sequence)
    padded_length = 2 * n - 1
    padded_seq = np.pad(sequence, (0, padded_length - n), 'constant')
    # FFT-based convolution
    fft_seq = np.fft.fft(padded_seq)
    conv_fft = fft_seq * np.conj(fft_seq)
    conv_result = np.fft.ifft(conv_fft).real[:padded_length]
    return conv_result

def generate_initial_population(pop_size: int, min_length: int = 100, max_length: int = 1000) -> List[np.ndarray]:
    """Generate initial population of sequences."""
    population = []
    for _ in range(pop_size):
        length = random.randint(min_length, max_length)
        # Generate sequence with random heights, clipped to [0, 1000]
        seq = np.random.uniform(0, 1000, length)
        population.append(seq)
    return population

def evaluate_individual(individual: np.ndarray) -> tuple[float, float]:
    """Evaluate an individual: returns (inv_c1, c1)."""
    if np.sum(individual) < 0.01:
        return 0.0, float('inf')
    conv_result = fft_convolve_sequence(individual)
    sum_seq = np.sum(individual)
    sequence_length = len(individual)
    inv_c1 = inv_c1_from_convolution(conv_result, sequence_length, sum_seq)
    c1 = compute_c1_from_convolution(conv_result, sequence_length, sum_seq)
    return inv_c1, c1

def selection(population: List[np.ndarray], fitness_scores: List[tuple[float, float]]) -> List[np.ndarray]:
    """Select top individuals for reproduction."""
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][0], reverse=True)
    selected_count = max(1, len(population) // 2)
    selected = [population[i] for i in sorted_indices[:selected_count]]
    return selected

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover between two parents."""
    length = min(len(parent1), len(parent2))
    child = np.copy(parent1)
    mask = np.random.rand(length) > 0.5
    child[mask] = parent2[mask]
    return child

def mutate(individual: np.ndarray, mutation_rate: float = 0.1, max_mutation: float = 100.0) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = np.copy(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise
            noise = np.random.normal(0, max_mutation)
            mutated[i] = max(0, mutated[i] + noise)
    return mutated

def evolve_population(population: List[np.ndarray], generations: int = 20) -> List[np.ndarray]:
    """Evolve population over several generations."""
    pop_size = len(population)
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_individual(ind) for ind in population]
        
        # Selection
        selected = selection(population, fitness_scores)
        
        # Create new population
        new_population = selected.copy()
        
        while len(new_population) < pop_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child = crossover(parent1, parent2)
            child = mutate(child)
            
            # Clip values
            child = np.clip(child, 0, 1000)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return population

def get_good_direction_to_move_into(
    sequence: List[float],
) -> Optional[List[float]]:
    """Improve the sequence using evolutionary spectral optimization."""
    if len(sequence) < 1:
        return None
    
    # Convert to numpy array and clip values
    sequence = np.array(sequence)
    sequence = np.clip(sequence, 0, 1000)
    
    # Try to find a better sequence through evolutionary search
    try:
        # Generate initial population
        population = generate_initial_population(20)
        # Include the current sequence as one of the individuals
        population.append(sequence)
        
        # Evolve the population
        evolved_population = evolve_population(population, 10)
        
        # Evaluate the evolved population
        fitness_scores = [evaluate_individual(ind) for ind in evolved_population]
        
        # Find the best individual
        best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i][0])
        best_individual = evolved_population[best_idx]
        
        # Ensure sum is not too small
        if np.sum(best_individual) < 0.01:
            return None
            
        return best_individual.tolist()
        
    except Exception as e:
        print(f"Error during evolution: {e}")
        return sequence

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    # Start with a random sequence
    n = random.randint(100, 1000)
    sequence = [random.uniform(0, 1000) for _ in range(n)]
    
    # Use evolutionary approach to improve
    improved_sequence = get_good_direction_to_move_into(sequence)
    
    if improved_sequence is not None:
        return improved_sequence
    else:
        return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
