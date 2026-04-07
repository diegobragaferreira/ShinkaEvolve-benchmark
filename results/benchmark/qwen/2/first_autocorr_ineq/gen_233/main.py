# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import random
import time
from typing import List, Tuple

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_convolution_fft(a: np.ndarray) -> np.ndarray:
    """Compute convolution using FFT for efficiency."""
    n = len(a)
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(a, padded_length)
    result = ifft(fa * np.conj(fb)).real
    return result[:n]

def compute_c1_constant(sequence: List[float]) -> float:
    """Compute the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return 0.0
    
    conv = compute_convolution_fft(a)
    max_conv = np.max(conv)
    n = len(a)
    
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def spectral_filter_sequence(sequence: List[float], num_bands: int = 5) -> List[float]:
    """
    Apply spectral filtering to emphasize certain frequency components.
    This helps in creating sequences with more favorable convolution properties.
    """
    n = len(sequence)
    if n < 1:
        return sequence
    
    # Compute the DFT of the sequence
    fft_coeffs = fft(sequence)
    power_spectrum = np.abs(fft_coeffs)**2
    
    # Create band-pass filters for different frequency ranges
    filtered_spectrum = np.zeros(n)
    
    # Divide frequencies into bands
    band_width = n // num_bands
    
    # Create a more selective filter that emphasizes middle frequencies
    for i in range(num_bands):
        start = i * band_width
        end = min((i + 1) * band_width, n)
        if i == num_bands // 2:  # Emphasize middle band
            filtered_spectrum[start:end] = 1.0
        else:
            filtered_spectrum[start:end] = 0.2  # Less emphasis on others
    
    # Smooth the filter
    smoothed_filter = np.convolve(filtered_spectrum, np.ones(5)/5, mode='same')
    
    # Apply filter in frequency domain
    filtered_fft = fft_coeffs * smoothed_filter
    filtered_seq = np.real(ifft(filtered_fft))
    
    # Ensure non-negativity and renormalize
    filtered_seq = np.maximum(filtered_seq, 0)
    sum_filtered = np.sum(filtered_seq)
    if sum_filtered > 0:
        filtered_seq = filtered_seq / sum_filtered * 10  # Scale appropriately
    
    # Clip to bounds
    filtered_seq = np.clip(filtered_seq, 0, 1000)
    
    return filtered_seq.tolist()

def create_initial_population(size: int, min_len: int = 100, max_len: int = 1000) -> List[List[float]]:
    """Generate diverse initial population using spectral techniques."""
    population = []
    for _ in range(size):
        # Random length
        n = random.randint(min_len, max_len)
        
        # Generate base sequence
        base_seq = np.random.exponential(scale=1.0, size=n)
        
        # Apply spectral filtering
        filtered_seq = spectral_filter_sequence(base_seq.tolist())
        
        # Normalize
        sum_seq = np.sum(filtered_seq)
        if sum_seq > 0:
            filtered_seq = [x / sum_seq for x in filtered_seq]
        
        population.append(filtered_seq)
    
    return population

def tournament_selection(population: List[List[float]], fitnesses: List[float], k: int = 3) -> List[float]:
    """Select individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx][:]

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform uniform crossover between two sequences."""
    min_len = min(len(parent1), len(parent2))
    
    # Create child with mix of both parents
    child = []
    for i in range(min_len):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # Handle different lengths
    if len(parent1) > min_len:
        child.extend(parent1[min_len:])
    elif len(parent2) > min_len:
        child.extend(parent2[min_len:])
    
    # Normalize
    sum_child = np.sum(child)
    if sum_child > 0:
        child = [x / sum_child for x in child]
    
    return child

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation with spectral guidance."""
    mutated = sequence.copy()
    n = len(mutated)
    
    # Determine number of mutations
    num_mutations = max(1, int(n * mutation_rate))
    
    for _ in range(num_mutations):
        # Choose index to mutate
        idx = random.randint(0, n - 1)
        # Apply multiplicative mutation
        delta = random.uniform(0.8, 1.2)
        mutated[idx] *= delta
        # Ensure non-negativity
        mutated[idx] = max(0, mutated[idx])
    
    # Re-normalize
    sum_mut = np.sum(mutated)
    if sum_mut > 0:
        mutated = [x / sum_mut for x in mutated]
        
    return mutated

def local_improve(sequence: List[float], max_iter: int = 10) -> List[float]:
    """Local improvement using spectral filtering and minor adjustments."""
    current = sequence[:]
    
    # Apply spectral filtering for smoothing
    filtered_seq = spectral_filter_sequence(current)
    
    # Normalize
    sum_filtered = np.sum(filtered_seq)
    if sum_filtered > 0:
        filtered_seq = [x / sum_filtered for x in filtered_seq]
    
    # Evaluate improvement
    current_fitness = compute_c1_constant(current)
    filtered_fitness = compute_c1_constant(filtered_seq)
    
    if filtered_fitness > current_fitness:
        current = filtered_seq[:]
    
    # Make small random adjustments
    for _ in range(max_iter):
        if random.random() < 0.5:
            mutated = mutate_sequence(current)
            mutated_fitness = compute_c1_constant(mutated)
            if mutated_fitness > current_fitness:
                current = mutated[:]
                current_fitness = mutated_fitness
    
    return current

def spectral_evolutionary_search(max_time_seconds: int = 180) -> List[float]:
    """Execute the spectral evolutionary optimization."""
    start_time = time.time()
    
    # Parameters
    pop_size = 30
    generations = 50
    elite_size = 6
    mutation_rate = 0.1
    
    # Initialize population
    population = create_initial_population(pop_size)
    
    best_individual = None
    best_fitness = 0.0
    
    for gen in range(generations):
        if time.time() - start_time > max_time_seconds - 2:
            break
            
        # Evaluate fitness
        fitnesses = [compute_c1_constant(seq) for seq in population]
        
        # Track best
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx][:]
        
        # Elitism
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elite_individuals = [population[i][:] for i in elite_indices]
        
        # Create new population
        new_population = elite_individuals[:]
        
        # Fill with offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > max_time_seconds - 2:
                break
                
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutated_child = mutate_sequence(child, mutation_rate)
            
            # Local improvement
            improved_child = local_improve(mutated_child)
            
            new_population.append(improved_child)
        
        population = new_population[:pop_size]
    
    # Final local search on best individual
    if best_individual is not None:
        best_individual = local_improve(best_individual, 50)
        best_fitness = compute_c1_constant(best_individual)
    
    # Return best found
    return best_individual if best_individual is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()
    
    # Run spectral evolutionary search
    best_sequence = spectral_evolutionary_search(170)
    
    # Final check
    final_fitness = compute_c1_constant(best_sequence)
    if final_fitness <= 0.0:
        # Fallback: create a random sequence
        n = random.randint(100, 1000)
        return [random.uniform(0.1, 1.0) for _ in range(n)]
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")