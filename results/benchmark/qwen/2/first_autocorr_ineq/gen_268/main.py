# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time
import random
from typing import List, Tuple

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    # Use FFT-based convolution for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution (the peak)
    max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards
    # Normalize and compute C1
    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')
    c1 = (2 * len(sequence) * max_conv) / sum_sq
    return c1

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def fft_mutation(freq_domain: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
    """
    Mutate a frequency domain representation by adding noise.
    This preserves the structural properties of the frequency domain.
    """
    # Add complex noise to frequency components
    noise_real = np.random.normal(0, mutation_strength, len(freq_domain))
    noise_imag = np.random.normal(0, mutation_strength, len(freq_domain))
    mutated_freq = freq_domain + noise_real + 1j * noise_imag
    
    # Transform back to time domain
    mutated_time = np.fft.ifft(mutated_freq).real
    return np.abs(mutated_time)  # Ensure non-negative

def fft_crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> np.ndarray:
    """Perform crossover in frequency domain."""
    if random.random() > crossover_rate:
        return parent1.copy()
    
    # Convert to frequency domain
    freq1 = np.fft.fft(parent1)
    freq2 = np.fft.fft(parent2)
    
    # Blend in frequency domain
    blend_factor = random.uniform(0.3, 0.7)
    crossed_freq = blend_factor * freq1 + (1 - blend_factor) * freq2
    
    # Transform back to time domain
    crossed_time = np.fft.ifft(crossed_freq).real
    return np.abs(crossed_time)  # Ensure non-negative

def evaluate_population(population: List[np.ndarray]) -> List[float]:
    """Evaluate the fitness (inverse C1) of a population."""
    return [compute_inv_c1(seq.tolist()) for seq in population]

def select_parents(population: List[np.ndarray], fitnesses: List[float], num_parents: int = 5) -> List[np.ndarray]:
    """Select top performers as parents for reproduction."""
    sorted_indices = np.argsort(fitnesses)[::-1][:num_parents]
    return [population[i] for i in sorted_indices]

def evolve_sequence(population_size: int = 20, generations: int = 50, mutation_strength: float = 0.1) -> np.ndarray:
    """Evolve a sequence using a population-based approach in frequency domain."""
    # Initialize population with diverse sequences
    population = []
    for _ in range(population_size):
        length = random.randint(100, 500)
        # Use exponential decay pattern as base
        decay_factor = 0.95
        seq = [1.0 * (decay_factor ** i) for i in range(length)]
        # Ensure minimum values
        seq = [max(x, 0.01) for x in seq]
        # Add slight noise
        noise_factor = 0.05
        seq = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in seq]
        seq = [max(x, 0.01) for x in seq]
        population.append(np.array(seq))
    
    # Memory to store best sequences
    memory = []
    best_fitness = float('-inf')
    best_sequence = None
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = evaluate_population(population)
        
        # Update memory and best sequence
        for i, (seq, fit) in enumerate(zip(population, fitnesses)):
            if fit > best_fitness:
                best_fitness = fit
                best_sequence = seq.copy()
            if len(memory) < 10:
                memory.append((seq.copy(), fit))
            else:
                # Replace worst in memory
                worst_mem_idx = np.argmin([f for _, f in memory])
                if fit > memory[worst_mem_idx][1]:
                    memory[worst_mem_idx] = (seq.copy(), fit)
        
        # Selection
        parents = select_parents(population, fitnesses)
        
        # Generate new population
        new_population = parents[:]  # Elitism
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = fft_crossover(parent1, parent2)
            
            # Mutate in frequency domain
            mutated_child = fft_mutation(child, mutation_strength)
            
            # Project onto feasible set
            mutated_child = np.clip(mutated_child, 0, 1000)
            if np.sum(mutated_child) < 0.01:
                mutated_child[0] = 0.1
            
            new_population.append(mutated_child)
        
        population = new_population
    
    return best_sequence

def adaptive_refinement(sequence: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """Refine the sequence by analyzing convolution properties."""
    seq = sequence.copy()
    
    for _ in range(max_iterations):
        # Compute convolution
        conv = fftconvolve(seq, seq, mode='full')
        conv_part = conv[len(seq)-1:]
        max_conv = np.max(conv_part)
        
        # Find indices where convolution peaks
        max_indices = np.where(conv_part >= 0.9 * max_conv)[0]
        
        # Adjust elements that contribute to peaks
        new_seq = seq.copy()
        for idx in max_indices[:min(3, len(max_indices))]:
            for offset in [-2, -1, 0, 1, 2]:
                pos = idx + offset
                if 0 <= pos < len(seq):
                    new_seq[pos] *= 0.995  # Fine adjustment
                    
        seq = np.clip(new_seq, 0, 1000)
        
        # Early stopping if sequence stabilizes
        if np.allclose(seq, new_seq, rtol=1e-6):
            break
    
    return seq

def search_for_best_sequence() -> list[float]:
    """Main search function using the frequency-domain evolutionary optimizer."""
    start_time = time.time()
    
    # Evolve a sequence
    evolved_sequence = evolve_sequence(population_size=15, generations=30, mutation_strength=0.05)
    
    # Final refinement using adaptive methods
    refined_sequence = adaptive_refinement(evolved_sequence)
    
    # Final evaluation
    final_inv_c1 = compute_inv_c1(refined_sequence.tolist())
    
    # Return the best performing sequence
    return refined_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")