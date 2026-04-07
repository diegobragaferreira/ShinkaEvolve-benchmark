# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
import time
from scipy.optimize import minimize
import cmath

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

def fourier_guided_mutation(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate sequence using fourier domain insights."""
    # Transform to frequency domain to analyze spectral properties
    n = len(sequence)
    freq_domain = np.fft.fft(sequence)
    magnitude_spectrum = np.abs(freq_domain)
    
    # Determine dominant frequencies and modify accordingly
    dominant_freq_idx = np.argsort(magnitude_spectrum)[-3:]  # Top 3 frequencies
    
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Modify based on frequency domain characteristics
            freq_contrib = 0
            if i < len(dominant_freq_idx):
                freq_contrib = magnitude_spectrum[dominant_freq_idx[i % len(dominant_freq_idx)]]
            
            # Adjust mutation factor based on spectral dominance
            mutation_factor = 1.0 + (freq_contrib / (np.max(magnitude_spectrum) + 1e-8) - 0.5) * max_mutation
            
            new_sequence[i] *= random.uniform(1 - max_mutation * mutation_factor, 1 + max_mutation * mutation_factor)
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    return new_sequence

def fourier_guided_crossover(seq1, seq2):
    """Perform crossover guided by fourier domain properties."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    # Pad sequences to same length
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))
    
    # Compute frequency domains
    fft1 = np.fft.fft(padded_seq1)
    fft2 = np.fft.fft(padded_seq2)
    
    # Combine in frequency domain
    mixed_fft = 0.5 * (fft1 + fft2)
    mixed_time = np.fft.ifft(mixed_fft).real
    
    # Return as sequence
    return [max(0, min(1000, val)) for val in mixed_time]

def fourier_guided_local_search(initial_sequence, max_iter=100):
    """Improve a sequence using fourier-guided local search."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    
    for _ in range(max_iter):
        # Try fourier-guided mutation
        mutated = fourier_guided_mutation(current_sequence)
        mutated_fitness = evaluate_sequence(mutated)
        
        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness
        else:
            # Try standard mutation if no improvement
            standard_mutated = [x * random.uniform(0.9, 1.1) for x in current_sequence]
            standard_mutated = [max(0, min(1000, x)) for x in standard_mutated]
            standard_fitness = evaluate_sequence(standard_mutated)
            if standard_fitness > current_fitness:
                current_sequence = standard_mutated
                current_fitness = standard_fitness
    
    return current_sequence, current_fitness

def search_for_best_sequence():
    """Main search function using Fourier-guided evolutionary approach."""
    start_time = time.time()
    max_time_seconds = 180
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try multiple initialization strategies
    for attempt in range(10):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Initialize with different patterns
        n = random.randint(100, 500)
        
        # Try various initialization methods
        init_methods = [
            generate_random_sequence(n),
            np.random.uniform(0.01, 100, n).tolist(),
            np.ones(n) * 10,
            [1.0] * (n // 2) + [0.0] * (n - n // 2)
        ]
        
        for init_seq in init_methods:
            # Perform fourier-guided local search
            optimized_seq, fitness = fourier_guided_local_search(init_seq, 100)
            
            if fitness > best_inv_c1:
                best_inv_c1 = fitness
                best_sequence = optimized_seq[:]

    # Final refinement
    if best_sequence is not None:
        final_seq, final_fitness = fourier_guided_local_search(best_sequence, 300)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq
    
    # If nothing found, fallback to random sequence
    if best_sequence is None:
        best_sequence = generate_random_sequence()
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")