# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
import random
import time

def convolve_fft(a, b):
    """Compute convolution using FFT for better performance."""
    n = len(a)
    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n]

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')

    # Compute convolution using FFT
    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence):
    """Evaluate fitness as inverse of C1 (higher is better)"""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def generate_initial_sequence(n):
    """Generate a good initial sequence based on exponential decay."""
    sequence = []
    for i in range(n):
        # Exponential decay with some noise to break symmetry
        base_val = max(0.01, 100 * np.exp(-i * 0.05))
        noise = random.uniform(0.9, 1.1)
        sequence.append(base_val * noise)
    return sequence

def local_search_refinement(sequence, max_iterations=20):
    """Apply local search refinement to improve sequence."""
    best_seq = sequence.copy()
    best_fitness = evaluate_fitness(best_seq)

    for _ in range(max_iterations):
        # Try small perturbations
        mutated = best_seq.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance to mutate each element
                mutated[i] *= random.uniform(0.9, 1.1)  # Small multiplicative change
                mutated[i] = max(0.01, mutated[i])  # Ensure non-negative

        mutated_fitness = evaluate_fitness(mutated)
        if mutated_fitness > best_fitness:
            best_seq = mutated
            best_fitness = mutated_fitness

    return best_seq

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses FFT-based convolution and local search optimization.
    """
    # Configuration
    max_time = 175  # Leave some time for cleanup
    start_time = time.time()

    # Start with a good initial guess
    n = random.randint(100, 1000)
    initial_sequence = generate_initial_sequence(n)

    # Apply local search refinement
    optimized_sequence = local_search_refinement(initial_sequence)

    # Continue optimizing until time runs out
    while time.time() - start_time < max_time:
        # Try to improve further with another round of local search
        improved_sequence = local_search_refinement(optimized_sequence)
        if evaluate_fitness(improved_sequence) > evaluate_fitness(optimized_sequence):
            optimized_sequence = improved_sequence
        else:
            break  # No improvement, stop optimizing

    return optimized_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")