# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve
import time

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return float('inf')

    # Compute autocorrelation using FFT for efficiency
    autocorr = fftconvolve(sequence, sequence[::-1], mode='full')
    # Take the second half which corresponds to the actual autocorrelation
    autocorr = autocorr[len(sequence)-1:]
    max_autocorr = np.max(autocorr)

    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    # C₁ = 2n * max(autocorr) / (sum(sequence))^2
    C1 = 2 * len(sequence) * max_autocorr / sum_sq

    return C1

def objective_function(sequence):
    """
    Objective function to minimize (negative of inverse of C₁).
    """
    C1 = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return float('inf')
    return -1.0 / C1

def generate_initial_sequence(n_min, n_max):
    """
    Generate a good initial sequence with structured properties.
    """
    n = np.random.randint(n_min, n_max)
    # Generate a structured sequence that might work well
    sequence = np.zeros(n)
    # Use a combination of exponential decay and sine wave for better structure
    for i in range(n):
        sequence[i] = np.exp(-i / (n / 3)) * (1 + 0.5 * np.sin(np.pi * i / n))
    # Normalize to ensure sum > 0.01
    sequence = sequence * (0.01 / (np.sum(sequence) + 1e-10))
    return sequence

def local_refinement(sequence, max_iter=20):
    """
    Apply local refinement to improve the sequence.
    """
    current = np.array(sequence)
    for _ in range(max_iter):
        # Simple gradient ascent: adjust each element slightly
        gradients = np.zeros_like(current)
        epsilon = 1e-4
        for i in range(len(current)):
            # Forward difference approximation
            perturbed = current.copy()
            perturbed[i] += epsilon
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negative
            val_plus = objective_function(perturbed)
            val_current = objective_function(current)
            gradients[i] = (val_plus - val_current) / epsilon

        # Update sequence using gradient ascent
        learning_rate = 0.01 * np.std(gradients) + 1e-6
        current += learning_rate * gradients
        current = np.maximum(current, 0)  # Ensure non-negative

        # Normalize
        sum_current = np.sum(current)
        if sum_current > 0:
            current = current * (0.01 / sum_current)

    return current.tolist()

def evolve_sequences():
    """
    Evolutionary optimization for finding optimal sequence with local refinement.
    """
    # Parameters
    popsize = 20
    maxiter = 100
    n_min, n_max = 100, 1000
    seed = 42

    np.random.seed(seed)

    # Start with a good initial sequence
    initial_seq = generate_initial_sequence(n_min, n_max)

    # Define bounds for each element in the sequence
    bounds = [(0, 1000) for _ in range(len(initial_seq))]

    # Run differential evolution optimization
    result = differential_evolution(
        objective_function,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        disp=False,
        strategy='best1bin'
    )

    best_sequence = result.x if result.success else initial_seq

    # Apply local refinement to improve the solution
    refined_sequence = local_refinement(best_sequence)

    return refined_sequence

def search_for_best_sequence():
    """
    Main function to find the best sequence by evolutionary optimization with refinement.
    """
    start_time = time.time()

    # Run the evolutionary optimization
    best_sequence = evolve_sequences()

    # Ensure the sequence meets minimum requirements
    if np.sum(best_sequence) < 0.01:
        # Reinitialize if necessary
        n = np.random.randint(100, 1000)
        best_sequence = np.random.rand(n)
        best_sequence = best_sequence * (0.01 / (np.sum(best_sequence) + 1e-10))

    end_time = time.time()

    return list(best_sequence)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")