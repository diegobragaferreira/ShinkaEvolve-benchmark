# EVOLVE-BLOCK-START

import numpy as np
import nevergrad as ng
from scipy.signal import fftconvolve
import time

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    Returns 1/C₁ which we want to maximize.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0

    # Compute convolution using FFT for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take the maximum of the convolution (excluding the zero-padding)
    max_conv = np.max(conv[len(sequence)-1:])

    # Calculate C₁ = 2*n*max(b) / (sum(a))^2
    sum_a = np.sum(sequence)
    n = len(sequence)

    if sum_a == 0:
        return 0.0

    C1 = 2 * n * max_conv / (sum_a ** 2)
    return 1 / C1  # Return reciprocal for maximization

def objective_function(sequence):
    """Objective function to maximize 1/C₁"""
    return compute_autocorrelation_constant(sequence)

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using Nevergrad."""
    start_time = time.time()
    best_value = -float('inf')
    best_sequence = []

    # Multi-start approach to avoid local optima
    for attempt in range(10):
        # Randomly sample sequence length between 100 and 1000
        n = np.random.randint(100, 1001)

        # Create optimizer with different strategies
        instrumentation = ng.p.Array(shape=(n,), lower=0, upper=1000)
        optimizer = ng.optimizers.CMA(instrumentation=instrumentation, budget=500)

        # Run optimization for this length
        try:
            recommendation = optimizer.minimize(objective_function, verbosity=0)
            candidate_sequence = recommendation.value
            value = objective_function(candidate_sequence)

            if value > best_value:
                best_value = value
                best_sequence = list(candidate_sequence)

        except Exception as e:
            continue

        # Early exit if time is almost up
        if time.time() - start_time > 170:
            break

    # Ensure at least one element exists
    if len(best_sequence) == 0:
        best_sequence = [1.0]

    # Clip values to [0, 1000] for practicality
    best_sequence = [max(0, min(1000, x)) for x in best_sequence]

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")