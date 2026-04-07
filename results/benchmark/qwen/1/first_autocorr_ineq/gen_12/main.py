# EVOLVE-BLOCK-START

import numpy as np
import nevergrad as ng
from scipy import signal
import time
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.

    Returns:
        float: The value of 1/C₁ (which we want to maximize)
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0

    # Compute convolution using FFT for efficiency
    conv = signal.convolve(sequence, sequence, mode='full')

    # Take the maximum of the convolution (excluding the zero padding)
    max_conv = np.max(conv[len(sequence)-1:])

    # Compute the constant C₁
    sum_seq = np.sum(sequence)
    n = len(sequence)

    # We want to maximize 1/C₁ = (sum(a))^2 / (2*n*max(b))
    if max_conv == 0:
        return float('inf')

    inv_c1 = (sum_seq ** 2) / (2 * n * max_conv)
    return inv_c1

def objective_function(x):
    """
    Objective function to minimize (we'll negate it since nevergrad minimizes).

    Args:
        x: Array of step heights

    Returns:
        Negative of 1/C₁ (since we want to maximize 1/C₁)
    """
    # Clip values to [0, 1000] as per constraints
    x = np.clip(x, 0, 1000)

    # Ensure at least one element is positive
    if np.sum(x) < 0.01:
        return float('inf')

    return -compute_autocorrelation_constant(x)

def search_for_best_sequence():
    """
    Function to search for the best coefficient sequence using nevergrad,
    with FFT-based convolution for efficiency.
    """
    # Initialize best score and sequence
    best_inv_c1 = 0.0
    best_sequence = []

    # Try different sequence lengths
    for n in [100, 200, 500, 1000]:
        # Define the optimization problem with bounds [0, 1000] for each dimension
        instrumentation = ng.p.Array(shape=(n,))
        optimizer = ng.optimizers.CMA(positional_bounds=(0, 1000))

        # Run optimization
        for _ in range(100):  # More iterations for better search
            try:
                # Ask for a candidate point
                candidate = optimizer.ask()

                # Evaluate the objective function
                value = objective_function(candidate.value)

                # Tell the optimizer the result
                optimizer.tell(candidate, value)

            except Exception as e:
                continue

        # Get the best solution found
        try:
            # Ask for the best solution
            best_candidate = optimizer.provide_recommendation()
            current_sequence = best_candidate.value

            # Evaluate the final score
            current_inv_c1 = compute_autocorrelation_constant(current_sequence)

            # Update if better
            if current_inv_c1 > best_inv_c1:
                best_inv_c1 = current_inv_c1
                best_sequence = current_sequence.copy()
        except Exception as e:
            continue

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")