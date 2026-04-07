# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import convolve
import time
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = convolve(sequence, sequence, mode='full')
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

def initialize_good_sequence(length=None):
    """Initialize a good starting sequence based on theoretical insights."""
    if length is None:
        length = random.randint(100, 500)  # Reasonable range

    # Start with a simple exponential decay pattern which often works well
    decay_factor = 0.95
    sequence = [1.0 * (decay_factor ** i) for i in range(length)]

    # Ensure minimum value and normalize
    sequence = [max(x, 0.01) for x in sequence]

    # Apply a bit of randomness to avoid local optima
    noise_factor = 0.1
    sequence = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in sequence]
    sequence = [max(x, 0.01) for x in sequence]

    return sequence

def optimize_with_nelder_mead(initial_sequence, max_iter=50):
    """Optimize using Nelder-Mead method which is more suitable for this problem."""
    def objective_func(seq_array):
        # Return negative because we want to maximize
        return -compute_inv_c1(seq_array.tolist())

    # Ensure we start with a valid sequence
    if np.sum(initial_sequence) < 0.01:
        initial_sequence = [0.1] + [0.0] * (len(initial_sequence) - 1)

    # Convert to numpy array for optimization
    x0 = np.array(initial_sequence, dtype=float)

    # Set bounds for optimization (non-negative and reasonable values)
    bounds = [(0, 1000) for _ in range(len(x0))]

    # Use Nelder-Mead for direct search without gradients
    result = optimize.minimize(
        objective_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'adaptive': True}
    )

    if result.success:
        optimized_seq = np.maximum(result.x, 0)  # Ensure non-negative
        # Ensure sum is reasonable
        if np.sum(optimized_seq) < 0.01:
            optimized_seq[0] = 0.1
        return optimized_seq.tolist()
    else:
        return initial_sequence

def smart_search():
    """Smart search approach that combines multiple strategies."""
    best_inv_c1 = 0
    best_sequence = None

    # Try multiple random starting points with good initialization
    for attempt in range(5):  # Reduced attempts for speed
        # Initialize with better patterns
        n = random.randint(100, 500)
        sequence = initialize_good_sequence(n)

        # Optimize this sequence
        optimized = optimize_with_nelder_mead(sequence, max_iter=30)

        # Evaluate
        inv_c1 = compute_inv_c1(optimized)

        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized[:]

    # If no improvement found, return a default good sequence
    if best_sequence is None:
        best_sequence = initialize_good_sequence(200)
        best_sequence = optimize_with_nelder_mead(best_sequence, max_iter=50)
        best_inv_c1 = compute_inv_c1(best_sequence)

    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Add a time limit check to ensure we don't exceed budget
    start_time = time.time()

    # Perform the smart search
    best_sequence = smart_search()

    # Additional local refinement if time permits
    if time.time() - start_time < 150:  # Leave some time for final refinement
        # Try a few more local optimizations on the best found
        refined_sequence = optimize_with_nelder_mead(best_sequence, max_iter=20)
        refined_inv_c1 = compute_inv_c1(refined_sequence)
        if refined_inv_c1 > compute_inv_c1(best_sequence):
            best_sequence = refined_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")