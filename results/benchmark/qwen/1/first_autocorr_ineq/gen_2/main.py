# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import random

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.

    Args:
        sequence: List of non-negative real numbers representing step heights

    Returns:
        C₁ value (float)
    """
    if len(sequence) == 0 or sum(sequence) < 0.01:
        return float('inf')

    # Convert to numpy array for efficient computation
    a = np.array(sequence)
    n = len(a)

    # Compute autoconvolution using FFT for efficiency
    # We use 'full' convolution and then extract the relevant part
    b = signal.convolve(a, a, mode='full')

    # The autoconvolution results in a sequence of length 2*n-1
    # The maximum correlation occurs at index n-1 (center)
    max_correlation = np.max(b)

    # Compute C₁ = 2n * max(b) / (sum(a))^2
    sum_a_squared = sum(a)**2
    if sum_a_squared == 0:
        return float('inf')

    c1 = (2 * n * max_correlation) / sum_a_squared

    return c1

def evaluate_sequence(sequence):
    """
    Evaluates a sequence by computing the inverse of C₁.
    This is what we want to maximize.

    Args:
        sequence: List of non-negative real numbers

    Returns:
        1/C₁ (higher is better)
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence
    elif c1 == 0:
        return float('inf')  # Perfect case
    else:
        return 1.0 / c1

def initialize_random_sequence(min_length=10, max_length=2000):
    """
    Initialize a random sequence with realistic parameters.

    Args:
        min_length: Minimum number of steps
        max_length: Maximum number of steps

    Returns:
        List of random non-negative numbers
    """
    n = random.randint(min_length, max_length)
    # Generate heights that are likely to be effective
    # Use a skewed distribution to favor larger values
    sequence = [random.uniform(0, 1000) for _ in range(n)]
    # Ensure at least one element is significant
    sequence[random.randint(0, n-1)] = random.uniform(1, 100)
    return sequence

def optimize_step_function():
    """
    Main optimization function using a combination of approaches.

    Returns:
        Best sequence found
    """
    best_inv_c1 = 0.0
    best_sequence = None
    start_time = time.time()
    max_time = 170  # Leave some buffer for cleanup

    # Multi-start approach with diverse initializations
    for attempt in range(50):  # Multiple attempts to avoid local optima
        if time.time() - start_time > max_time:
            break

        # Initialize with different strategies
        if attempt < 10:
            # Random initialization
            sequence = initialize_random_sequence()
        elif attempt < 20:
            # Low diversity initialization
            n = random.randint(50, 500)
            sequence = [random.uniform(1, 100) for _ in range(n)]
        elif attempt < 30:
            # High diversity initialization
            n = random.randint(200, 1000)
            sequence = [random.uniform(0, 1000) for _ in range(n)]
        else:
            # Very high diversity
            n = random.randint(1000, 2000)
            sequence = [random.uniform(0, 1000) for _ in range(n)]

        # Ensure at least one element is significant
        if sum(sequence) < 0.01:
            sequence[random.randint(0, len(sequence)-1)] = 1.0

        # Evaluate the initial sequence
        current_inv_c1 = evaluate_sequence(sequence)

        if current_inv_c1 > best_inv_c1:
            best_inv_c1 = current_inv_c1
            best_sequence = sequence[:]

        # Simple local search approach
        for _ in range(20):  # Small number of local iterations
            if time.time() - start_time > max_time:
                break

            # Perturb the sequence slightly
            new_sequence = sequence[:]
            idx = random.randint(0, len(new_sequence)-1)
            new_sequence[idx] = max(0, new_sequence[idx] + random.gauss(0, 10))

            # Ensure non-negativity and minimum sum constraint
            if sum(new_sequence) < 0.01:
                new_sequence[random.randint(0, len(new_sequence)-1)] += 1.0

            new_inv_c1 = evaluate_sequence(new_sequence)

            if new_inv_c1 > current_inv_c1:
                sequence = new_sequence[:]
                current_inv_c1 = new_inv_c1

                if current_inv_c1 > best_inv_c1:
                    best_inv_c1 = current_inv_c1
                    best_sequence = sequence[:]

    # Final check of best sequence
    if best_sequence is not None:
        final_c1 = compute_autocorrelation_constant(best_sequence)
        final_inv_c1 = evaluate_sequence(best_sequence)
        return best_sequence, final_inv_c1, final_c1
    else:
        # Return default sequence if nothing was found
        return [1.0], 1.0, 2.0

def search_for_best_sequence():
    """
    Function to search for the best coefficient sequence.

    Returns:
        Tuple of (best_sequence, inv_c1_score, c1_value)
    """
    try:
        sequence, inv_c1, c1 = optimize_step_function()
        return sequence, inv_c1, c1
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to a simple approach
        fallback_sequence = [1.0] * 100
        fallback_inv_c1 = evaluate_sequence(fallback_sequence)
        fallback_c1 = compute_autocorrelation_constant(fallback_sequence)
        return fallback_sequence, fallback_inv_c1, fallback_c1

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")