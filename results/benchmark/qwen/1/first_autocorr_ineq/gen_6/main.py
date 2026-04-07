# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
import random

def compute_autocorrelation_constant(sequence):
    """
    Compute the first autocorrelation inequality constant C₁ for a given sequence.

    Args:
        sequence: List of non-negative real numbers representing step heights

    Returns:
        tuple: (C₁, 1/C₁, max_convolution_value, sum_of_sequence)
    """
    if len(sequence) == 0:
        return float('inf'), 0.0, 0.0, 0.0

    # Convert to numpy array
    a = np.array(sequence)
    sum_a = np.sum(a)

    # Avoid division by zero
    if sum_a < 1e-10:
        return float('inf'), 0.0, 0.0, sum_a

    # Compute autoconvolution using FFT for efficiency
    # Convolution of a with itself
    b = fftconvolve(a, a, mode='full')
    # Take only the relevant part (the actual convolution, not the full cross-correlation)
    b = b[len(a)-1:2*len(a)-1]  # This corresponds to a*a convolution

    max_b = np.max(b)

    # Compute C₁ = 2n * max(b) / (sum(a))^2
    n = len(a)
    c1 = 2 * n * max_b / (sum_a ** 2)

    # Return inverse for maximization
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1, max_b, sum_a

def evaluate_sequence(sequence):
    """
    Evaluate a sequence and return its performance metric.

    Args:
        sequence: List of non-negative real numbers

    Returns:
        float: Performance metric (1/C₁) - higher is better
    """
    try:
        c1, inv_c1, _, sum_a = compute_autocorrelation_constant(sequence)
        # Only accept valid sequences with meaningful sum
        if sum_a < 0.01:
            return 0.0
        return inv_c1
    except Exception as e:
        print(f"Error evaluating sequence: {e}")
        return 0.0

def create_random_sequence(length=None):
    """Create a random sequence with specified or random length."""
    if length is None:
        length = random.randint(100, 500)

    # Create sequence with some randomness but ensure at least one element is significant
    sequence = []
    for _ in range(length):
        # Generate values in a reasonable range
        val = random.uniform(0, 1000)
        sequence.append(val)

    # Ensure at least one element is non-zero and reasonably sized
    if sum(sequence) < 0.01:
        sequence[random.randint(0, len(sequence)-1)] += 0.01

    return sequence

def optimize_sequence():
    """Use differential evolution to find the best sequence."""
    def objective_function(x):
        # Convert to list and apply constraints
        sequence = [max(0, min(1000, val)) for val in x]

        # Evaluate performance (we want to maximize 1/C₁)
        score = evaluate_sequence(sequence)
        # Since scipy.optimize minimizes, we negate for maximization
        return -score

    # Set up bounds (0 to 1000 for each element)
    bounds = [(0, 1000) for _ in range(100)]

    # Use differential evolution for global optimization
    result = optimize.differential_evolution(
        objective_function,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True,
        tol=1e-6
    )

    # Convert best solution back to sequence
    best_sequence = [max(0, min(1000, val)) for val in result.x]

    # Verify the result
    final_score = evaluate_sequence(best_sequence)
    return best_sequence, final_score

def search_for_best_sequence():
    """Search for the best coefficient sequence using improved optimization."""
    start_time = time.time()
    timeout = 170  # Leave 10 seconds for cleanup

    best_sequence = None
    best_score = 0.0

    try:
        # Try several random initializations to avoid local optima
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break

            # Create a random sequence
            sequence = create_random_sequence(random.randint(100, 500))

            # Evaluate this sequence
            score = evaluate_sequence(sequence)
            if score > best_score:
                best_score = score
                best_sequence = sequence[:]

            # Also try the optimization approach
            try:
                optimized_seq, optimized_score = optimize_sequence()
                if optimized_score > best_score:
                    best_score = optimized_score
                    best_sequence = optimized_seq[:]
            except Exception as e:
                print(f"Optimization attempt failed: {e}")
                continue

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Final verification
    if best_sequence is not None:
        final_c1, final_inv_c1, max_b, sum_a = compute_autocorrelation_constant(best_sequence)
        print(f"Final results:")
        print(f"  C₁ = {final_c1:.6f}")
        print(f"  1/C₁ = {final_inv_c1:.6f}")
        print(f"  Max convolution value = {max_b:.6f}")
        print(f"  Sum of sequence = {sum_a:.6f}")
        print(f"  Benchmark ratio = {final_c1 / 1.5031:.6f}")
        return best_sequence
    else:
        # Fallback to a basic sequence if nothing worked
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")