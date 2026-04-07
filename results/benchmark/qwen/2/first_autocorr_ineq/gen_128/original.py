# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import random
import time

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

def quadratic_programming_optimization(max_time_seconds=180):
    """Optimizes the sequence using Quadratic Programming approach."""
    start_time = time.time()

    best_sequence = None
    best_inv_c1 = 0.0

    # Try several random initializations
    for attempt in range(20):
        if time.time() - start_time > max_time_seconds:
            break

        # Generate a random sequence
        initial_seq = generate_random_sequence()

        # Use scipy's minimize with L-BFGS-B for continuous optimization
        def objective(x):
            # Convert to proper sequence format
            seq = np.abs(x)  # Ensure non-negative
            if np.sum(seq) < 0.01:
                return float('inf')

            c1 = compute_c1_constant(seq)
            if c1 == float('inf'):
                return float('inf')
            return -1.0 / c1  # Minimize negative to maximize 1/C1

        # Set bounds for each variable
        bounds = [(0, 1000) for _ in range(len(initial_seq))]

        # Add some noise to make it more interesting
        x0 = np.array(initial_seq) + np.random.normal(0, 0.1, len(initial_seq))
        x0 = np.maximum(x0, 0)  # Ensure non-negative

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            if res.success:
                optimized_seq = np.abs(res.x)
                inv_c1_val = evaluate_sequence(optimized_seq)

                if inv_c1_val > best_inv_c1:
                    best_inv_c1 = inv_c1_val
                    best_sequence = optimized_seq.tolist()
        except:
            continue

    # If no good solution was found, return a random sequence
    if best_sequence is None:
        best_sequence = generate_random_sequence()

    return best_sequence, best_inv_c1

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    sequence, fitness = quadratic_programming_optimization(180)
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")