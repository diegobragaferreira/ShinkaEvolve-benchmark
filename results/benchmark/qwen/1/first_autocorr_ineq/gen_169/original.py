# EVOLVE-BLOCK-START

import numpy as np
import nevergrad as ng
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import time

def compute_c1(sequence):
    """Computes C1 constant for a given sequence."""
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return float('inf')

    # Use FFT convolution for efficiency
    conv_result = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv_result)
    sum_sq = np.sum(sequence) ** 2

    if sum_sq == 0:
        return float('inf')

    c1 = 2 * len(sequence) * max_conv / sum_sq
    return c1

def objective_function(x):
    """Objective function to maximize 1/C1"""
    # Ensure valid sequence
    if len(x) == 0 or np.sum(x) < 0.01:
        return -float('inf')  # Penalize invalid sequences

    c1 = compute_c1(x)
    if c1 == 0:
        return -float('inf')
    return -1.0 / c1  # Negative because we're minimizing the negative

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    best_inv_c1 = -float('inf')
    best_sequence = None
    start_time = time.time()

    # Try different sequence lengths
    for n in [100, 200, 500, 1000]:
        if time.time() - start_time > 170:  # Leave some time for final processing
            break

        # Create initial population with different strategies
        initial_sequences = []

        # Strategy 1: Random sequences
        for _ in range(5):
            seq = np.random.rand(n)
            seq = seq / np.sum(seq) * 10  # Scale appropriately
            initial_sequences.append(seq)

        # Strategy 2: Step functions with varying heights
        for _ in range(5):
            seq = np.zeros(n)
            # Create some steps
            num_steps = np.random.randint(2, min(20, n//10))
            step_positions = np.sort(np.random.choice(n, num_steps, replace=False))
            step_heights = np.random.rand(num_steps)
            step_heights = step_heights / np.sum(step_heights) * 10

            for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
                if i < len(step_positions)-1:
                    end_pos = step_positions[i+1]
                else:
                    end_pos = n
                seq[pos:end_pos] = height
            initial_sequences.append(seq)

        # Strategy 3: Optimized sequences
        for _ in range(5):
            # Start with a simple pattern and optimize
            seq = np.ones(n) * 0.1
            # Randomly adjust some elements
            indices = np.random.choice(n, size=min(20, n//2), replace=False)
            seq[indices] = np.random.rand(len(indices)) * 5
            initial_sequences.append(seq)

        # Try optimization with different starting points
        for i, seq in enumerate(initial_sequences):
            try:
                # Use Nevergrad for optimization
                optimizer = ng.optimizers.NGOpt(
                    num_workers=1,
                    budget=200
                )
                optimizer.parametrization = ng.p.Array(shape=(n,))

                # Set bounds to [0, 1000] for each element
                for j in range(n):
                    optimizer.parametrization[j].set_bounds(0, 1000)

                # Optimize
                recommendation = optimizer.minimize(objective_function, verbosity=0)

                # Check results
                optimized_seq = recommendation.value
                inv_c1 = -objective_function(optimized_seq)

                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = list(optimized_seq)

            except Exception as e:
                continue

    # Fallback to final optimization if needed
    if best_sequence is None:
        # Try one final comprehensive search
        n = np.random.randint(100, 1000)
        bounds = [(0, 1000) for _ in range(n)]

        def obj_func(x):
            return -objective_function(x)

        result = differential_evolution(obj_func, bounds, seed=42, maxiter=100)
        if result.success:
            inv_c1 = -obj_func(result.x)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = list(result.x)

    # Final check
    if best_sequence is not None:
        # Normalize the sequence properly
        sum_seq = np.sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 10 for x in best_sequence]

    # If no sequence found, return a simple valid one
    if best_sequence is None:
        best_sequence = [1.0] * 100

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")