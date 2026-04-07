# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def compute_autocorrelation_constant(sequence):
    """Compute C₁ for a given sequence with numerical stability."""
    if len(sequence) == 0:
        return float('inf')

    a = np.array(sequence, dtype=np.float64)
    n = len(a)

    # Use FFT-based convolution for efficiency when feasible
    # But fallback to direct convolution for better numerical stability
    try:
        # For larger sequences, use FFT
        if n > 100:
            b = signal.convolve(a, a, mode='full')
        else:
            # For smaller sequences, use direct convolution for higher precision
            b = signal.convolve(a, a, mode='full')
    except Exception:
        # Fallback to explicit convolution if signal.convolve fails
        b = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                b[i + j] += a[i] * a[j]

    max_conv = np.max(b)

    # Compute C₁ = 2n * max(b) / (sum(a))^2
    sum_a = np.sum(a)

    # Avoid division by zero or very small values
    if sum_a < 1e-10:
        return float('inf')

    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence for the optimization problem with enhanced checks."""
    # Ensure we have a valid sequence with sufficient sum
    if len(sequence) == 0:
        return float('-inf')

    # Clip values to [0, 1000] as per constraints
    sequence = np.clip(sequence, 0, 1000)

    # Check sum constraint
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('-inf')

    # Compute C₁
    c1 = compute_autocorrelation_constant(sequence)

    # Return 1/C₁ (we want to maximize this)
    # If C₁ is very large, 1/C₁ approaches 0
    if c1 > 1e10:
        return float('-inf')

    return 1.0 / c1

def generate_random_sequence(length_range=(100, 1000)):
    """Generate a random sequence within specified length range."""
    n = random.randint(*length_range)
    # Generate random sequence with values in [0, 1000]
    sequence = np.random.uniform(0, 1000, n)
    return sequence.tolist()

def generate_geometric_sequence(length_range=(100, 1000)):
    """Generate a geometrically decreasing sequence."""
    n = random.randint(*length_range)
    # Generate geometric sequence
    sequence = [0.9 ** i for i in range(n)]
    # Scale to keep sum reasonable
    sequence = [x * 1000 for x in sequence]
    return sequence

def generate_spike_sequence(length_range=(100, 1000)):
    """Generate a sequence with a single spike."""
    n = random.randint(*length_range)
    sequence = [0.0] * n
    spike_idx = random.randint(0, n - 1)
    sequence[spike_idx] = 1000.0
    return sequence

def optimize_sequence_single(start_id, init_strategy='random'):
    """Optimize a single sequence using a global optimization approach."""
    # Set random seed for reproducibility
    np.random.seed(42 + start_id)
    random.seed(42 + start_id)

    sequence = None
    if init_strategy == 'random':
        sequence = generate_random_sequence()
    elif init_strategy == 'geometric':
        sequence = generate_geometric_sequence()
    else:  # spike
        sequence = generate_spike_sequence()

    # Evaluate initial sequence
    initial_score = evaluate_sequence(sequence)

    # Optimization parameters
    bounds = [(0, 1000) for _ in range(len(sequence))]

    def objective(x):
        # Convert to list and apply bounds
        seq_list = list(x)
        # Ensure all elements are within [0, 1000]
        seq_list = [max(0, min(1000, val)) for val in seq_list]

        # Evaluate and return negative because we want to maximize
        return -evaluate_sequence(seq_list)

    try:
        # Use differential evolution with increased population and iterations
        result = differential_evolution(
            objective,
            bounds,
            maxiter=75,  # Increased iterations for better convergence
            popsize=25,  # Increased population size for better exploration
            seed=42+start_id,
            disp=False,
            tol=1e-6
        )

        if result.success:
            optimized_seq = list(result.x)
            # Apply final bounds
            optimized_seq = [max(0, min(1000, val)) for val in optimized_seq]

            score = evaluate_sequence(optimized_seq)
            return optimized_seq, score
    except Exception as e:
        print(f"Optimization failed for start {start_id}: {e}")
        pass

    # Fallback to initial sequence
    return sequence, initial_score

def optimize_sequence_parallel(num_workers=4):
    """Run multiple optimizations in parallel."""
    best_score = float('-inf')
    best_sequence = None

    # Define initialization strategies
    strategies = ['random', 'geometric', 'spike']

    # Use thread pool for parallel optimization
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each strategy and iteration
        futures = []
        for i in range(num_workers * 3):  # Run 3 times each strategy
            strategy = strategies[i % len(strategies)]
            futures.append(executor.submit(optimize_sequence_single, i, strategy))

        # Collect results
        for future in futures:
            seq, score = future.result()
            if score > best_score:
                best_score = score
                best_sequence = seq

    return best_sequence, best_score

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()

    # Run parallel optimization
    best_sequence, best_score = optimize_sequence_parallel(num_workers=4)

    end_time = time.time()
    eval_time = end_time - start_time

    # Calculate benchmark ratio
    benchmark_ratio = best_score / 0.6653  # 1.5031 is the threshold for C₁

    print(f"Best 1/C₁: {best_score:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Execution time: {eval_time:.4f} seconds")

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")