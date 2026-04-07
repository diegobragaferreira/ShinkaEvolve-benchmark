# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from collections import deque, defaultdict
import warnings

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Memoization cache for convolution results
_conv_cache = {}

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better efficiency with caching"""
    n = len(seq)
    if n == 0:
        return np.array([])

    # Create a unique hashable key for caching
    seq_key = tuple(seq)
    if seq_key in _conv_cache:
        return _conv_cache[seq_key]

    # Pad to size 2*n-1 for full convolution
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    # Compute FFT-based convolution
    fft_seq = fft(padded_seq)
    conv_result = ifft(fft_seq * np.conj(fft_seq)).real[:2*n-1]

    # Cache the result
    _conv_cache[seq_key] = conv_result
    return conv_result

def compute_autocorrelation_constant(sequence):
    """Compute the C1 constant for a given sequence"""
    if len(sequence) == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    # Use FFT for efficient convolution
    conv_result = compute_convolution_fft(sequence)
    max_conv = np.max(conv_result)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_a ** 2)

    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing its inverse C1"""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence
    return 1.0 / c1  # Return 1/C1 as the objective

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build constraint matrix efficiently
    # Each constraint corresponds to a convolution element
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Add non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    # Add bounds to avoid numerical issues
    bounds = [(0, 1000) for _ in range(n)]  # Clips heights to [0, 1000]

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        return None

def get_gradient_estimate(sequence, epsilon=1e-4):
    """Estimate gradient using finite differences"""
    n = len(sequence)
    if n == 0:
        return None

    grad = []
    for i in range(n):
        # Create perturbed sequences
        seq_plus = sequence.copy()
        seq_minus = sequence.copy()

        seq_plus[i] += epsilon
        seq_minus[i] -= epsilon

        # Evaluate both and estimate derivative
        val_plus = evaluate_sequence(seq_plus)
        val_minus = evaluate_sequence(seq_minus)

        grad_i = (val_plus - val_minus) / (2 * epsilon)
        grad.append(grad_i)

    return np.array(grad)

def adaptive_sequence_length():
    """Adaptively choose sequence length based on performance considerations"""
    # Sample from a log-uniform distribution to explore various sizes
    base_length = 500
    return int(np.random.lognormal(np.log(base_length), 0.5))

def generate_step_function(n, step_count=None):
    """Generate a step function with specified number of steps."""
    if step_count is None:
        step_count = max(1, n // 3)  # Default to about 1/3 of n steps

    # Create a step function
    step_width = n // step_count
    if step_width == 0:
        step_width = 1

    step_function = []
    for i in range(step_count):
        step_height = np.random.uniform(0.1, 10.0)
        start_idx = i * step_width
        end_idx = min((i+1) * step_width, n)
        step_function.extend([step_height] * (end_idx - start_idx))

    # Fill any remaining elements
    while len(step_function) < n:
        step_function.append(np.random.uniform(0.1, 10.0))

    return step_function[:n]

def get_better_direction(sequence):
    """Generate a better sequence direction using a hybrid approach."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Normalize the sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution and RHS for LP
    conv_result = compute_convolution_fft(normalized_sequence)
    rhs = np.max(conv_result)

    # Solve LP to get improved direction
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    # Normalize the resulting sequence
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Use weighted average with gradient-based step
    grad = get_gradient_estimate(sequence)
    if grad is not None:
        # Combine gradient information with LP solution
        t = 0.1  # Blend factor
        new_sequence = [
            (1-t)*x + t*y + 0.01*grad[i] for i, (x, y) in enumerate(zip(sequence, normalized_g_fun))
        ]
    else:
        new_sequence = [
            (1-0.05)*x + 0.05*y for x, y in zip(sequence, normalized_g_fun)
        ]

    # Ensure non-negativity and clipping
    new_sequence = [max(0, min(x, 1000)) for x in new_sequence]

    return new_sequence

def local_search_improvement(initial_seq, max_iter=50, use_cached=False):
    """Perform local search improvements around a sequence."""
    current_seq = initial_seq.copy()
    current_score = evaluate_sequence(current_seq)

    if use_cached:
        # Use cached evaluations to avoid recomputation
        cached_evaluations = {}
        def cached_evaluate(seq):
            seq_tuple = tuple(seq)
            if seq_tuple in cached_evaluations:
                return cached_evaluations[seq_tuple]
            else:
                val = evaluate_sequence(seq)
                cached_evaluations[seq_tuple] = val
                return val
    else:
        cached_evaluate = evaluate_sequence

    for _ in range(max_iter):
        # Get a better direction
        better_dir = get_better_direction(current_seq)
        if better_dir is None:
            break

        # Try multiple steps
        for step_size in [0.05, 0.1, 0.2]:
            candidate_seq = [
                max(0, min(x + step_size * (y-x), 1000))
                for x, y in zip(current_seq, better_dir)
            ]

            candidate_score = cached_evaluate(candidate_seq)
            if candidate_score > current_score:
                current_seq = candidate_seq
                current_score = candidate_score
                break  # Accept the improvement

    return current_seq

def special_step_function_search(max_time=180):
    """Specialized search for step functions with enhanced configurations."""
    start_time = time.time()

    # History to store recent best scores for early stopping
    recent_scores = deque(maxlen=10)

    best_sequence = None
    best_score = 0.0
    best_c1 = float('inf')

    iteration = 0
    max_iterations = 10000

    # Phase 1: Specialized step function generation with adaptive parameters
    while iteration < max_iterations and time.time() - start_time < max_time:
        # Generate a step function with a specific number of steps
        n = adaptive_sequence_length()
        step_count = max(1, int(np.random.lognormal(np.log(n/4), 0.3)))

        sequence = generate_step_function(n, step_count)

        # Improved local search on the step function
        improved_seq = local_search_improvement(sequence, max_iter=50)

        score = evaluate_sequence(improved_seq)
        c1 = compute_autocorrelation_constant(improved_seq)

        if score > best_score:
            best_score = score
            best_sequence = improved_seq.copy()
            best_c1 = c1
            print(f"Iteration {iteration}: New best score = {score:.6f}, C1 = {c1:.6f}")

            # Check benchmark
            benchmark_ratio = 1.5031 / c1
            if benchmark_ratio > 1.0:
                print(f"BEAT BENCHMARK at iteration {iteration}! Ratio = {benchmark_ratio:.6f}")
                break

        # Store recent scores for convergence detection
        recent_scores.append(score)

        # Early stopping if scores are not improving
        if len(recent_scores) == recent_scores.maxlen:
            if abs(max(recent_scores) - min(recent_scores)) < 1e-6:
                print(f"Early stopping at iteration {iteration}")
                break

        iteration += 1

    # Clear cache to avoid memory buildup
    _conv_cache.clear()

    return best_sequence

def search_for_best_sequence(max_time=180) -> list[float]:
    """Main function to search for the best coefficient sequence."""
    return special_step_function_search(max_time)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")