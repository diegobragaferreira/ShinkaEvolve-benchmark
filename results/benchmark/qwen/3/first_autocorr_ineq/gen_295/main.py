# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random
import time
from collections import deque
from typing import List, Tuple, Optional

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000
CACHE_SIZE = 1000
INITIAL_POPULATION_COUNT = 10
ADAPTIVE_T_START = 0.05
ADAPTIVE_T_DECAY = 0.98
STAGNATION_THRESHOLD = 20
RESTART_THRESHOLD = 3
HISTORICAL_SAMPLES_COUNT = 5
FIDELITY_SWITCH_THRESHOLD = 0.01

# Cache for storing previously computed C1 values
c1_cache = {}

def _get_c1_cache_key(sequence: List[float]) -> str:
    """Generate a hashable key for caching."""
    return str(tuple(round(x, 8) for x in sequence))

def _cache_c1_value(key: str, value: float):
    """Cache a computed C1 value."""
    if len(c1_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(c1_cache))
        del c1_cache[oldest_key]
    c1_cache[key] = value

def compute_c1_cached(sequence: List[float]) -> float:
    """Cached computation of C1 constant for a sequence."""
    key = _get_c1_cache_key(sequence)
    if key in c1_cache:
        return c1_cache[key]

    n = len(sequence)
    if n == 0:
        result = float('inf')
    else:
        # Compute convolution using FFT
        conv = fftconvolve(np.array(sequence), np.array(sequence), mode='full')
        conv = conv[:2*n-1]
        max_conv = np.max(conv)
        sum_sq = np.sum(sequence)**2

        if sum_sq < 1e-10:
            result = float('inf')
        else:
            c1 = 2 * n * max_conv / sum_sq
            result = c1

    _cache_c1_value(key, result)
    return result

def compute_inv_c1_cached(sequence: List[float]) -> float:
    """Cached computation of 1/C1 constant for a sequence."""
    c1 = compute_c1_cached(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def get_good_direction_to_move_into(sequence: list[float], iteration: int = 0) -> list[float] | None:
    """Returns the direction to move into the sequence with enhanced adaptive parameters."""
    start_time = time.time()

    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Prevent division by zero
    if sum_sequence < 1e-10:
        return None

    # Normalize with adaptive factor
    adaptive_factor = np.sqrt(2 * n)
    normalized_sequence = [x * adaptive_factor / sum_sequence for x in sequence]

    # Use FFT for large sequences, direct convolution for small ones
    if n > 100:
        conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
        conv_result = conv_result[:2*n-1]
    else:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)

    rhs = np.max(conv_result)

    # Try solving LP with improved constraints
    g_fun = solve_convolution_lp(normalized_sequence, rhs, n)

    if g_fun is None:
        # Enhanced fallback: try with modified RHS and different method
        rhs_fallback = rhs * 1.1
        g_fun = solve_convolution_lp(normalized_sequence, rhs_fallback, n)

    if g_fun is None:
        # Final fallback: simple gradient ascent with dynamic step
        t = min(0.05, 0.01 + 0.01 * np.log(n + 1))
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize g_fun
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * adaptive_factor / sum_g_fun for x in g_fun]

    # Adaptive step-size with convergence awareness and time limit checking
    t = min(0.05, 0.01 + 0.01 * np.log(n + 1))

    # Further clamp step size to prevent excessive changes
    t = min(t, 0.02)

    # Apply momentum-based smoothing for better convergence
    if iteration > 0:
        # This is a placeholder for momentum term that would be computed externally
        # In a full implementation, we'd store previous directions and apply momentum
        pass

    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    # Time check to prevent too long evaluations
    if time.time() - start_time > 0.1:
        return None

    return new_sequence

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS with enhanced constraints."""
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Adaptive convolution method selection
    # 1. For small sequences (< 50), always use direct convolution for stability
    # 2. For larger sequences, assess numerical stability before choosing FFT
    if n < 50:
        f_conv = np.convolve(f_sequence, f_sequence)
    else:
        # Compute statistics for adaptive choice
        max_val = np.max(f_sequence)
        std_val = np.std(f_sequence)
        sum_val = np.sum(f_sequence)

        # Thresholds for numerical stability assessment
        stable_condition = (
            max_val < 1e4 and
            std_val < 1e3 and
            sum_val > 1e-10 and
            n < 5000  # Avoid very large FFTs
        )

        # Choose method based on stability and size
        if stable_condition:
            try:
                f_conv = fftconvolve(f_sequence, f_sequence, mode='full')
                f_conv = f_conv[:2*n-1]
                # Additional validation for numerical issues
                if np.any(np.isnan(f_conv)) or np.any(np.isinf(f_conv)) or np.max(np.abs(f_conv)) > 1e12:
                    # Fall back to direct method if FFT becomes unstable
                    f_conv = np.convolve(f_sequence, f_sequence)
            except:
                # Fallback to the most robust direct method
                f_conv = np.convolve(f_sequence, f_sequence)
        else:
            # Use direct convolution for potentially unstable FFT scenarios
            f_conv = np.convolve(f_sequence, f_sequence)

    # Improved constraint matrix creation with careful indexing
    for k in range(2 * n - 1):
        # Create constraint row for convolution bound
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

    # Try multiple methods to solve LP with enhanced error handling and timeout control
    try:
        # Try the 'highs' method first for better performance, with timeout protection
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                                 options={'presolve': True, 'maxiter': 1000})
    except Exception as e:
        # Fallback to 'simplex' with relaxed tolerances and additional error handling
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex',
                                     options={'maxiter': 1000, 'tol': 1e-8})
        except:
            return None

    if result.success:
        g_sequence = result.x
        # Validate solution
        if np.any(np.isnan(g_sequence)) or np.any(np.isinf(g_sequence)):
            return None
        # Ensure non-negativity and reasonable values
        g_sequence = np.maximum(g_sequence, 0)
        if np.sum(g_sequence) < 1e-10:
            return None
        return g_sequence
    else:
        return None

def initialize_historical_sequence() -> list[float]:
    """Initialize a sequence by sampling from historical good sequences."""
    # Sample from a set of known good sequences
    historical_samples = [
        # These are sample sequences that have shown good performance for C1
        [1.0]*100,  # Uniform distribution
        [1.0]*50 + [0.1]*50,  # Spike sequence
        [1.0/i for i in range(1, 101)],  # Harmonic decay
        [1.0]*20 + [0.0]*80,  # Sparse sequence
    ]

    # Add some randomness to diversity
    selected_sample = random.choice(historical_samples)

    # Apply small random perturbations
    perturbed = [max(0.01, x * random.uniform(0.8, 1.2)) for x in selected_sample]

    return perturbed

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with enhanced diversity."""
    # Initialize with deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Multi-start approach with historical sequences
    best_sequence = None
    best_inv_c1 = 0.0
    start_time = time.time()

    # Start with some good historical sequences
    for _ in range(HISTORICAL_SAMPLES_COUNT):
        candidate = initialize_historical_sequence()
        candidate_inv_c1 = compute_inv_c1_cached(candidate)
        if candidate_inv_c1 > best_inv_c1:
            best_inv_c1 = candidate_inv_c1
            best_sequence = candidate

    # Continue optimization if we have time
    iteration = 0
    while time.time() - start_time < MAX_TIME_SECONDS - 5 and iteration < MAX_ITERATIONS:
        if best_sequence is None:
            best_sequence = initialize_historical_sequence()

        # Try gradient-based improvement
        improved_sequence = get_good_direction_to_move_into(best_sequence, iteration)

        if improved_sequence is not None:
            improved_inv_c1 = compute_inv_c1_cached(improved_sequence)
            if improved_inv_c1 > best_inv_c1:
                best_sequence = improved_sequence
                best_inv_c1 = improved_inv_c1

        # Add some exploration with random mutations periodically
        if iteration % 10 == 0:
            # Inject some random diversity
            mutated_sequence = [x * random.uniform(0.9, 1.1) for x in best_sequence]
            mutated_sequence = [max(0.01, x) for x in mutated_sequence]
            mutated_inv_c1 = compute_inv_c1_cached(mutated_sequence)
            if mutated_inv_c1 > best_inv_c1:
                best_sequence = mutated_sequence
                best_inv_c1 = mutated_inv_c1

        iteration += 1

    # Final refinement
    if best_sequence is not None:
        # Try a few more rounds of fine tuning
        for i in range(5):
            refined = get_good_direction_to_move_into(best_sequence, iteration+i)
            if refined is not None:
                refined_inv_c1 = compute_inv_c1_cached(refined)
                if refined_inv_c1 > best_inv_c1:
                    best_sequence = refined
                    best_inv_c1 = refined_inv_c1
                else:
                    break
            else:
                break

    # Ensure we have a valid result
    if best_sequence is None:
        best_sequence = [1.0] * 100  # Fallback to simple case

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")