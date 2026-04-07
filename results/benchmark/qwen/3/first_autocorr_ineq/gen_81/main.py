# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
import time
import random

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_GENERATIONS = 500
POPULATION_SIZE = 100
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7

def convolve_fft(a, b):
    """Compute convolution using FFT for efficiency."""
    n = len(a)
    pad_size = 2 * n - 1
    a_padded = np.pad(a, (0, pad_size - n), 'constant')
    b_padded = np.pad(b, (0, pad_size - n), 'constant')
    a_fft = np.fft.fft(a_padded)
    b_fft = np.fft.fft(b_padded)
    conv_result = np.fft.ifft(a_fft * np.conj(b_fft))
    return np.real(conv_result[:pad_size])

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)
    sum_sq = np.sum(sequence)**2

    if sum_sq < 1e-10:
        return float('inf')

    c1 = 2 * n * max_conv / sum_sq
    return c1

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    try:
        n = len(sequence)
        if n < MIN_SEQ_LENGTH:
            return None

        # Normalize sequence for processing
        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            return None

        # Normalize to avoid numerical issues
        normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

        # Compute the target RHS for LP solver
        conv = convolve_fft(normalized_sequence, normalized_sequence)
        rhs = np.max(conv)

        # Solve the LP optimization
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)
        if g_fun is None:
            return None

        # Normalize the result and create new sequence
        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            return None

        normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g
        t = 0.01
        new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

        return new_sequence
    except Exception as e:
        return None

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        # Create the constraint matrix using the convolution structure
        a_ub = []
        b_ub = []

        # Generate convolution constraints
        f_seq = np.array(f_sequence)
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_seq[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Add non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)
        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Define objective function (we want to minimize negative sum)
        c = -np.ones(n)

        # Solve the linear program
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            # Ensure non-negativity due to numerical errors
            g_sequence = np.maximum(g_sequence, 0)
            return g_sequence.tolist()
        else:
            # Try again with relaxed constraints if needed
            try:
                # Relax the RHS by 5%
                rhs_relaxed = rhs * 0.95
                a_ub_relaxed = a_ub.copy()
                b_ub_relaxed = [r * 0.95 for r in b_ub]

                result_relaxed = optimize.linprog(c, A_ub=a_ub_relaxed, b_ub=b_ub_relaxed, method='highs')
                if result_relaxed.success:
                    g_sequence = result_relaxed.x
                    g_sequence = np.maximum(g_sequence, 0)
                    return g_sequence.tolist()
            except:
                pass
            return None
    except Exception:
        return None

def initialize_sequence():
    """Initialize a promising sequence for optimization."""
    # Start with a structured sequence
    n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    # Create a sequence with decreasing values to encourage sparsity
    sequence = [1.0 / (i + 1) for i in range(n)]
    # Normalize to have reasonable magnitude
    total = sum(sequence)
    sequence = [x * 2.0 / total for x in sequence]
    return sequence

def adaptive_local_search(sequence, max_iter=LOCAL_SEARCH_LIMIT):
    """Perform adaptive local search around the current sequence."""
    best_sequence = sequence.copy()
    best_c1 = compute_c1_constant(best_sequence)

    for i in range(max_iter):
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        if improved_sequence is not None:
            new_c1 = compute_c1_constant(improved_sequence)
            if new_c1 < best_c1:
                best_sequence = improved_sequence
                best_c1 = new_c1
            else:
                # Stop if no improvement
                break
        else:
            # Break if optimization fails
            break

    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Search for the best coefficient sequence."""
    start_time = time.time()

    # Initialize with a promising sequence
    best_sequence = initialize_sequence()
    prev_c1 = compute_c1_constant(best_sequence)

    # Adaptive evolutionary loop
    for gen in range(MAX_GENERATIONS):
        if time.time() - start_time > MAX_TIME_SECONDS - 5:
            break

        # Try gradient-based improvement
        improved_sequence = get_good_direction_to_move_into(best_sequence)

        if improved_sequence is not None:
            new_c1 = compute_c1_constant(improved_sequence)
            if new_c1 < prev_c1:
                best_sequence = improved_sequence
                prev_c1 = new_c1
                # Perform adaptive local search after significant improvement
                if (prev_c1 - new_c1) / prev_c1 > IMPROVEMENT_THRESHOLD:
                    best_sequence = adaptive_local_search(best_sequence)
                continue

        # If gradient fails or doesn't improve, try evolutionary approach
        # Hybrid: try both random and structured mutations
        n = max(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, int(len(best_sequence) * 0.95)))

        # Try differential evolution for better global search
        try:
            def objective(x):
                # Minimize negative inverse C1 (i.e., maximize inverse C1)
                return -1.0 / compute_c1_constant(x.tolist())

            bounds = [(0.01, 100.0)] * n
            result = differential_evolution(objective, bounds, maxiter=20, popsize=10, seed=42)
            if not np.isnan(result.fun):
                de_sequence = result.x.tolist()
                de_c1 = compute_c1_constant(de_sequence)
                if de_c1 < prev_c1:
                    best_sequence = de_sequence
                    prev_c1 = de_c1
                    continue
        except:
            pass

        # Fallback to random perturbation
        if random.random() < 0.5:
            # Random perturbation
            best_sequence = [random.uniform(0.1, 2.0) for _ in range(n)]
        else:
            # Structured mutation
            mutated_sequence = best_sequence.copy()
            for i in range(len(mutated_sequence)):
                if random.random() < MUTATION_RATE:
                    mutated_sequence[i] *= random.uniform(0.8, 1.2)
            best_sequence = mutated_sequence

    # Final adaptive local search
    best_sequence = adaptive_local_search(best_sequence, max_iter=5)

    # Final check
    final_c1 = compute_c1_constant(best_sequence)
    if final_c1 >= 1.5031:
        # Attempt one final optimization
        refined = get_good_direction_to_move_into(best_sequence)
        if refined is not None:
            test_c1 = compute_c1_constant(refined)
            if test_c1 < final_c1:
                best_sequence = refined

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")