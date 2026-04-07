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
INITIAL_POPULATION_COUNT = 10
ADAPTIVE_T_START = 0.05
ADAPTIVE_T_DECAY = 0.98

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

def get_good_direction_to_move_into(sequence: list[float], generation: int = 0) -> list[float] | None:
    """Returns the direction to move into the sequence with adaptive step size."""
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

        # Adaptive step size that decreases over time
        t = ADAPTIVE_T_START * (ADAPTIVE_T_DECAY ** generation)
        new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

        return new_sequence
    except Exception as e:
        return None

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS with constraint relaxation."""
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
            # Try relaxed constraints if the original fails
            try:
                b_ub_relaxed = np.array(b_ub) * 0.95
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub_relaxed, method='highs')
                if result.success:
                    g_sequence = result.x
                    g_sequence = np.maximum(g_sequence, 0)
                    return g_sequence.tolist()
            except Exception:
                pass
            return None
    except Exception:
        return None

def initialize_sequence():
    """Initialize a promising sequence for optimization."""
    # Try different initialization strategies to find a good starting point
    strategy = random.choice(['harmonic', 'random', 'spike'])

    if strategy == 'harmonic':
        # Start with a structured sequence
        n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        # Create a sequence with decreasing values to encourage sparsity
        sequence = [1.0 / (i + 1) for i in range(n)]
        # Normalize to have reasonable magnitude
        total = sum(sequence)
        sequence = [x * 2.0 / total for x in sequence]
    elif strategy == 'spike':
        n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        # Create a sparse sequence with one large spike
        sequence = [0.0] * n
        spike_idx = random.randint(0, n-1)
        sequence[spike_idx] = random.uniform(1.0, 5.0)
    else:  # random
        n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        sequence = [random.uniform(0.1, 2.0) for _ in range(n)]

    return sequence

def search_for_best_sequence() -> list[float]:
    """Search for the best coefficient sequence with enhanced initialization."""
    start_time = time.time()

    # Multi-start initialization
    best_sequence = None
    best_c1 = float('inf')

    for _ in range(INITIAL_POPULATION_COUNT):
        candidate_sequence = initialize_sequence()
        candidate_c1 = compute_c1_constant(candidate_sequence)
        if candidate_c1 < best_c1:
            best_c1 = candidate_c1
            best_sequence = candidate_sequence

    if best_sequence is None:
        best_sequence = initialize_sequence()

    prev_c1 = best_c1
    stagnation_counter = 0
    last_improvement_gen = 0

    # Evolutionary loop
    for gen in range(MAX_GENERATIONS):
        if time.time() - start_time > MAX_TIME_SECONDS - 5:
            break

        # Try gradient-based improvement
        improved_sequence = get_good_direction_to_move_into(best_sequence, gen)

        if improved_sequence is not None:
            new_c1 = compute_c1_constant(improved_sequence)
            if new_c1 < prev_c1:
                best_sequence = improved_sequence
                prev_c1 = new_c1
                stagnation_counter = 0
                last_improvement_gen = gen
                continue

        # If no improvement, check for stagnation
        if gen - last_improvement_gen > 20:
            stagnation_counter += 1
            if stagnation_counter > 3:
                # Restart with new initialization
                best_sequence = initialize_sequence()
                prev_c1 = compute_c1_constant(best_sequence)
                last_improvement_gen = gen
                stagnation_counter = 0
        else:
            # Apply hybrid mutation
            n = max(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, int(len(best_sequence) * 0.95)))
            if random.random() < 0.5:
                # Structured perturbation for exploration
                mutated_sequence = best_sequence.copy()
                # Modify some elements with structured changes
                for i in range(len(mutated_sequence)):
                    if random.random() < MUTATION_RATE:
                        # Apply either random change or systematic perturbation
                        if random.random() < 0.5:
                            mutated_sequence[i] *= random.uniform(0.8, 1.2)
                        else:
                            mutated_sequence[i] = random.uniform(0.1, 2.0)
                best_sequence = mutated_sequence
            else:
                # Random perturbation
                best_sequence = [random.uniform(0.1, 2.0) for _ in range(n)]

    # Final check and refinement
    final_c1 = compute_c1_constant(best_sequence)
    if final_c1 >= 1.5031:
        # Attempt one final optimization
        refined = get_good_direction_to_move_into(best_sequence, MAX_GENERATIONS)
        if refined is not None:
            test_c1 = compute_c1_constant(refined)
            if test_c1 < final_c1:
                best_sequence = refined

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")