# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Pad to size 2*n-1 for full convolution
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    # FFT convolution
    fft_seq = fft(padded_seq)
    conv_result = ifft(fft_seq * np.conj(fft_seq)).real
    # Return only the relevant part (first 2*n-1 elements)
    return conv_result[:2*n-1]

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence with improved strategies."""
    n = len(sequence)
    if n < 1:
        return None

    # Compute current convolution
    try:
        conv_result = compute_convolution_fft(sequence)
        max_conv = np.max(conv_result)
    except Exception:
        # Fallback to direct convolution if FFT fails
        conv_result = np.convolve(sequence, sequence)
        max_conv = np.max(conv_result)

    # Normalize sequence for better numerics
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None

    normalized_sequence = [x / sum_sequence for x in sequence]

    # Adaptive step size - decrease as we get closer to optimum
    base_t = 0.01
    t = base_t * (1.0 / (1.0 + n / 1000.0))  # Decrease with sequence length

    # Solve LP with better initialization and fallback strategies
    g_fun = solve_convolution_lp_with_fallback(normalized_sequence, max_conv)

    if g_fun is None:
        # Try simple gradient ascent as fallback
        try:
            # Simple gradient ascent - move towards increasing values
            g_fun = [max(0, x + 0.01 * (random.random() - 0.5)) for x in sequence]
            # Normalize again
            sum_g = np.sum(g_fun)
            if sum_g > 0:
                g_fun = [x / sum_g for x in g_fun]
        except:
            return None

    # Apply the update
    if g_fun is not None:
        sum_g = np.sum(g_fun)
        if sum_g > 0:
            normalized_g_fun = [x / sum_g for x in g_fun]
            new_sequence = [
                (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
            ]
            return new_sequence

    return sequence

def solve_convolution_lp_with_fallback(f_sequence, rhs):
    """Solves the convolution LP with fallback strategies."""
    n = len(f_sequence)
    if n < 1:
        return None

    # Try normal LP approach first
    g_fun = solve_convolution_lp(f_sequence, rhs)

    if g_fun is not None:
        return g_fun

    # Fallback 1: Try with slightly relaxed constraints
    try:
        # Relax the constraint slightly
        g_fun = solve_convolution_lp(f_sequence, rhs * 1.01)
        if g_fun is not None:
            return g_fun
    except:
        pass

    # Fallback 2: Return simple pattern
    try:
        # Return a simple uniform pattern
        return np.ones(n) / n
    except:
        pass

    # Fallback 3: Return original sequence (no change)
    return f_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n < 1:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Generate constraint matrix using FFT for efficiency
    try:
        # Use FFT to generate convolution constraints efficiently
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)
    except:
        # Fallback to manual construction
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except:
        # If optimization fails, return None to trigger fallback
        return None


def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with improved strategy."""
    # Start with a diverse range of sequences to avoid local minima
    n = np.random.randint(100, 1000)
    best_sequence = [np.random.random() * 10 for _ in range(n)]
    best_fitness = compute_inverse_c1(best_sequence)

    # Track elite sequences to preserve top performers
    elite_sequences = [best_sequence]
    elite_fitness = [best_fitness]

    # Run multiple iterations to find better solutions
    for iteration in range(20):
        # Preserve elite sequences periodically
        if iteration % 5 == 0 and iteration > 0:
            # Retain top 10% of sequences
            sorted_indices = sorted(range(len(elite_fitness)),
                                  key=lambda i: elite_fitness[i], reverse=True)
            elite_sequences = [elite_sequences[i] for i in sorted_indices[:len(elite_sequences)//10]]
            elite_fitness = [elite_fitness[i] for i in sorted_indices[:len(elite_sequences)//10]]

        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            new_fitness = compute_inverse_c1(h_function)
            if new_fitness > best_fitness:
                best_sequence = h_function
                best_fitness = new_fitness

                # Add to elite sequences
                elite_sequences.append(best_sequence)
                elite_fitness.append(best_fitness)
        else:
            # If we can't improve, try a new random sequence
            n = np.random.randint(100, 1000)
            new_sequence = [np.random.random() * 10 for _ in range(n)]
            new_fitness = compute_inverse_c1(new_sequence)
            if new_fitness > best_fitness:
                best_sequence = new_sequence
                best_fitness = new_fitness

                # Add to elite sequences
                elite_sequences.append(best_sequence)
                elite_fitness.append(best_fitness)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")