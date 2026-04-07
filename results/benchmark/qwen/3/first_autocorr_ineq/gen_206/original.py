# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with adaptive parameters."""
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
    else:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)

    rhs = np.max(conv_result)

    # Try solving LP with adaptive constraints
    g_fun = solve_convolution_lp(normalized_sequence, rhs, n)

    if g_fun is None:
        # Fallback: try with modified RHS
        rhs_fallback = rhs * 1.1
        g_fun = solve_convolution_lp(normalized_sequence, rhs_fallback, n)

    if g_fun is None:
        # Final fallback: simple gradient ascent
        t = 0.01
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize g_fun
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * adaptive_factor / sum_g_fun for x in g_fun]

    # Adaptive step-size
    t = min(0.05, 0.01 + 0.005 * np.log(n + 1))
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS with adaptive constraints and curvature awareness."""
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Generate convolution constraints with adaptive handling
    if n > 100:
        # For large sequences, use FFT to get convolution constraints efficiently
        f_conv = fftconvolve(f_sequence, f_sequence, mode='full')
        f_conv = f_conv[:2*n-1]
    else:
        f_conv = np.convolve(f_sequence, f_sequence)

    # Create constraint matrix
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    # Adaptive tolerance for optimization
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except:
        # Fallback to 'simplex' method if 'highs' fails
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex')
        except:
            return None

    if result.success:
        g_sequence = result.x
        # Ensure non-negativity and reasonable values
        g_sequence = np.maximum(g_sequence, 0)
        if np.sum(g_sequence) < 1e-10:
            return None

        # Apply curvature-aware correction if sequence is large enough
        if n > 50:
            # Estimate curvature using finite differences more carefully
            epsilon = 1e-4
            # Compute finite differences along each dimension
            curvature_correction = np.zeros(n)

            # For each dimension, compute second derivative approximation
            for i in range(n):
                # Create perturbed sequences
                perturbed_plus = f_sequence.copy()
                perturbed_minus = f_sequence.copy()

                # Small perturbations in both directions
                perturbed_plus[i] += epsilon
                perturbed_minus[i] -= epsilon

                # Compute max convolutions for each perturbed sequence
                if n > 100:
                    b_plus = fftconvolve(perturbed_plus, perturbed_plus, mode='full')[:2*n-1]
                    b_minus = fftconvolve(perturbed_minus, perturbed_minus, mode='full')[:2*n-1]
                else:
                    b_plus = np.convolve(perturbed_plus, perturbed_plus)
                    b_minus = np.convolve(perturbed_minus, perturbed_minus)

                # Compute second derivative approximation: f''(x) ≈ [f(x+h) - 2*f(x) + f(x-h)] / h^2
                second_derivative = (np.max(b_plus) + np.max(b_minus) - 2 * np.max(f_conv)) / (epsilon ** 2)
                curvature_correction[i] = max(0, second_derivative)  # Ensure non-negative

            # Apply curvature correction to the direction
            # Combine curvature with original solution using weighted average
            correction_weight = 0.05  # Small weight to prevent over-correction
            corrected_direction = g_sequence + correction_weight * curvature_correction

            # Ensure non-negativity of corrected direction
            corrected_direction = np.maximum(corrected_direction, 0)

            # Normalize corrected direction to maintain proper scaling
            sum_corrected = np.sum(corrected_direction)
            if sum_corrected > 0:
                corrected_direction = corrected_direction * (np.sum(g_sequence) / sum_corrected)

            g_sequence = corrected_direction

        return g_sequence
    else:
        return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with adaptive search."""
    # Adaptive initialization with log scaling
    n_init = max(10, int(math.log(1000) * 50))
    best_sequence = np.random.rand(n_init).tolist()

    # Add some diversity with small positive values
    for i in range(len(best_sequence)):
        if best_sequence[i] < 0.01:
            best_sequence[i] = 0.01

    # Evolve for a few iterations with adaptive strategy
    for _ in range(20):  # Reduced iterations to save time
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # If evolution fails, try to recover with a simple mutation
            index = np.random.randint(len(best_sequence))
            best_sequence[index] = max(0.01, best_sequence[index] * (1 + np.random.normal(0, 0.1)))

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")