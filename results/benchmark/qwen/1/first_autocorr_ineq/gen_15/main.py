# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random

def compute_autocorrelation_constant(sequence):
    """Computes the autocorrelation constant C₁ for a given sequence."""
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0

    # Compute convolution using FFT for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take the maximum of the convolution (excluding the zero-padding)
    max_conv = np.max(conv[len(sequence)-1:])

    # Calculate C₁ = 2*n*max(b) / (sum(a))^2
    sum_a = np.sum(sequence)
    n = len(sequence)

    if sum_a == 0:
        return 0.0

    C1 = 2 * n * max_conv / (sum_a ** 2)
    return 1 / C1  # Return reciprocal for maximization

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using finite difference approximation."""
    n = len(sequence)
    if n == 0:
        return None

    # Normalize sequence to avoid numerical issues
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Use a more principled normalization
    normalized_sequence = np.array(sequence) / sum_sequence

    # Compute current autocorrelation constant
    current_value = compute_autocorrelation_constant(sequence)

    # Approximate gradient using finite differences
    epsilon = 1e-4
    step_direction = np.zeros(n)

    for i in range(n):
        # Create perturbed sequence
        perturbed_sequence = normalized_sequence.copy()
        perturbed_sequence[i] += epsilon

        # Compute new value
        new_value = compute_autocorrelation_constant(perturbed_sequence * sum_sequence)

        # Gradient approximation
        step_direction[i] = (new_value - current_value) / epsilon

    # Normalize the step direction
    step_norm = np.linalg.norm(step_direction)
    if step_norm > 0:
        step_direction = step_direction / step_norm

    # Move in the direction of steepest ascent
    t = 0.01
    new_sequence = (1 - t) * np.array(sequence) + t * step_direction * sum_sequence

    # Ensure non-negativity and clip values
    new_sequence = np.clip(new_sequence, 0, 1000)

    return new_sequence.tolist()

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build the convolution constraint matrix
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def initialize_good_sequence(length=None):
    """Initialize a good starting sequence."""
    if length is None:
        length = random.randint(100, 1000)

    # Create a sequence with exponential decay to balance mass and convolution
    # This helps to reduce the peak convolution while maintaining significant total mass
    sequence = [1000 * np.exp(-i/10) for i in range(length)]

    # Normalize to have reasonable total mass
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]

    return sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Multi-start approach to escape local optima
    best_score = 0
    best_sequence = None

    # Try multiple starting points
    for _ in range(10):
        # Initialize with a better structured sequence
        sequence = initialize_good_sequence()

        # Allow some iterations of improvement
        for _ in range(50):
            h_function = get_good_direction_to_move_into(sequence)
            if h_function is not None:
                sequence = h_function
            else:
                break

        # Evaluate final sequence
        score = compute_autocorrelation_constant(sequence)
        if score > best_score:
            best_score = score
            best_sequence = sequence[:]

    # If no good sequence found, return a default one
    if best_sequence is None:
        best_sequence = [1.0] * 100

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")