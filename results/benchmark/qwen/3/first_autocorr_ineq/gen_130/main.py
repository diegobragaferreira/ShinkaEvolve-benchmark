# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random

# Global variables for curvature tracking
curvature_history = []
hessian_approx = None

def compute_hessian_approximation(sequence, epsilon=1e-5):
    """Approximate the Hessian using finite differences for curvature awareness."""
    n = len(sequence)
    approx_hessian = np.zeros((n, n))

    for i in range(n):
        # Perturb dimension i
        delta = np.zeros(n)
        delta[i] = epsilon

        # Forward difference approximation
        seq_plus = np.array(sequence) + delta
        seq_minus = np.array(sequence) - delta

        # Compute gradients for these points
        _, grad_plus = compute_gradient_at_point(seq_plus)
        _, grad_minus = compute_gradient_at_point(seq_minus)

        # Finite difference for Hessian column
        approx_hessian[:, i] = (grad_plus - grad_minus) / (2 * epsilon)

    return approx_hessian

def compute_gradient_at_point(sequence):
    """Helper to compute objective and gradient at a point (used for Hessian)."""
    # Simplified version for gradient estimation
    n = len(sequence)
    sum_seq = np.sum(sequence)

    if sum_seq < 1e-10:
        return 0.0, np.zeros(n)

    # Compute convolution
    if n > 100:
        conv_result = fftconvolve(sequence, sequence, mode='full')
        conv_result = conv_result[:2*n-1]
    else:
        conv_result = np.convolve(sequence, sequence)

    max_conv = np.max(conv_result)
    objective = -(2 * n * max_conv) / (sum_seq ** 2)  # Negated for maximization

    # Simplified gradient calculation (this could be made more accurate with symbolic differentiation)
    grad = np.full(n, -objective / sum_seq)

    return objective, grad

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with enhanced adaptive parameters and curvature correction."""
    global curvature_history, hessian_approx

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

    # Apply curvature-aware correction to improve direction
    corrected_direction = np.array(normalized_g_fun)

    # Store recent curvature information
    curvature_history.append(corrected_direction.copy())
    if len(curvature_history) > 5:
        curvature_history.pop(0)

    # Compute Hessian approximation for curvature info
    if n > 10 and len(curvature_history) >= 3:
        try:
            hessian_approx = compute_hessian_approximation(sequence)
            # Use curvature for adaptive step size
            # Simple approach: if Hessian is nearly singular, reduce step size
            if np.linalg.det(hessian_approx) < 1e-8:
                corrected_direction *= 0.5  # Reduce step if flat region detected
        except:
            pass  # Continue without curvature correction if computation fails

    # Adaptive step-size with convergence awareness
    t = min(0.05, 0.01 + 0.01 * np.log(n + 1))

    # Apply curvature correction to step size
    if hessian_approx is not None and n > 10:
        try:
            # Adjust step if the Hessian is close to singular
            cond = np.linalg.cond(hessian_approx)
            if cond > 1e6:
                t *= 0.5  # Reduce step if ill-conditioned
        except:
            pass  # Skip step adjustment if computation fails

    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, corrected_direction)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS with enhanced constraints."""
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Dynamic decision for convolution method based on numerical behavior
    conv_method = 'fft' if n > 100 else 'direct'
    if conv_method == 'fft':
        # Use FFT convolution but check for numerical issues
        try:
            f_conv = fftconvolve(f_sequence, f_sequence, mode='full')
            f_conv = f_conv[:2*n-1]
            # Validate that convolution values are within expected ranges
            if np.any(np.isnan(f_conv)) or np.any(np.isinf(f_conv)):
                conv_method = 'direct'
        except:
            conv_method = 'direct'

    if conv_method == 'direct':
        # Use direct convolution for better numerical control
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

    # Try multiple methods to solve LP, with improved error handling
    try:
        # Try the 'highs' method first for better performance
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                                 options={'presolve': True, 'maxiter': 1000})
    except Exception as e:
        # Fallback to 'simplex' with relaxed tolerances
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

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with enhanced diversity."""
    # Initialize with deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Enhanced initialization: use multiple strategies for better diversity
    init_strategies = []

    # Strategy 1: Random uniform distribution
    n_init1 = max(10, int(math.log(1000) * 50))
    seq1 = np.random.rand(n_init1).tolist()

    # Strategy 2: Exponential distribution for heavy-tailed distributions
    n_init2 = max(10, int(math.log(500) * 30))
    seq2 = [np.random.exponential(1.0) for _ in range(n_init2)]

    # Strategy 3: Gamma distribution for varied shapes
    n_init3 = max(10, int(math.log(800) * 40))
    seq3 = [np.random.gamma(2.0, 1.0) for _ in range(n_init3)]

    # Combine strategies and ensure positivity
    combined_seq = seq1 + seq2 + seq3
    combined_seq = [max(x, 0.01) for x in combined_seq]

    # Select best initial sequence
    best_sequence = combined_seq[:100]  # Cap at 100 elements

    # Evolution loop with adaptive refinement
    max_iterations = 50  # Increase iterations for better convergence
    for iteration in range(max_iterations):
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # If evolution fails, try to recover with a more robust mutation
            index = np.random.randint(len(best_sequence))
            # Use multiplicative mutation with larger perturbations
            perturbation = np.random.normal(1, 0.2)  # Allow larger variations
            best_sequence[index] = max(0.01, best_sequence[index] * perturbation)

        # Periodic diversity injection
        if iteration % 10 == 0:
            # Inject some new diversity every 10 iterations
            new_index = np.random.randint(len(best_sequence))
            best_sequence[new_index] = max(0.01, np.random.exponential(1.0))

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")