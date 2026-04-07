# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import numba
from typing import List
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import qr

@numba.jit(nopython=True)
def compute_autoconvolution_norms_sparse(f_vals: np.ndarray, n_points: int = 10000) -> tuple:
    """
    Optimized computation of autoconvolution norms using sparse representation principles
    """
    # Create step function on [-1/4, 1/4] with given values
    step_width = 0.5 / len(f_vals)

    # More efficient grid creation
    x_grid = np.linspace(-0.25, 0.25, n_points)

    # Create piecewise constant function using vectorized operations
    f = np.zeros(n_points)

    # Vectorized assignment of step values
    for i in range(len(f_vals)):
        start_idx = int(i * n_points / len(f_vals))
        end_idx = int((i + 1) * n_points / len(f_vals))
        if i == len(f_vals) - 1:
            end_idx = n_points
        f[start_idx:end_idx] = f_vals[i]

    # Efficient convolution using FFT-based approach
    # Create extended arrays for proper convolution
    f_extended = np.pad(f, (0, len(f)), mode='constant', constant_values=0)
    g = np.convolve(f_extended, f_extended[::-1], mode='valid')[:n_points]

    # Compute norms with optimized vectorized operations
    g_squared = g * g
    h = x_grid[1] - x_grid[0]

    # More accurate norm computation using composite trapezoidal rule
    norm_g2_sq = h * (g_squared[0] + g_squared[-1]) / 2 + h * np.sum(g_squared[1:-1])

    # ||g||_1
    norm_g1 = h * np.sum(np.abs(g))

    # ||g||_inf
    norm_ginf = np.max(np.abs(g))

    return norm_g2_sq, norm_g1, norm_ginf

@numba.jit(nopython=True)
def evaluate_c2_sparse(f_vals: np.ndarray) -> float:
    """Evaluate C2 = ||g||_2^2 / (||g||_1 * ||g||_inf) with improved numerical stability"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms_sparse(f_vals)

    # Handle numerical issues with better thresholds
    if norm_g1 < 1e-15 or norm_ginf < 1e-15:
        return 0.0

    return norm_g2_sq / (norm_g1 * norm_ginf)

def construct_function() -> List[float]:
    """Construct step-function with high C2 value using sparse convex optimization approach."""

    # Use dimensionality reduction to identify key components
    # Generate initial candidate basis functions
    n_steps = 150

    # Create basis functions (Gaussian-like shapes)
    basis_functions = []

    # Generate radial basis functions centered at different positions
    centers = np.linspace(0.05, 0.95, 15)  # 15 centers across domain
    widths = np.logspace(-2, -0.5, 10)     # 10 different widths

    for center in centers:
        for width in widths:
            # Create Gaussian-like basis function
            x = np.linspace(0, 1, n_steps)
            basis = np.exp(-((x - center) / width) ** 2) * 0.5 + 0.2
            basis_functions.append(basis)

    # Use SVD to find most important combinations
    basis_matrix = np.array(basis_functions).T
    svd = TruncatedSVD(n_components=min(50, len(basis_functions)), random_state=42)
    compressed_basis = svd.fit_transform(basis_matrix)
    reconstructed_basis = svd.inverse_transform(compressed_basis)

    # Create optimized sparse representation
    # Sample from the compressed space to get good candidates
    sample_indices = np.random.choice(len(reconstructed_basis), size=25, replace=False)
    initial_candidates = [reconstructed_basis[i] for i in sample_indices]

    # Combine to form final step function
    # Use convex combination approach
    weights = np.random.rand(len(initial_candidates))
    weights = weights / np.sum(weights)

    # Create final step function as weighted combination
    final_function = np.zeros(n_steps)
    for i, (candidate, weight) in enumerate(zip(initial_candidates, weights)):
        final_function += weight * np.clip(candidate, 0, None)

    # Normalize the function
    total = np.sum(final_function)
    if total > 0:
        final_function = final_function / total * 2.0

    # Refine using gradient-based optimization with constraints
    def objective(x):
        return -evaluate_c2_sparse(x)

    # Constraints and bounds
    bounds = [(0.0, 3.0) for _ in range(n_steps)]

    try:
        # Use L-BFGS-B for refinement on the reduced representation
        result = minimize(
            objective,
            final_function,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )

        final_values = result.x
        final_values = np.clip(final_values, 0, None)

        # Normalize again
        total = np.sum(final_values)
        if total > 0:
            final_values = final_values / total * 2.0

    except Exception as e:
        # Fallback to initial guess
        final_values = final_function

    # Return as list of floats
    return [float(x) for x in final_values]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")