# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import numba
from typing import List

@numba.jit(nopython=True)
def compute_autoconvolution_norms(f_vals: np.ndarray, n_points: int = 15000) -> tuple:
    """
    Compute the norms for autoconvolution g = f*f using piecewise linear integration
    """
    # Create step function on [-1/4, 1/4] with given values
    step_width = 0.5 / len(f_vals)
    x = np.linspace(-0.25, 0.25, len(f_vals))

    # Create piecewise constant function
    f = np.zeros(n_points)
    x_grid = np.linspace(-0.25, 0.25, n_points)

    # Interpolate step function onto grid more accurately
    for i in range(len(f_vals)):
        start_x = -0.25 + i * step_width
        end_x = start_x + step_width
        start_idx = max(0, int((start_x + 0.25) / 0.5 * n_points))
        end_idx = min(n_points, int((end_x + 0.25) / 0.5 * n_points))
        if i == len(f_vals) - 1:
            end_idx = n_points
        f[start_idx:end_idx] = f_vals[i]

    # Compute autoconvolution g = f * f (discrete convolution)
    g = convolve(f, f[::-1], mode='full')
    g = g[len(g)//2:]  # Take positive half

    # Truncate to match x_grid size
    g = g[:n_points]

    # Compute norms using trapezoidal-like piecewise integration
    # For ||g||_2^2: integrate g^2 using improved trapezoidal rule
    g_squared = g * g
    # Using corrected trapezoidal rule: (y1 + y2) * h / 2
    # But since we're integrating g^2, we use a modified version
    # For accurate quadratic integration, we use Simpson's-like approach
    # Here we do weighted average of consecutive points for better precision
    h = x_grid[1] - x_grid[0]
    norm_g2_sq = 0.0
    for i in range(len(g_squared)-1):
        # Trapezoidal approximation for integral of g^2
        norm_g2_sq += (g_squared[i] + g_squared[i+1]) * h / 2

    # For ||g||_1
    norm_g1 = 0.0
    for i in range(len(g)):
        norm_g1 += np.abs(g[i]) * h

    # For ||g||_inf
    norm_ginf = 0.0
    for i in range(len(g)):
        if np.abs(g[i]) > norm_ginf:
            norm_ginf = np.abs(g[i])

    return norm_g2_sq, norm_g1, norm_ginf

@numba.jit(nopython=True)
def evaluate_c2(f_vals: np.ndarray) -> float:
    """Evaluate C2 = ||g||_2^2 / (||g||_1 * ||g||_inf)"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_vals)

    # Handle numerical issues
    if norm_g1 < 1e-15 or norm_ginf < 1e-15:
        return 0.0

    return norm_g2_sq / (norm_g1 * norm_ginf)

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using improved initialization and optimization."""
    # Initialize with improved pattern that concentrates more energy near center
    # This pattern has a sharper peak in the middle with controlled tails
    initial_guess = []
    n_steps = 150  # Increased number of steps for better resolution

    # Create better initial pattern with sharp center peak
    for i in range(n_steps):
        # Position normalized to [0,1]
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Create pattern with sharper peak and controlled tails
        # Central region with higher values, tapering edges
        if pos < 0.3 or pos > 0.7:
            # Tapered edges
            val = 0.2 * np.exp(-((pos - 0.5) * 8)**2) + 0.1
        else:
            # Sharp center region
            val = 1.5 * np.exp(-((pos - 0.5) * 5)**2) + 0.5
        initial_guess.append(max(0.0, val))

    # Normalize to control overall magnitude
    total = sum(initial_guess)
    if total > 0:
        initial_guess = [x/total * 2.0 for x in initial_guess]

    # Use scipy's differential evolution for global search
    def objective(f_vals):
        return -evaluate_c2(np.array(f_vals))  # Negative because we minimize

    # Set bounds for each parameter
    bounds = [(0.0, 3.0) for _ in range(n_steps)]

    try:
        # First phase: coarse global search with more iterations for better exploration
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=80,  # More iterations for better convergence
            popsize=20,   # Larger population for better diversity
            tol=1e-5
        )

        optimized_values = result.x

        # Second phase: refine with local optimization
        refined_result = minimize(
            objective,
            optimized_values,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}  # More iterations for refinement
        )

        final_values = refined_result.x
        # Clip negative values and normalize
        final_values = np.clip(final_values, 0, None)
        total = np.sum(final_values)
        if total > 0:
            final_values = final_values / total * 2.0

    except Exception:
        # Fallback to initial guess if optimization fails
        final_values = initial_guess

    # Convert to list of floats
    return [float(x) for x in final_values]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")