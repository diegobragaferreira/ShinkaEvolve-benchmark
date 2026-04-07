# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
from numba import njit
import time
import warnings

# Set seeds for reproducibility
np.random.seed(42)

@njit
def compute_autoconvolution_norms(f_values):
    """
    Compute the autoconvolution g = f*f and return its L2, L1, and L-infinity norms.
    Uses piecewise linear integration for L2 norm.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Use optimized NumPy convolution instead of manual loops
    g = np.convolve(f_values, f_values, mode='full')

    # Compute norms
    # L2 norm squared using more accurate piecewise integration
    l2_norm_squared = 0.0
    if len(g) >= 2:
        # For piecewise linear integration of g², we compute:
        # ∫ g² dx ≈ Σ (h/3)(y₁² + y₁y₂ + y₂²)
        # where h is the step size (assumed to be 1.0 for normalized case)
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            # Using trapezoidal rule for g² integration
            l2_norm_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm - integrate |g| using trapezoidal rule
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)  # Normalize by number of intervals

    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))

    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2(l2_norm_squared, l1_norm, l_inf_norm):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def construct_function() -> list[float]:
    """
    Quadratic Programming Approach to maximize C2.
    Instead of evolutionary algorithms, this approach uses convex optimization.
    """
    start_time = time.time()

    # Problem dimensions
    n = 1000  # Number of steps - fixed for consistent benchmarking

    # Initialize with a good starting point based on known patterns
    # Try to create a function that promotes flat autoconvolution profile
    initial_guess = np.ones(n) * 0.5

    # Create a structured pattern that tends to give good C2 values
    # Based on mathematical intuition: spread out peaks to reduce sharpness
    for i in range(n):
        if i % 4 == 0:
            initial_guess[i] = 1.0
        elif i % 4 == 1:
            initial_guess[i] = 0.8
        elif i % 4 == 2:
            initial_guess[i] = 0.6
        else:
            initial_guess[i] = 0.4

    # Add some randomness to escape local minima
    initial_guess += np.random.normal(0, 0.05, n)
    initial_guess = np.maximum(initial_guess, 0.0)  # Ensure non-negative

    # Define the objective function to maximize C2
    # We'll minimize the negative of C2, which is equivalent to maximizing C2
    def objective(x):
        try:
            l2_sq, l1, l_inf = compute_autoconvolution_norms(x)
            c2 = calculate_c2(l2_sq, l1, l_inf)
            return -c2  # Negative because we want to maximize
        except:
            return 1e10  # Large penalty for invalid solutions

    # Define constraints
    # All values must be non-negative (this is handled by bounds)
    bounds = [(0.0, None) for _ in range(n)]

    # Solve the optimization problem using SLSQP method which handles bounds well
    try:
        # Use a combination of initial guess and optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )

        # If optimization failed, return initial guess
        if not result.success:
            warnings.warn("Optimization failed, returning initial guess")
            final_solution = initial_guess
        else:
            final_solution = result.x

        # Ensure non-negativity in final result
        final_solution = np.maximum(final_solution, 0.0)

    except Exception as e:
        warnings.warn(f"Optimization error: {e}, returning initial guess")
        final_solution = initial_guess

    elapsed = time.time() - start_time
    if elapsed > 85:
        # Return a basic heuristic if time limit approached
        return [np.random.random() for _ in range(500)]

    # Convert back to list format
    return final_solution.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")