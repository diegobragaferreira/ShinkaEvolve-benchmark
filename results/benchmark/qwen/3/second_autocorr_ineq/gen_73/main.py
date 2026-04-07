# EVOLVE-BLOCK-START

import numpy as np
import cvxpy as cp
from scipy.optimize import differential_evolution
from numba import njit

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the autoconvolution g = f*f and return its norms.
    Uses fast numba-compiled operations.
    """
    n = len(f_vals)
    # Autoconvolution using direct computation
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared)
    norm_g1 = np.sum(np.abs(g))
    norm_g_inf = np.max(np.abs(g))

    return norm_g2_squared, norm_g1, norm_g_inf

@njit
def calculate_c2(f_vals):
    """
    Calculate C2 value for given step function values.
    """
    norm_g2_squared, norm_g1, norm_g_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_g1 < 1e-15 or norm_g_inf < 1e-15:
        return 0.0

    c2 = norm_g2_squared / (norm_g1 * norm_g_inf)
    return c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using sparse convex optimization approach."""
    
    # Use a hybrid approach: start with an intelligent initialization based on 
    # the principle that we want a distribution that leads to a flatter autoconvolution
    # We'll use a combination of geometric distribution and targeted peak placement
    
    # Set up problem dimensions
    n_steps = np.random.randint(500, 2000)  # Larger range for better exploration
    
    # Generate a sparse initialization that's more likely to lead to good solutions
    # Based on mathematical understanding: we want peaks that don't create too sharp spikes
    # A good strategy is to place values with geometric decay and a bit of randomness
    
    # Create base function with geometric distribution
    base_vals = np.geomspace(1, 0.01, num=n_steps//2)
    
    # Add some randomness to avoid local optima trapping
    noise = np.random.exponential(0.1, n_steps//2)
    
    # Combine and ensure positivity
    f_vals = np.concatenate([base_vals, noise])
    
    # Trim to exact length
    if len(f_vals) > n_steps:
        f_vals = f_vals[:n_steps]
    elif len(f_vals) < n_steps:
        # Pad with zeros
        f_vals = np.pad(f_vals, (0, n_steps - len(f_vals)), 'constant')
    
    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0.0)
    
    # Normalize to avoid extreme values
    if np.sum(f_vals) > 0:
        f_vals = f_vals / np.sum(f_vals) * 100
    
    # Apply a final optimization using differential evolution for global search
    # This helps us escape local optima from the initialization
    def objective(f_vals_list):
        # Convert to numpy array for computation
        vals = np.array(f_vals_list)
        # Return negative since we're minimizing
        return -calculate_c2(vals)
    
    # Use differential evolution for robust global optimization
    bounds = [(0.0, 100.0)] * n_steps
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42
        )
        
        if result.success:
            optimized_values = result.x
            # Ensure non-negativity
            optimized_values = np.maximum(optimized_values, 0.0)
            return optimized_values.tolist()
        else:
            return f_vals.tolist()
            
    except Exception:
        return f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
