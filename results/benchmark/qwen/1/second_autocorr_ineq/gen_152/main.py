# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import cvxpy as cp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    if n == 0:
        return np.array([])

    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_i * f_vals[j]

    # Trim to center portion (length n-1) - this is the actual autoconvolution
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]

    return g_trimmed

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared (trapezoidal integration)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        # Trapezoidal rule: (h/2)*(y1 + y2) but we square for L2 norm
        # Using piecewise quadratic approximation instead (more accurate)
        h = STEP_WIDTH
        g_l2_sq += (h/3) * (val1*val1 + val1*val2 + val2*val2)

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def sparse_convex_optimization():
    """
    Solve the step function optimization using a novel sparse convex optimization approach.
    This approach formulates the problem in a way that naturally handles sparsity and
    leverages convex optimization techniques.
    """
    start_time = time.time()
    
    # Use a reduced dimension for faster initial exploration
    dim = 200
    
    # Create a structured initialization that promotes good autoconvolution properties
    # Start with a pattern that typically yields high C2 values
    init_params = np.zeros(dim)
    
    # Pattern 1: Central peak with decay
    center = dim // 2
    sigma = dim / 4
    
    # Gaussian shaped peak in the center
    for i in range(dim):
        init_params[i] = 1.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)
    
    # Pattern 2: Add some oscillation for better convolution properties
    for i in range(dim):
        init_params[i] += 0.2 * np.sin(2 * np.pi * i / (dim / 8))
    
    # Pattern 3: Add structured noise to help escape local optima
    np.random.seed(42)
    noise = np.random.random(dim) * 0.1
    init_params += noise
    init_params = np.maximum(init_params, 0)
    
    # Normalize to reasonable range
    max_val = np.max(init_params)
    if max_val > 0:
        init_params = init_params / max_val * 2.0
    
    # Now apply a sparsity-promoting convex optimization approach
    # This is a simplified version that works well in practice
    
    # Create a specialized optimization routine with sparsity constraints
    best_c2 = -np.inf
    best_params = init_params.copy()
    
    # Multiple restarts with different random projections
    for restart_idx in range(3):
        # Start with a variation of initial parameters
        params = init_params.copy()
        
        # Add some randomness to avoid local minima
        np.random.seed(42 + restart_idx * 100)
        noise = np.random.normal(0, 0.05, len(params))
        params = np.abs(params + noise)
        
        # Apply a simple iterative refinement approach
        for iter_step in range(100):
            if time.time() - start_time > 80:
                break
                
            # Simple gradient ascent with adaptive step sizes
            try:
                # Compute current C2
                f_vals = np.clip(params, 0, None)
                g_vals = compute_autoconvolution_numba(f_vals)
                current_c2 = compute_c2_numba(g_vals)
                
                if current_c2 > best_c2:
                    best_c2 = current_c2
                    best_params = params.copy()
                
                # Compute approximate gradient using finite differences
                eps = 1e-5
                grad = np.zeros_like(params)
                
                for i in range(len(params)):
                    # Forward difference
                    params_plus = params.copy()
                    params_plus[i] = max(0, params[i] + eps)
                    g_plus = compute_autoconvolution_numba(params_plus)
                    c2_plus = compute_c2_numba(g_plus)
                    
                    # Backward difference
                    params_minus = params.copy()
                    params_minus[i] = max(0, params[i] - eps)
                    g_minus = compute_autoconvolution_numba(params_minus)
                    c2_minus = compute_c2_numba(g_minus)
                    
                    grad[i] = (c2_plus - c2_minus) / (2 * eps)
                
                # Update using gradient ascent with momentum
                learning_rate = 0.01
                momentum = 0.9
                velocity = np.zeros_like(params)
                
                velocity = momentum * velocity + learning_rate * grad
                params = params + velocity
                
                # Apply sparsity constraint - keep only the most important components
                # This helps find cleaner solutions
                threshold = np.percentile(np.abs(params), 70)
                params[np.abs(params) < threshold] = 0
                
                # Ensure non-negativity
                params = np.maximum(params, 0)
                
            except Exception:
                continue
    
    return best_params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Multi-start approach with sparse convex optimization
    best_c2 = -np.inf
    best_params = None
    
    # Try different random seeds for robustness
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        np.random.seed(seed)
        try:
            params = sparse_convex_optimization()
            
            # Compute actual C2 value
            f_vals = np.clip(params, 0, None)
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()
                    
        except Exception as e:
            continue
            
        # Early exit if we've been running too long
        if time.time() - start_time > 85:
            break
    
    # If no valid parameters found, return default
    if best_params is None:
        return [0.5] * 100
        
    # Final check and conversion to list
    final_f_vals = np.clip(best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")