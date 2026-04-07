# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple
import time
import nevergrad as ng
from scipy import integrate

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for efficient computation
    f = np.array(f_values)

    # Compute autoconvolution g = f * f using fast convolution
    g = signal.convolve(f, f, mode='full')

    # Only keep the middle portion corresponding to valid convolution
    center_idx = len(g) // 2
    half_len = len(f)
    g = g[center_idx - half_len + 1:center_idx + half_len]

    # Compute norms using proper numerical integration for accuracy
    # L2 norm squared using trapezoidal rule for piecewise linear approximation
    g_squared = g * g
    # Use scipy's integration for more precise computation
    x = np.linspace(-0.5, 0.5, len(g))
    norm_l2_sq = integrate.simpson(g_squared, x)
    
    # L1 norm
    norm_l1 = integrate.simpson(np.abs(g), x)
    
    # L-infinity norm
    norm_linf = np.max(np.abs(g))

    return norm_l2_sq, norm_l1, norm_linf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    try:
        norm_l2_sq, norm_l1, norm_linf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_l1 <= 1e-12 or norm_linf <= 1e-12:
            return 0.0

        c2 = norm_l2_sq / (norm_l1 * norm_linf)
        return c2
    except Exception:
        return 0.0

def smart_step_function_constructor(n_steps: int = 200) -> List[float]:
    """
    Construct step function using Chebyshev nodes for smoother convolution
    This creates a more favorable convolution structure
    """
    # Generate Chebyshev nodes for better distribution
    theta = np.pi * np.arange(1, n_steps + 1) / (n_steps + 1)
    cheb_nodes = np.cos(theta)
    
    # Map to [-1/4, 1/4] interval
    scaled_nodes = cheb_nodes * 0.125
    
    # Create step function - use a smooth profile that will produce good convolution properties
    # We'll use a triangular shape centered at 0 with peaks at the Chebyshev nodes
    # But since the nodes are positions, we create a step function with heights based on a smooth function
    base_heights = 0.5 + 0.5 * np.sin(2 * np.pi * scaled_nodes)  # Oscillating pattern
    # Apply a smooth weighting function that gives higher weights to central values
    central_weight = 1.0 + 0.5 * np.exp(-10 * scaled_nodes**2)  
    heights = base_heights * central_weight
    
    # Ensure non-negativity and normalize
    heights = np.maximum(heights, 0)
    # Normalize to prevent extreme values that might cause numerical issues
    heights = heights / (np.sum(heights) + 1e-10) * 10
    
    return heights.tolist()

def gradient_free_construct_function() -> List[float]:
    """Construct step function using gradient-free optimization with Nevergrad"""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Use a smarter initialization
    initial_guess = smart_step_function_constructor(200)
    
    # Define the optimization problem
    def objective(x):
        # Clip negative values
        x = np.maximum(x, 0)
        # Calculate C2 (we want to maximize it, so minimize negative)
        return -calculate_c2(x.tolist())
    
    # Create optimizer with appropriate settings
    # OnePlusOne is efficient for small problems, CMA-ES works well for general problems
    optimizer = ng.optimizers.OnePlusOne(
        dimension=len(initial_guess),
        budget=500,  # Limit evaluations to stay within time constraints
        num_workers=1
    )
    
    # Optimize
    recommendation = optimizer.minimize(objective, verbosity=0)
    
    # Get the best solution
    result = recommendation.args[0]
    
    # Ensure non-negativity
    result = np.maximum(result, 0)
    
    # Return as list
    return result.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Try gradient-free optimization first
    start_time = time.time()
    try:
        result = gradient_free_construct_function()
        elapsed = time.time() - start_time
        if elapsed < 85:  # Leave some margin for final calculations
            return result
    except Exception as e:
        pass

    # Fallback to simpler approach if optimization fails or times out
    return [random.uniform(0, 1) for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")