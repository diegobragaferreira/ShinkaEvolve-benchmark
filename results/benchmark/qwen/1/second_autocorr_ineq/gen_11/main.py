# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂² (L2 norm squared), ||g||₁ (L1 norm), ||g||∞ (L-infinity norm)
    
    Args:
        f_values: list of step function heights
        
    Returns:
        tuple: (c2_value, benchmark_ratio, eval_time)
    """
    start_time = time.time()
    
    # Convert to numpy array for efficient computation
    f = np.array(f_values)
    
    # Ensure non-negative values
    f = np.maximum(f, 0)
    
    # Compute autoconvolution g = f * f (discrete convolution)
    # Using 'full' mode to get complete convolution result
    g = convolve(f, f, mode='full')
    
    # The convolution result has length 2*n - 1
    # We center it properly for the interval [-1/4, 1/4]
    # The middle element corresponds to index n-1
    
    # Extract the central portion that represents the main interval
    n = len(f)
    middle_idx = n - 1
    half_width = n  # This should capture the main convolution support
    
    # Take the central part of the convolution
    g_centered = g[middle_idx - half_width + 1 : middle_idx + half_width]
    
    # Compute the norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² - integrate using trapezoidal rule manually for piecewise linear
    # Since we're dealing with discrete values, we approximate the integral
    # For a piecewise linear function with equal spacing, we use the trapezoidal rule
    # But since we don't know exact spacing, we just sum the squares
    norm_g_2_squared = np.sum(g_squared)
    
    # ||g||₁ - sum of absolute values
    norm_g_1 = np.sum(g_abs)
    
    # ||g||∞ - maximum absolute value
    norm_g_inf = np.max(g_abs)
    
    # Avoid division by zero
    if norm_g_1 == 0 or norm_g_inf == 0:
        c2 = 0.0
    else:
        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
    
    eval_time = time.time() - start_time
    benchmark_ratio = c2 / 0.962 if c2 > 0 else 0.0
    
    return c2, benchmark_ratio, eval_time

def evaluate_function(params):
    """
    Evaluate objective function for optimization
    """
    # Convert params to step function heights
    f_values = np.clip(params, 0, None)  # Ensure non-negative
    
    # Compute the C2 value
    c2, _, _ = compute_autoconvolution_norms(f_values)
    
    # Return negative because we want to maximize C2
    return -c2

def sophisticated_initialization():
    """
    Generate a sophisticated initial step function 
    based on mathematical insights about maximizing C2
    """
    n_steps = np.random.randint(500, 5000)
    
    # Create an initial pattern based on mathematical intuition
    # Try to balance flatness (good for L2/L1 ratio) with peakiness (good for L2/L_infty)
    f_values = []
    
    # Use multiple strategies to initialize
    strategy = np.random.choice(['uniform', 'alternating', 'gaussian'])
    
    if strategy == 'uniform':
        # Simple uniform distribution
        f_values = [0.5] * n_steps
    elif strategy == 'alternating':
        # Alternating high/low values
        for i in range(n_steps):
            if i % 2 == 0:
                f_values.append(np.random.uniform(0.7, 1.0))
            else:
                f_values.append(np.random.uniform(0.0, 0.3))
    else:  # gaussian
        # Create a bell-shaped pattern
        x = np.linspace(-1, 1, n_steps)
        mu, sigma = 0, 0.3
        gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        # Scale so that maximum is around 1
        scale_factor = 1.0 / np.max(gauss)
        f_values = (gauss * scale_factor).tolist()
        
    # Add some noise for exploration
    noise_level = 0.1
    f_values = [max(0, val + np.random.normal(0, noise_level)) for val in f_values]
    
    return f_values

def evolutionary_optimization():
    """
    Perform evolutionary optimization to find best step function
    """
    # Generate initial population
    initial_population = [sophisticated_initialization() for _ in range(10)]
    
    # Get dimensions from first individual
    n_dimensions = len(initial_population[0])
    
    # Set bounds for each parameter (step height)
    bounds = [(0, 1) for _ in range(n_dimensions)]
    
    # Perform differential evolution optimization
    result = differential_evolution(
        evaluate_function,
        bounds,
        maxiter=100,  # Limited iterations due to time constraint
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Extract optimized solution
    optimized_f_values = np.clip(result.x, 0, None).tolist()
    
    return optimized_f_values

def construct_function() -> list[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    Uses evolutionary algorithm to optimize the step function.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Use evolutionary optimization
    try:
        f_values = evolutionary_optimization()
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        f_values = sophisticated_initialization()
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
