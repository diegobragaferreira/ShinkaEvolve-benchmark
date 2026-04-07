# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from numba import jit, prange
import nevergrad as ng
from scipy import signal

# Set random seeds for reproducibility
np.random.seed(42)

@jit(nopython=True, fastmath=True)
def compute_autoconvolution_numba(f_vals):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size, dtype=np.float64)

    # Compute convolution directly with numba optimization
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True, fastmath=True)
def compute_trapezoidal_norms_numba(g_vals, dx):
    """
    Compute trapezoidal norms efficiently with numba
    """
    n = len(g_vals)

    # For L2 norm squared using trapezoidal rule
    l2_norm_sq = 0.0
    if n >= 2:
        # Trapezoidal rule: sum of (y[i]^2 + y[i+1]^2)/2 * dx
        for i in range(n-1):
            l2_norm_sq += (g_vals[i] * g_vals[i] + g_vals[i+1] * g_vals[i+1]) * dx / 2.0
    elif n == 1:
        l2_norm_sq = g_vals[0] * g_vals[0] * dx

    # For L1 norm using trapezoidal rule (average of adjacent heights * dx)
    l1_norm = 0.0
    if n >= 2:
        for i in range(n-1):
            l1_norm += (abs(g_vals[i]) + abs(g_vals[i+1])) * dx / 2.0
    elif n == 1:
        l1_norm = abs(g_vals[0]) * dx

    # Infinity norm
    linf_norm = 0.0
    for i in range(n):
        val = abs(g_vals[i])
        if val > linf_norm:
            linf_norm = val

    return l1_norm, l2_norm_sq, linf_norm

def compute_autoconvolution_norms(f_values: List[float]):
    """
    Compute the three norms needed for C₂ calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4]
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    dx = 0.5 / n_steps  # Step size

    # Create piecewise constant function from step heights
    f = np.array(f_values)

    # Compute autoconvolution g = f * f using numba optimized version
    g = compute_autoconvolution_numba(f)

    # Adjust indices for the correct domain
    # Result has length 2*n_steps - 1
    g_len = len(g)

    # Extract the central region corresponding to [-1/4, 1/4]
    # This takes the middle n_steps elements of the full convolution
    central_start = (g_len - n_steps) // 2
    central_end = central_start + n_steps
    g_centered = g[central_start:central_end]

    # Compute norms using numba optimized version with trapezoidal integration
    g_abs = np.abs(g_centered)

    # Compute norms using numba - using trapezoidal rules for more accurate integration
    norm_1, norm_2_sq, norm_inf = compute_trapezoidal_norms_numba(g_abs, dx)

    return norm_2_sq, norm_1, norm_inf

def evaluate_c2(params):
    """Evaluate fitness for individual (step function heights)"""
    try:
        # Convert params to list of floats
        f_values = [max(0.0, float(x)) for x in params]  # Ensure non-negative

        # Compute the norms
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0

        # Calculate C₂
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        return 0.0

def generate_harmonic_initial_function(num_steps: int) -> List[float]:
    """Generate an initial function based on harmonic analysis principles"""
    # Create a smooth, oscillating pattern that should produce good autoconvolution properties
    # Start with a sinusoidal pattern that decays to create a nice distribution
    
    # Base frequencies for harmonics
    frequencies = [1, 2, 3, 4, 5]
    amplitudes = [1.0, 0.7, 0.5, 0.3, 0.2]
    
    # Generate x values from -0.25 to 0.25
    x = np.linspace(-0.25, 0.25, num_steps, endpoint=False) + 0.5/num_steps/2
    
    # Create function as sum of harmonics
    f_vals = np.zeros(num_steps)
    for freq, amp in zip(frequencies, amplitudes):
        f_vals += amp * np.sin(2 * np.pi * freq * x)
    
    # Add polynomial decay component
    x_centered = x / 0.25
    decay = 1.0 - x_centered**2
    f_vals *= decay
    
    # Ensure non-negative values
    f_vals = np.maximum(f_vals, 0.0)
    
    # Normalize to prevent extreme peaks
    if np.sum(f_vals) > 0:
        f_vals = f_vals / np.sum(f_vals) * num_steps * 0.5
    
    return f_vals.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using hybrid optimization."""
    
    TIME_LIMIT = 85  # seconds
    start_time = time.time()
    
    # First, try to get a good initial guess with harmonic functions
    initial_guess_size = np.random.randint(200, 1000)
    initial_guess = generate_harmonic_initial_function(initial_guess_size)
    
    # Use Nevergrad optimizer for black-box optimization
    # Define the optimization problem
    dimension = len(initial_guess)
    
    # Create optimizer with adaptive algorithm
    optimizer = ng.optimizers.CMA(dimension=dimension, budget=min(1000, max(100, dimension*10)))
    
    # Set up the evaluation function for Nevergrad
    def objective_function(x):
        # x is the parameter vector (step heights)
        # We convert to list and evaluate
        fitness = evaluate_c2(x)
        # Nevergrad maximizes, so we minimize negative
        return -fitness
    
    # Set initial point
    optimizer.suggest(initial_guess)
    
    # Optimize
    try:
        # Run optimization for limited time
        while time.time() - start_time < TIME_LIMIT - 1:
            candidate = optimizer.ask()
            fitness = objective_function(candidate)
            optimizer.tell(candidate, fitness)
            
            # Check if we're close to time limit
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
    except Exception as e:
        pass
    
    # Get the best result
    try:
        best_x = optimizer.provide_recommendation().value
        best_f_values = [max(0.0, float(x)) for x in best_x]
    except:
        # Fallback to initial guess
        best_f_values = initial_guess
    
    # Final validation and refinement
    try:
        final_fitness = evaluate_c2(best_f_values)
        if final_fitness <= 0:
            # If the result is bad, try a simpler approach
            simple_size = np.random.randint(200, 1000)
            simple_guess = [np.random.random() for _ in range(simple_size)]
            return simple_guess
    except:
        # If there was an error, return a valid random function
        fallback_size = np.random.randint(200, 1000)
        return [np.random.random() for _ in range(fallback_size)]
    
    return best_f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")