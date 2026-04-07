# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit
import math

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute L2^2 norm using trapezoidal integration
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # L-infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # L2^2 norm using trapezoidal rule
    if len(g_vals) >= 2:
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2 with numerical stability checks
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def evaluate_convolution_profile(f_vals):
    """Evaluate the convolution profile and return C2 metric"""
    try:
        # Ensure non-negative values
        f_vals = np.clip(f_vals, 0, None)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute C2
        c2 = compute_c2_numba(g_vals)
        
        return c2
    except Exception:
        return 0.0

def generate_step_function_from_segments(segments_data, total_points=1000):
    """Generate step function from segments data"""
    # segments_data: list of tuples (start_pos, end_pos, height)
    f_vals = np.zeros(total_points)
    
    # Normalize segments to point indices
    for start_pos, end_pos, height in segments_data:
        start_idx = int(start_pos * total_points)
        end_idx = int(end_pos * total_points)
        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_points-1))
        end_idx = max(0, min(end_idx, total_points))
        
        if start_idx < end_idx:
            f_vals[start_idx:end_idx] = height
    
    return f_vals

def create_structured_pattern(base_size=1000):
    """Create a structured pattern designed to maximize convolution properties"""
    # Create a pattern with multiple peaks and valleys that create beneficial convolution shapes
    f_vals = np.zeros(base_size)
    
    # Add multiple peaks with varying heights and widths
    peaks = [
        (0.1, 0.3, 1.0),
        (0.4, 0.6, 0.7),
        (0.7, 0.9, 1.0),
        (0.2, 0.4, 0.3),
        (0.6, 0.8, 0.3)
    ]
    
    for start, end, height in peaks:
        start_idx = int(start * base_size)
        end_idx = int(end * base_size)
        if start_idx < end_idx:
            f_vals[start_idx:end_idx] = height
    
    # Add some noise to break symmetry and promote diversity
    noise = np.random.normal(0, 0.05, base_size)
    f_vals = np.clip(f_vals + noise, 0, None)
    
    return f_vals

def adaptive_evolution_step(current_solution, iteration):
    """Adaptively evolve solution with different intensities based on iteration"""
    mutated = current_solution.copy()
    
    # Vary mutation intensity based on iteration
    mutation_intensity = max(0.01, 0.1 * (1.0 - iteration / 100.0))
    
    # Randomly modify some elements
    for i in range(len(mutated)):
        if np.random.random() < 0.1:  # 10% chance to mutate
            # Add normally distributed noise
            noise = np.random.normal(0, mutation_intensity * mutated[i] if mutated[i] > 0 else 0.1)
            mutated[i] = max(0, mutated[i] + noise)
    
    return mutated

def convolution_guided_evolution():
    """Main evolutionary algorithm guided by convolution properties"""
    best_c2 = -np.inf
    best_solution = None
    
    # Initialize with structured patterns
    initial_patterns = []
    for i in range(5):
        pattern = create_structured_pattern(1000)
        initial_patterns.append(pattern)
    
    # Evaluate initial patterns
    for i, pattern in enumerate(initial_patterns):
        c2 = evaluate_convolution_profile(pattern)
        if c2 > best_c2:
            best_c2 = c2
            best_solution = pattern.copy()
    
    # Evolution loop
    for iteration in range(100):
        if time.time() - start_time > 85:
            break
            
        # Create new candidate through mutation
        candidate = adaptive_evolution_step(best_solution, iteration)
        c2_candidate = evaluate_convolution_profile(candidate)
        
        # Accept better solutions
        if c2_candidate > best_c2:
            best_c2 = c2_candidate
            best_solution = candidate.copy()
        
        # Occasionally do global search to avoid local optima
        if iteration % 20 == 0 and iteration > 0:
            global_search_pattern = create_structured_pattern(1000)
            c2_global = evaluate_convolution_profile(global_search_pattern)
            if c2_global > best_c2:
                best_c2 = c2_global
                best_solution = global_search_pattern.copy()
    
    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    global start_time
    start_time = time.time()
    
    try:
        # Use convolution-guided evolution approach
        best_solution = convolution_guided_evolution()
        
        if best_solution is not None:
            # Ensure the solution is valid
            final_solution = np.clip(best_solution, 0, None).tolist()
            return final_solution
        else:
            # Fallback to structured initialization
            return create_structured_pattern(1000).tolist()
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple structured pattern
        return create_structured_pattern(1000).tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")