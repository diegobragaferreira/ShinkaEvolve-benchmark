# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
from scipy.spatial.distance import euclidean
import time
from numba import jit
import warnings
from itertools import combinations
import random

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

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
    """Compute C2 value using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms using trapezoidal integration for L2^2
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute L2^2 norm using trapezoidal integration
    if len(g_vals) >= 2:
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        # Correct step width for domain [-1/2, 1/2] with len(g_vals) points
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2 with improved numerical stability
    epsilon = 1e-16
    if g_l1 > epsilon and g_max > epsilon:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_autoconvolution_fourier(f_vals):
    """
    Compute autoconvolution using FFT-based method for efficiency.
    More numerically stable and faster for large arrays.
    """
    n = len(f_vals)
    
    # Use FFT to compute convolution efficiently (f*f = FFT^-1(FFT(f)^2))
    # Zero-pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    f_padded = np.pad(f_vals, (0, padded_length - n), mode='constant', constant_values=0)
    
    # FFT-based convolution
    F = fft(f_padded)
    G_fft = F * F  # Point-wise multiplication in frequency domain
    g = ifft(G_fft).real[:padded_length]
    
    return g

def spectral_pattern_generator(dim):
    """
    Generate patterns using spectral domain insights.
    Creates functions that are likely to yield high C₂ values.
    """
    # Generate multiple candidate patterns using different spectral characteristics
    patterns = []
    
    # Pattern 1: Low frequency dominance with harmonics
    x = np.linspace(0, 1, dim)
    pattern1 = 0.6 + 0.3 * np.cos(2 * np.pi * x * 2) + 0.1 * np.sin(2 * np.pi * x * 7)
    pattern1 = np.maximum(pattern1, 0)
    patterns.append(pattern1)
    
    # Pattern 2: Multi-scale structure
    pattern2 = np.zeros(dim)
    pattern2 += 0.5 * np.cos(2 * np.pi * x * 3)  # Medium frequency
    pattern2 += 0.3 * np.sin(2 * np.pi * x * 12) # High frequency
    pattern2 += 0.2 * np.sin(2 * np.pi * x * 5)  # Mid frequency
    pattern2 = np.maximum(pattern2, 0)
    patterns.append(pattern2)
    
    # Pattern 3: Asymmetric structure for better convolution properties
    pattern3 = np.zeros(dim)
    for i in range(dim):
        pos = i / dim
        pattern3[i] = 0.4 + 0.3 * np.sin(pos * np.pi * 8) + 0.1 * np.cos(pos * np.pi * 16)
    pattern3 = np.maximum(pattern3, 0)
    patterns.append(pattern3)
    
    # Pattern 4: Sparse structure with selective peaks
    pattern4 = np.zeros(dim)
    for i in range(dim):
        if i % 8 == 0:
            pattern4[i] = 1.0
        elif i % 8 == 4:
            pattern4[i] = 0.3
        else:
            pattern4[i] = 0.1
    patterns.append(pattern4)
    
    # Select the best pattern based on initial C2 evaluation
    best_pattern = patterns[0]
    best_c2 = -1.0
    
    for pattern in patterns:
        try:
            f_vals = np.clip(pattern, 0, None)
            g_vals = compute_autoconvolution_fourier(f_vals)
            c2 = compute_c2_numba(g_vals)
            if c2 > best_c2:
                best_c2 = c2
                best_pattern = pattern.copy()
        except Exception:
            continue
    
    return best_pattern.tolist()

def adaptive_stochastic_search(initial_params, max_evaluations=500):
    """
    Implement adaptive stochastic search that varies exploration vs exploitation
    based on convergence progress.
    """
    current_params = np.array(initial_params)
    best_params = current_params.copy()
    best_c2 = -np.inf
    
    # Initialize with some random perturbations to escape local traps
    for i in range(len(current_params)):
        if np.random.random() < 0.3:  # 30% chance to perturb
            perturbation = np.random.normal(0, 0.1)
            current_params[i] = max(0, current_params[i] + perturbation)
    
    # Evaluate initial point
    try:
        g = compute_autoconvolution_fourier(current_params)
        c2 = compute_c2_numba(g)
        if c2 > best_c2:
            best_c2 = c2
            best_params = current_params.copy()
    except:
        c2 = -1e10
    
    evaluations = 1
    stagnation_counter = 0  # Track if we're not improving
    
    # Adaptive search loop
    while evaluations < max_evaluations:
        # Adaptively change strategy based on progress
        if stagnation_counter > 10:
            # Increase exploration
            step_size = 0.05
        else:
            # Focus on exploitation
            step_size = 0.01
            
        # Generate candidate by adding noise
        candidate = current_params + np.random.normal(0, step_size, len(current_params))
        candidate = np.maximum(candidate, 0)  # Ensure non-negativity
        
        try:
            g = compute_autoconvolution_fourier(candidate)
            candidate_c2 = compute_c2_numba(g)
            evaluations += 1
            
            if candidate_c2 > best_c2:
                best_c2 = candidate_c2
                best_params = candidate.copy()
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1
                
            # Move towards better solution
            if candidate_c2 > c2:
                current_params = candidate.copy()
                c2 = candidate_c2
            else:
                # Occasionally accept worse solutions to escape local minima
                if np.random.random() < 0.1:
                    current_params = candidate.copy()
                    c2 = candidate_c2
                    
        except:
            stagnation_counter += 1
            continue
    
    return best_params

def geometric_convex_refinement(initial_params, max_iterations=100):
    """
    Refinement using geometric principles and convexity exploitation.
    Rather than purely gradient-based, uses simplex-like local search.
    """
    current_params = np.array(initial_params)
    n = len(current_params)
    
    # Create initial simplex (n+1 vertices)
    simplex = [current_params.copy()]
    for i in range(n):
        vertex = current_params.copy()
        # Perturb one dimension slightly
        vertex[i] = max(0, vertex[i] + 0.01 * np.random.randn())
        simplex.append(vertex)
    
    # Get initial function values
    values = []
    for vertex in simplex:
        try:
            g = compute_autoconvolution_fourier(vertex)
            c2 = compute_c2_numba(g)
            values.append(c2)
        except:
            values.append(-1e10)
    
    # Simplex optimization loop
    for iteration in range(max_iterations):
        # Sort simplex by function values
        sorted_indices = np.argsort(values)[::-1]  # Descending order (best first)
        simplex = [simplex[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Stop if improvement is negligible
        if values[0] - values[-1] < 1e-8:
            break
            
        # Compute centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect worst point
        reflected = centroid + centroid - simplex[-1]
        reflected = np.maximum(reflected, 0)  # Non-negativity constraint
        
        try:
            g = compute_autoconvolution_fourier(reflected)
            reflected_c2 = compute_c2_numba(g)
        except:
            reflected_c2 = -1e10
            
        # Accept reflection if it's better
        if reflected_c2 > values[0]:
            # Expand if even better
            expanded = centroid + 2 * (reflected - centroid)
            expanded = np.maximum(expanded, 0)
            
            try:
                g = compute_autoconvolution_fourier(expanded)
                expanded_c2 = compute_c2_numba(g)
            except:
                expanded_c2 = -1e10
                
            if expanded_c2 > reflected_c2:
                simplex[-1] = expanded
                values[-1] = expanded_c2
            else:
                simplex[-1] = reflected
                values[-1] = reflected_c2
        else:
            # Contract
            contraction = centroid + 0.5 * (simplex[-1] - centroid)
            contraction = np.maximum(contraction, 0)
            
            try:
                g = compute_autoconvolution_fourier(contraction)
                contraction_c2 = compute_c2_numba(g)
            except:
                contraction_c2 = -1e10
                
            if contraction_c2 > values[-1]:
                simplex[-1] = contraction
                values[-1] = contraction_c2
            else:
                # Shrink simplex
                best_point = simplex[0]
                for i in range(1, len(simplex)):
                    simplex[i] = best_point + 0.5 * (simplex[i] - best_point)
                    try:
                        g = compute_autoconvolution_fourier(simplex[i])
                        values[i] = compute_c2_numba(g)
                    except:
                        values[i] = -1e10
    
    # Return best solution found
    best_index = np.argmax(values)
    return simplex[best_index]

def multi_stage_spectral_pipeline():
    """
    Complete multi-stage optimization pipeline.
    Combines spectral initialization, stochastic search, and geometric refinement.
    """
    start_time = time.time()
    
    # Stage 1: Spectral initialization to get promising starting point
    dim = np.random.randint(600, 1000)
    initial_params = spectral_pattern_generator(dim)
    
    # Stage 2: Adaptive stochastic search for global exploration
    stochastic_result = adaptive_stochastic_search(initial_params, max_evaluations=300)
    
    # Stage 3: Geometric convex refinement for local exploitation
    final_result = geometric_convex_refinement(stochastic_result, max_iterations=100)
    
    return final_result

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid spectral optimization."""
    start_time = time.time()
    
    try:
        # Run multi-stage optimization pipeline
        final_params = multi_stage_spectral_pipeline()
        
        # Final validation
        g = compute_autoconvolution_fourier(final_params)
        c2_val = compute_c2_numba(g)
        
        print(f"Best C2 found: {c2_val}")
        
        # Return as list of floats
        return [float(x) for x in final_params]
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to structured initialization
        dim = 1000
        fallback_params = spectral_pattern_generator(dim)
        return fallback_params

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")