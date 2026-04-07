# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
import math

# Core computation module with enhanced numerical stability
@jit(nopython=True)
def compute_convolution_norms_jit(f_values, dx):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    JIT compiled version with enhanced numerical stability.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Autoconvolution using fast convolution approach
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Efficient convolution implementation
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j]

    # Compute norms with enhanced numerical stability
    # For ||g||₂² using trapezoidal-like integration: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = 0.0
    for i in range(len(g)):
        g1 += abs(g[i])
    g1 *= dx

    # ||g||∞ = max(|g_i|)
    ginf = 0.0
    for i in range(len(g)):
        val = abs(g[i])
        if val > ginf:
            ginf = val

    return g2_sq, g1, ginf

def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using direct computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution using direct convolution sum
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms(f_values)

    if g1 <= 1e-15 or ginf <= 1e-15:
        return 0.0

    return g2_sq / (g1 * ginf)

# Harmonic approximation functions for better initialization
def generate_harmonic_basis(n_steps, num_harmonics=5):
    """Generate step function using harmonic basis functions"""
    x = np.linspace(-0.25, 0.25, n_steps)
    f_values = np.zeros(n_steps)
    
    # Generate harmonics with decreasing amplitudes
    for i in range(1, num_harmonics + 1):
        # Sine and cosine components
        amplitude = 1.0 / (i * i)  # Amplitude decreases with frequency
        f_values += amplitude * np.sin(2 * np.pi * i * x)
        f_values += amplitude * np.cos(2 * np.pi * i * x)
    
    # Ensure all values are non-negative
    f_values = np.maximum(f_values, 0)
    
    # Normalize to reasonable scale
    total_area = np.sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = f_values / total_area * 2.0
    
    return f_values.tolist()

def generate_multi_scale_initial_function(n_steps, scales=[1, 2, 4]):
    """Generate initial function with multiple scales for better exploration"""
    f_values = np.zeros(n_steps)
    
    # Use different scaling factors to create multiresolution structure
    for scale in scales:
        if scale <= n_steps:
            # Create a pattern at this scale
            base_pattern = np.zeros(n_steps)
            segment_length = n_steps // scale
            if segment_length > 0:
                # Create a bump pattern at this scale
                for i in range(scale):
                    start_idx = i * segment_length
                    end_idx = min((i + 1) * segment_length, n_steps)
                    if end_idx > start_idx:
                        # Create a smooth transition
                        mid_point = (start_idx + end_idx) // 2
                        if mid_point < n_steps:
                            for j in range(start_idx, end_idx):
                                if j < mid_point:
                                    base_pattern[j] = (j - start_idx) / (mid_point - start_idx)
                                else:
                                    base_pattern[j] = (end_idx - j) / (end_idx - mid_point)
            
            # Add to overall function (with some randomness)
            f_values += base_pattern * (1 + np.random.random() * 0.2)
    
    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    
    # Normalize
    total_area = np.sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = f_values / total_area * 2.0
    
    return f_values.tolist()

# Enhanced optimization approach
def adaptive_coordinate_refinement(f_values, max_iterations=20, time_limit=85):
    """Enhanced coordinate-wise refinement with adaptive step sizes"""
    start_time = time.time()
    current_f = np.array(f_values)
    current_c2 = compute_c2(f_values)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations and (time.time() - start_time) < time_limit:
        improved = False
        iteration += 1
        
        # Sample indices adaptively - focus on areas with higher variance
        # This is a simplified approach to adaptive sampling
        indices_to_try = list(range(len(current_f)))
        np.random.shuffle(indices_to_try)
        
        # Try a subset of indices for efficiency
        indices_sample = indices_to_try[:min(10, len(indices_to_try))]
        
        for i in indices_sample:
            if (time.time() - start_time) >= time_limit:
                break
                
            original_value = current_f[i]
            
            # Adaptive step sizing based on current value
            base_step = max(0.01, 0.05 * original_value)
            
            # Try different step sizes
            step_sizes = [base_step, base_step * 2, base_step * 5]
            
            for step in step_sizes:
                # Try increasing and decreasing
                for direction in [1, -1]:
                    if (time.time() - start_time) >= time_limit:
                        break
                    test_f = current_f.copy()
                    new_val = original_value + direction * step
                    test_f[i] = max(0, new_val)
                    
                    new_c2 = compute_c2(test_f.tolist())
                    if new_c2 > current_c2:
                        current_f = test_f
                        current_c2 = new_c2
                        improved = True
                        
    return current_f.tolist(), current_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using harmonic-based approach."""
    # Set parameters
    n_steps = 200
    max_time = 85  # seconds
    start_time = time.time()
    
    # Strategy 1: Try harmonic basis initialization
    if (time.time() - start_time) < max_time * 0.3:
        harmonic_init = generate_harmonic_basis(n_steps, num_harmonics=6)
        try:
            harmonic_c2 = compute_c2(harmonic_init)
        except:
            harmonic_c2 = 0.0
    else:
        harmonic_c2 = 0.0
    
    # Strategy 2: Try multi-scale initialization
    if (time.time() - start_time) < max_time * 0.6:
        multi_scale_init = generate_multi_scale_initial_function(n_steps, scales=[1, 2, 3, 4])
        try:
            multi_scale_c2 = compute_c2(multi_scale_init)
        except:
            multi_scale_c2 = 0.0
    else:
        multi_scale_c2 = 0.0
    
    # Strategy 3: Standard initialization
    standard_init = []
    for _ in range(n_steps):
        standard_init.append(abs(np.random.gauss(0.5, 0.2)))
    
    try:
        standard_c2 = compute_c2(standard_init)
    except:
        standard_c2 = 0.0
    
    # Choose the best initialization
    candidates = [
        ("harmonic", harmonic_init, harmonic_c2),
        ("multi_scale", multi_scale_init, multi_scale_c2),
        ("standard", standard_init, standard_c2)
    ]
    
    # Sort by C2 score and take the best
    best_strategy, best_f, best_c2 = max(candidates, key=lambda x: x[2])
    
    # If even the standard initialization isn't improving, create a better pattern
    if best_c2 < 0.1:
        # Use a more structured approach for very poor starting points
        pattern_f = []
        half = n_steps // 2
        for i in range(n_steps):
            if i < half:
                pattern_f.append(1.0 - (i / half))
            else:
                pattern_f.append((i - half) / half)
        pattern_f = [max(0, x) for x in pattern_f]
        try:
            best_f = pattern_f
            best_c2 = compute_c2(pattern_f)
        except:
            pass
    
    # Apply enhanced coordinate-wise refinement
    refined_f, refined_c2 = adaptive_coordinate_refinement(best_f, max_iterations=15, time_limit=max_time)
    
    # Final check if refinement improved things
    if refined_c2 > best_c2:
        best_f = refined_f
        best_c2 = refined_c2
    
    # Additional post-refinement step if time allows
    if (time.time() - start_time) < max_time * 0.9:
        # A few more targeted improvements
        final_f = best_f.copy()
        for _ in range(5):
            if (time.time() - start_time) >= max_time:
                break
            # Targeted random perturbations focused on high-value areas
            test_f = final_f.copy()
            # Select a few indices to modify
            indices = np.random.choice(len(test_f), min(5, len(test_f)//10), replace=False)
            for idx in indices:
                change = np.random.normal(0, 0.05 * test_f[idx])
                test_f[idx] = max(0, test_f[idx] + change)
            
            new_c2 = compute_c2(test_f)
            if new_c2 > best_c2:
                final_f = test_f
                best_c2 = new_c2
                
        best_f = final_f
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")