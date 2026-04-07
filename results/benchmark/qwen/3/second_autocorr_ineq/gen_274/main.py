# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import convolve
from scipy.optimize import minimize
import time
from numba import njit
import warnings

@njit
def compute_sparse_convolution_norms(f_values):
    """
    Compute convolution norms using optimized sparse approach.
    Uses FFT for fast convolution computation.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Use FFT-based convolution for efficiency - O(n log n) instead of O(n^2)
    # Pad to power of 2 for efficient FFT
    padded_size = 1
    while padded_size < 2 * n_steps - 1:
        padded_size *= 2
    
    # Prepare arrays for FFT
    f_padded = np.zeros(padded_size)
    f_padded[:n_steps] = f_values
    
    # FFT convolution: conv(f,f) = ifft(fft(f) * fft(f))
    f_fft = fft(f_padded)
    g_fft = f_fft * f_fft
    g_padded = ifft(g_fft).real
    
    # Extract relevant portion (autoconvolution result)
    g = g_padded[:2*n_steps-1]
    
    # Step size for numerical integration
    dx = 0.5 / n_steps

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

@njit
def calculate_c2_fast(g2_sq, g1, ginf):
    """Fast C2 calculation with numerical stability checks"""
    if g1 < 1e-15 or ginf < 1e-15:
        return 0.0
    return g2_sq / (g1 * ginf)

def compute_c2_robust(f_values):
    """Robust C2 computation with error handling"""
    try:
        g2_sq, g1, ginf = compute_sparse_convolution_norms(f_values)
        return calculate_c2_fast(g2_sq, g1, ginf)
    except Exception:
        return 0.0

def gaussian_bump(x, center, width, height):
    """Generate a Gaussian-shaped bump"""
    return height * np.exp(-0.5 * ((x - center) / width)**2)

def construct_geometric_initial_function(n_steps):
    """Construct initial function using multi-scale Gaussian patterns"""
    # Create a smooth, bell-shaped function that encourages flat convolution profiles
    x = np.linspace(-0.25, 0.25, n_steps)

    # Generate multiple overlapping Gaussian bumps with different scales and parameters
    f_values = np.zeros(n_steps)

    # Base large-scale Gaussian bump at center
    f_values += gaussian_bump(x, 0.0, 0.15, 1.0)

    # Medium-scale bumps to add structure
    f_values += 0.7 * gaussian_bump(x, -0.1, 0.08, 0.8)
    f_values += 0.6 * gaussian_bump(x, 0.1, 0.09, 0.7)

    # Fine-scale bumps to add detail and avoid overly smooth functions
    f_values += 0.3 * gaussian_bump(x, -0.05, 0.03, 0.5)
    f_values += 0.4 * gaussian_bump(x, 0.05, 0.04, 0.6)

    # Very fine-scale component for additional variation
    f_values += 0.2 * gaussian_bump(x, 0.0, 0.02, 0.3)

    # Add some sinusoidal modulation to create more complex but controlled variations
    modulation = 0.1 * np.sin(8 * np.pi * x) + 0.15
    f_values = f_values * (1 + modulation)

    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)

    # Normalize to reasonable scale
    total_area = np.sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = f_values / total_area * 2.0

    return f_values.tolist()

def multiresolution_optimization(initial_f, max_time):
    """Multi-resolution optimization approach"""
    start_time = time.time()
    
    # Start with coarser resolution for global search
    coarse_resolution = max(50, len(initial_f) // 4)
    
    # Create coarse version of initial function
    coarse_f = np.array(initial_f)[::max(1, len(initial_f) // coarse_resolution)]
    
    # Optimize coarse version first
    coarse_c2 = compute_c2_robust(coarse_f.tolist())
    
    # Refine to fine resolution
    fine_f = initial_f.copy()
    fine_c2 = compute_c2_robust(fine_f)
    
    # Local search on fine resolution
    for iter_num in range(30):
        if time.time() - start_time > max_time:
            break
            
        improved = False
        # Sample indices for efficiency
        indices = np.random.choice(len(fine_f), min(8, len(fine_f)//4), replace=False)
        
        for i in indices:
            if time.time() - start_time > max_time:
                break
                
            # Try multiple perturbation sizes
            original_value = fine_f[i]
            steps = [0.005, 0.01, 0.02, 0.05]
            
            for step in steps:
                for direction in [1, -1]:
                    if time.time() - start_time > max_time:
                        break
                    test_f = fine_f.copy()
                    new_val = original_value + direction * step
                    test_f[i] = max(0, new_val)
                    
                    new_c2 = compute_c2_robust(test_f)
                    if new_c2 > fine_c2:
                        fine_f = test_f
                        fine_c2 = new_c2
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            break
    
    return fine_f, fine_c2

def adaptive_local_search(initial_f, max_time):
    """Enhanced local search with multiple improvement strategies"""
    start_time = time.time()
    refined_f = np.array(initial_f)
    old_c2 = compute_c2_robust(refined_f.tolist())
    
    # Adaptive improvement schedule
    improvement_schedule = [
        {'steps': [0.005, 0.01, 0.02], 'iterations': 10},
        {'steps': [0.01, 0.02, 0.05], 'iterations': 15},
        {'steps': [0.02, 0.05, 0.1], 'iterations': 20}
    ]
    
    for stage_info in improvement_schedule:
        if time.time() - start_time > max_time:
            break
            
        for _ in range(stage_info['iterations']):
            if time.time() - start_time > max_time:
                break
                
            improved = False
            # Focus on important indices
            indices = np.random.choice(len(refined_f), min(12, len(refined_f)//3), replace=False)
            
            for i in indices:
                if time.time() - start_time > max_time:
                    break
                    
                # Try different step sizes
                for step in stage_info['steps']:
                    if time.time() - start_time > max_time:
                        break
                    for direction in [1, -1]:
                        if time.time() - start_time > max_time:
                            break
                        test_f = refined_f.copy()
                        new_val = test_f[i] + direction * step
                        test_f[i] = max(0, new_val)
                        
                        new_c2 = compute_c2_robust(test_f.tolist())
                        if new_c2 > old_c2:
                            refined_f = test_f
                            old_c2 = new_c2
                            improved = True
                            break
                    if improved:
                        break
                        
            if not improved:
                break
                
    return refined_f.tolist(), old_c2

def construct_function() -> list[float]:
    """Main function to construct step-function with high C2 value"""
    # Set up parameters
    n_steps = 200  # Fixed number for consistency with benchmarks
    max_time = 85  # seconds
    start_time = time.time()
    
    # Create initial function using geometric construction
    initial_f = construct_geometric_initial_function(n_steps)
    
    # Multi-resolution optimization
    optimized_f, final_c2 = multiresolution_optimization(initial_f, max_time)
    
    # Final local refinement
    refined_f, final_c2 = adaptive_local_search(optimized_f, max_time - (time.time() - start_time))
    
    # Ensure non-negativity and proper normalization
    final_solution = np.maximum(0.0, refined_f)
    
    # Normalize for numerical stability
    total = np.sum(final_solution)
    if total > 0:
        final_solution = final_solution / total * 10
    
    return final_solution.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")