# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
import time
from numba import jit
import warnings

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
    This is more numerically stable and faster for large arrays.
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

def spectral_feasible_set_initialization(dim):
    """
    Initialize using spectral domain approach with feasible set constraints.
    Creates patterns that are likely to yield high C2 values through spectral analysis.
    """
    # Create a pattern based on the principle that high C2 often comes from 
    # functions with specific spectral properties - low frequency dominant signals
    # with some high-frequency components for structure
    
    # Base pattern: low frequency cosine with some high frequencies for complexity
    x = np.linspace(0, 1, dim)
    
    # Main low frequency component
    low_freq = 0.6 + 0.3 * np.cos(2 * np.pi * x * 2)  # Two periods
    
    # Add some high frequency components for structure
    high_freq = 0.1 * np.sin(2 * np.pi * x * 15)  # 15 periods
    mid_freq = 0.1 * np.sin(2 * np.pi * x * 8)    # 8 periods
    
    # Combine with a base sine component
    base_sine = 0.2 * np.sin(2 * np.pi * x * 3)   # 3 periods
    
    pattern = low_freq + high_freq + mid_freq + base_sine
    
    # Add a structured pattern to enhance performance
    # Alternate peaks and valleys
    for i in range(dim):
        if i % 4 == 0:
            pattern[i] += 0.3
        elif i % 4 == 2:
            pattern[i] -= 0.2
            
    # Ensure non-negativity and normalize
    pattern = np.maximum(pattern, 0)
    
    # Normalize to reasonable range
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 2.0
    
    return pattern.tolist()

def convex_optimization_framework(initial_params):
    """
    Main optimization using convex relaxation and spectral properties.
    Transforms the problem to optimize in spectral domain, then maps back.
    """
    def objective(coeffs):
        """Convert spectral coeffs to function values and evaluate C2"""
        # Convert coeffs back to function space
        # For simplicity, we'll work directly with the full parameter space
        # but use the spectral insights for better convergence
        try:
            # Ensure non-negative
            f_vals = np.clip(coeffs, 0, None)
            
            # Compute autoconvolution
            g_vals = compute_autoconvolution_fourier(f_vals)
            
            # Compute C2
            c2 = compute_c2_numba(g_vals)
            
            return -c2  # Negative because we minimize
        except Exception:
            return 1e10  # Penalty for invalid solution
    
    # Use trust-constr method which handles bounds well and is robust
    try:
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10},
            tol=1e-10
        )
        return result.x
    except Exception as e:
        # Fallback to L-BFGS-B if trust-constr fails
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=[(0, 10) for _ in range(len(initial_params))],
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            return result.x
        except Exception:
            # Last resort: simple gradient descent with manual line search
            return initial_params

def advanced_spectral_initialization(dim):
    """
    Advanced initialization using spectral domain analysis.
    Identifies promising regions in the spectrum that lead to high C2 values.
    """
    # Create a highly structured yet varied pattern based on spectral theory
    pattern = np.zeros(dim)
    
    # Create a pattern with multiple frequencies that tend to produce good C2 values
    x = np.linspace(0, 1, dim)
    
    # Primary low frequency (dominant)
    pattern += 0.5 + 0.3 * np.cos(2 * np.pi * x * 2)
    
    # Secondary structure with harmonics
    pattern += 0.1 * np.sin(2 * np.pi * x * 5)
    pattern += 0.05 * np.cos(2 * np.pi * x * 10)
    
    # Add some high frequency noise for complexity
    pattern += 0.05 * np.random.randn(dim)
    
    # Ensure non-negativity
    pattern = np.maximum(pattern, 0)
    
    # Normalize to reasonable scale
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 3.0
    
    return pattern.tolist()

def iterative_spectrum_refinement(initial_params, max_iterations=200):
    """
    Iteratively refine using spectral domain analysis to improve convergence.
    """
    current_params = np.array(initial_params)
    
    # Precompute the autoconvolution operator in spectral domain for reuse
    n = len(current_params)
    fft_len = 2 * n - 1
    
    # Perform first evaluation to establish baseline
    try:
        g = compute_autoconvolution_fourier(current_params)
        current_c2 = compute_c2_numba(g)
    except Exception:
        current_c2 = -1e10
    
    # Iterative improvement using gradient information from spectral domain
    for iteration in range(max_iterations):
        try:
            # Simple gradient ascent in function space (based on finite differences)
            # This is a simplified approach but effective for this problem
            step_size = 0.01
            grad = np.zeros_like(current_params)
            
            # Estimate gradient using finite differences
            eps = 1e-5
            for i in range(len(current_params)):
                # Perturb parameter i
                perturbed_plus = current_params.copy()
                perturbed_minus = current_params.copy()
                
                perturbed_plus[i] = max(0, current_params[i] + eps)
                perturbed_minus[i] = max(0, current_params[i] - eps)
                
                # Compute finite difference
                g_plus = compute_autoconvolution_fourier(perturbed_plus)
                g_minus = compute_autoconvolution_fourier(perturbed_minus)
                
                c2_plus = compute_c2_numba(g_plus)
                c2_minus = compute_c2_numba(g_minus)
                
                grad[i] = (c2_plus - c2_minus) / (2 * eps)
            
            # Update parameters
            current_params = current_params + step_size * grad
            current_params = np.maximum(current_params, 0)  # Non-negativity
            
            # Re-evaluate
            g = compute_autoconvolution_fourier(current_params)
            new_c2 = compute_c2_numba(g)
            
            # Early stopping if improvement is minimal
            if new_c2 - current_c2 < 1e-8:
                break
                
            current_c2 = new_c2
            
        except Exception:
            # If any error occurs, break out of loop
            break
    
    return current_params

def spectral_convex_pipeline():
    """
    Complete pipeline using spectral convex optimization approach.
    Combines spectral initialization, convex optimization, and iterative refinement.
    """
    start_time = time.time()
    
    # Initialize with spectral domain approach
    dim = np.random.randint(600, 1000)
    initial_params = advanced_spectral_initialization(dim)
    
    # Apply convex optimization framework
    optimized_params = convex_optimization_framework(initial_params)
    
    # Post-process with iterative refinement
    refined_params = iterative_spectrum_refinement(optimized_params, max_iterations=100)
    
    return refined_params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using spectral convex optimization."""
    start_time = time.time()
    
    try:
        # Run main optimization pipeline
        final_params = spectral_convex_pipeline()
        
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
        fallback_params = advanced_spectral_initialization(dim)
        return fallback_params

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")