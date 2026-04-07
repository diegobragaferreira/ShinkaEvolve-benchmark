# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
from numba import jit
import time

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast computation of autoconvolution norms using Numba JIT compilation.
    """
    n = len(f_values)
    if n < 1:
        return 0.0, 0.0, 0.0

    # Convert to numpy array for fast operations
    f = np.array(f_values, dtype=np.float64)
    
    # Create the step function on [-1/4, 1/4] with equal spacing
    dx = 0.5 / (n - 1) if n > 1 else 0.5
    
    # Precompute convolution manually for efficiency
    # Autoconvolution g[k] = sum f[i] * f[k-i] for valid indices
    g = np.zeros(2 * n - 1)
    
    # Manual convolution loop (optimized for autoconvolution)
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < len(g):
                g[k] += f[i] * f[j]
    
    # Keep only the middle part (proper autoconvolution)
    g_middle = g[n-1:2*n-1]
    
    # Create x-axis for g (interval [-0.5, 0.5])
    g_x = np.linspace(-0.5, 0.5, len(g_middle))
    
    # Compute the required norms
    # ||g||₂² (L2 norm squared)
    # Using trapezoidal integration approximation 
    g_sq = g_middle * g_middle
    area = 0.0
    for i in range(len(g_middle) - 1):
        h = g_x[i+1] - g_x[i]
        area += h * (g_sq[i] + g_sq[i+1]) / 2
    
    norm_2_sq = area

    # ||g||₁ (L1 norm) - approximate via summation
    norm_1 = np.sum(np.abs(g_middle)) * dx  # dx is the step size

    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g_middle))

    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation with improved accuracy.
    """
    return compute_autoconvolution_norms_fast(f_values)

def adaptive_gaussian_construction():
    """
    Construct step function based on adaptive Gaussian peaks with enhanced strategies.
    """
    # Use fewer points to keep it efficient but maintain resolution
    n_points = 2000  # Fixed number for consistency and speed
    
    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_points)

    # Initialize function
    f = np.zeros(n_points)

    # Place peaks strategically to optimize autoconvolution properties
    # Use log-uniform distribution for peak positions to avoid clustering
    n_peaks = 15  # Fixed number for better consistency
    peak_positions = []
    peak_amplitudes = []
    
    # Generate positions with better spacing strategy
    # Using inverse transform sampling to distribute peaks more evenly
    for i in range(n_peaks):
        # Prefer placing peaks away from edges for better smoothness
        # Use a distribution that favors central positions with some variability
        pos = np.random.beta(0.8, 0.8) * 0.4 - 0.2  # Map to [-0.2, 0.2] with preference for center
        
        # Check distance from existing peaks to avoid too close proximity
        min_dist = min([abs(pos - p) for p in peak_positions] + [1.0])
        if min_dist < 0.015:  # Tighter minimum distance threshold
            continue

        # Determine amplitude with exponential decay
        amp = np.random.exponential(1.0)  

        peak_positions.append(pos)
        peak_amplitudes.append(amp)

    # Create Gaussian peaks with better control over shapes
    for pos, amp in zip(peak_positions, peak_amplitudes):
        # Width of Gaussian - smaller values make sharper peaks, but avoid too sharp
        sigma = np.random.uniform(0.015, 0.04)
        gaussian = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
        f += gaussian

    # Ensure non-negativity
    f = np.maximum(f, 0.0)

    # Normalize to prevent extremely large values that cause numerical issues  
    max_val = np.max(f)
    if max_val > 0:
        f /= (max_val * 2.0)  # Scale down for stability

    # Apply more sophisticated smoothing with larger window
    window_size = min(101, n_points // 10)
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        # Apply Gaussian-like smoothing kernel
        kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/4)**2)
        kernel = kernel / np.sum(kernel)
        f_smooth = np.convolve(f, kernel, mode='same')
        f = f_smooth

    # Final check for any remaining negative values due to edge effects
    f = np.maximum(f, 0.0)

    # Add some noise to escape local minima if needed
    noise_level = 0.001
    f += np.random.normal(0, noise_level, len(f))

    # Ensure non-negativity again after noise addition
    f = np.maximum(f, 0.0)

    return f.tolist()

def construct_function() -> list[float]:
    """
    Construct step-function with high C2 value using adaptive Gaussian method.
    """
    # Try multiple attempts to get a good function  
    best_c2 = -1
    best_f = None
    num_attempts = 10  # Increased attempts for better exploration
    
    start_time = time.time()
    
    for attempt in range(num_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break
            
        try:
            # Generate function using adaptive Gaussian construction
            f_values = adaptive_gaussian_construction()

            # Calculate norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 0 or norm_inf <= 0:
                continue

            # Compute C2
            c2 = norm_2_sq / (norm_1 * norm_inf)

            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values.copy()

        except Exception as e:
            warnings.warn(f"Attempt {attempt} failed with error: {str(e)}")
            continue

    # Return the best function found
    if best_f is not None:
        return best_f
    else:
        # Fallback to original if nothing works
        return [np.random.random()] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
