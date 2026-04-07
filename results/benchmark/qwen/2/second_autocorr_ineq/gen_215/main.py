# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple
import time
from numba import jit
import warnings

# Numba optimized computation for faster execution
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast computation of autoconvolution norms using Numba JIT compilation.
    """
    if len(f_values) < 1:
        return 0.0, 0.0, 0.0

    # Convert to numpy array for fast operations
    f = np.array(f_values, dtype=np.float64)
    n = len(f)
    
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
    norm_1 = np.sum(np.abs(g_middle)) * (0.5 / (n - 1))  # dx is the step size

    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g_middle))

    return norm_2_sq, norm_1, norm_inf

def compute_c2_fast(f_values: List[float]) -> float:
    """Fast C2 computation using optimized norms"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
    
    # Avoid division by zero with numerical stability
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_structured_gaussian_function(n_steps: int) -> List[float]:
    """
    Create structured step function with logarithmically spaced Gaussian peaks
    with controlled spacing and amplitude scaling.
    """
    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Initialize function
    f_values = np.zeros(n_steps)
    
    # Determine number of peaks based on function length
    n_peaks = max(3, min(20, n_steps // 50))
    
    # Generate peak positions with logarithmic spacing and minimum separation
    peak_positions = []
    peak_widths = []
    peak_heights = []
    
    # Use logarithmic spacing to distribute peaks across multiple scales
    if n_peaks > 1:
        # Create log-uniform positions from left to right
        log_min = np.log(0.01)  # Minimum relative distance
        log_max = np.log(0.48)  # Maximum relative distance (leave margin for edges)
        log_positions = np.logspace(log_min, log_max, n_peaks, base=np.e)
        
        # Map to actual positions in [-0.25, 0.25] with edge constraints
        total_range = 0.5
        offset = 0.02  # Minimum distance from edges
        
        # Distribute peaks logarithmically with appropriate boundary handling
        for i in range(n_peaks):
            if n_peaks == 1:
                pos = 0.0  # Center for single peak
            else:
                # Map log-spaced to actual domain with edge offsets
                rel_pos = log_positions[i] if i < len(log_positions) else 0.5
                pos = -0.25 + offset + rel_pos * (total_range - 2*offset)
            
            # Ensure bounds are respected
            pos = np.clip(pos, -0.25 + offset, 0.25 - offset)
            peak_positions.append(float(pos))
    else:
        # Single peak centered
        peak_positions.append(0.0)
    
    # Add small random perturbations to positions to avoid perfect symmetry
    for i in range(len(peak_positions)):
        if n_peaks > 1:
            peak_positions[i] += np.random.uniform(-0.01, 0.01)
        # Ensure within bounds
        peak_positions[i] = np.clip(peak_positions[i], -0.25 + 0.02, 0.25 - 0.02)
    
    # Generate peak parameters with mathematical optimization
    for i in range(n_peaks):
        # Width inversely related to height for better control and flat autoconvolution
        width = np.random.uniform(0.008, 0.025)
        peak_widths.append(width)
        # Height inversely proportional to width to maintain balance
        height = np.random.uniform(0.8, 2.0)
        peak_heights.append(height)
    
    # Create Gaussian curves for each peak with mathematical optimization
    for center, width, height in zip(peak_positions, peak_widths, peak_heights):
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_values += gaussian
    
    # Apply mathematically principled smoothing
    if n_steps > 50:
        # Use a simple Gaussian kernel for smoothing
        kernel_size = min(21, n_steps // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            kernel = np.exp(-0.5 * np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)**2 / (kernel_size/4)**2)
            kernel = kernel / np.sum(kernel)
            f_values = np.convolve(f_values, kernel, mode='same')
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)
    
    # Normalize to reasonable range but preserve peak structure
    if np.max(f_values) > 0:
        # Scale to approximately unit max but allow some headroom for better autoconvolution
        f_values = f_values / np.max(f_values) * 1.2
    
    return f_values.tolist()

def selective_optimization(initial_function: List[float], n_steps: int) -> List[float]:
    """
    Apply selective differential evolution on peak parameters for fine-tuning
    """
    try:
        # Work with np array for faster operations
        current_func = np.array(initial_function)
        
        # For better performance, work with a subset of parameters
        sample_size = min(100, n_steps // 2)
        if sample_size > 0:
            # Sample indices for optimization (avoiding edge points for stability)
            sample_indices = sorted(random.sample(
                range(max(2, sample_size), min(n_steps - 2, n_steps)), 
                sample_size
            ))
            
            # Objective function that works with sampled parameters
            def objective(params):
                # Reconstruct function with updated parameters
                temp_func = current_func.copy()
                for i, idx in enumerate(sample_indices):
                    if i < len(params):
                        temp_func[idx] = max(0, params[i])
                
                try:
                    c2_val = compute_c2_fast(temp_func.tolist())
                    return -c2_val  # Negative because we want to maximize
                except Exception:
                    return 1e10
            
            # Boundaries for the parameters
            bounds = [(0.0, 3.0) for _ in range(sample_size)]
            
            # Perform differential evolution with reduced complexity
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=5,
                seed=42,
                disp=False
            )
            
            if result.success and len(result.x) >= sample_size:
                # Update the function with optimized values
                final_func = current_func.copy()
                for i, idx in enumerate(sample_indices):
                    if i < len(result.x):
                        final_func[idx] = max(0, result.x[i])
                
                return final_func.tolist()
                
    except Exception:
        pass
    
    # Return original if optimization fails
    return initial_function

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses a hybrid approach combining structured initialization with selective optimization.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    # Determine number of steps with time consideration
    # Allocate roughly 80% of time for primary computation
    remaining_time = 80  # seconds
    n_steps = min(10000, max(100, 1000 + int(np.random.randint(0, 500))))
    
    # Phase 1: Create structured Gaussian-based initialization
    structured_function = create_structured_gaussian_function(n_steps)
    
    # Phase 2: Apply selective optimization to fine-tune parameters
    if time.time() - start_time > remaining_time - 5:
        return structured_function
    
    optimized_function = selective_optimization(structured_function, n_steps)
    
    # Phase 3: Final validation and adjustment
    try:
        # Compute C2 for final function
        final_c2 = compute_c2_fast(optimized_function)
        
        # If we got a very poor result, fall back to a better constructed function
        if final_c2 < 0.1:
            # Recreate with different parameters
            alternative_function = create_structured_gaussian_function(n_steps)
            alt_c2 = compute_c2_fast(alternative_function)
            if alt_c2 > final_c2:
                optimized_function = alternative_function
                
    except Exception:
        # Fallback to simple approach if something goes wrong
        optimized_function = [0.5] * n_steps
    
    return optimized_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")