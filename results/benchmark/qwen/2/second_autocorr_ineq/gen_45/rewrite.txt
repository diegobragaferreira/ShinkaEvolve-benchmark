# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    n_steps = len(f)
    
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n_steps

    # Compute autoconvolution using discrete convolution
    g = np.convolve(f, f, mode='full')
    # Trim g to the correct size (this accounts for the convolution)
    g = g[len(f)-1:2*len(f)-1]

    # Compute L2 norm squared using piecewise linear integration
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        # Trapezoidal-like integration for quadratic function
        # Using formula for integral of ax^2 + bx + c over [x0,x1]
        # But here we approximate with piecewise linear segments
        # So we use: (dx/3)(y0^2 + y0*y1 + y1^2)
        y0, y1 = g[i], g[i+1]
        norm_2_squared += (dx/3) * (y0**2 + y0*y1 + y1**2)

    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g))

    # Infinity norm
    norm_inf = np.max(np.abs(g))

    # Handle numerical edge cases
    if norm_1 <= 1e-15:
        norm_1 = 1e-15
    if norm_inf <= 1e-15:
        norm_inf = 1e-15

    return norm_2_squared, norm_1, norm_inf

def create_adaptive_step_function() -> list[float]:
    """
    Create an adaptive step function optimized for high C2 values.
    Uses a hybrid approach with adaptive peak positioning and amplitude control.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Parameters for function construction
    n_steps = 3000  # Increased steps for better resolution
    max_attempts = 200  # Maximum attempts to find good function
    max_time = 85  # Time limit in seconds
    
    best_c2 = -1
    best_function = None
    start_time = time.time()
    
    # Strategy 1: Adaptive peak positioning with logarithmic distribution
    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > max_time:
            break
            
        # Create positions using logarithmic distribution to avoid clustering
        # Generate logarithmically spaced positions away from center
        n_peaks = min(200, n_steps // 10)  # Dynamic number of peaks
        
        # Create logarithmic spacing for peak positions
        log_positions = np.logspace(np.log10(0.001), np.log10(0.25), n_peaks//2)
        # Mirror and combine for symmetric distribution
        positions = np.concatenate([-log_positions[::-1], log_positions])
        # Ensure we don't exceed our step count
        positions = positions[:n_steps]
        
        # Add some randomness to break systematic patterns
        noise_level = 0.02
        positions += np.random.normal(0, noise_level, len(positions))
        
        # Clip to valid range and sort
        positions = np.clip(positions, -0.25, 0.25)
        positions = np.sort(positions)
        
        # Generate amplitudes with adaptive behavior
        # Peaks near center get higher amplitude, with natural decay
        amplitudes = np.exp(-8 * positions**2)  # Stronger center concentration
        
        # Add adaptive noise with dynamic scaling
        adaptive_noise = np.random.normal(0, 0.15, len(positions))
        amplitudes += adaptive_noise
        
        # Ensure non-negative amplitudes
        amplitudes = np.maximum(amplitudes, 0)
        
        # Apply constraint-aware normalization to prevent extreme autoconvolution spikes
        if np.sum(amplitudes) > 0:
            amplitudes = amplitudes / np.sum(amplitudes) * 150  # Scaled appropriately
        
        # Apply additional constraint checking
        try:
            # Quick test to avoid obviously bad configurations
            test_norm_2_sq, test_norm_1, test_norm_inf = compute_autoconvolution_norms_fast(amplitudes.tolist())
            
            # If the ratio is already very poor, skip this configuration
            if test_norm_1 > 0 and test_norm_inf > 0:
                test_c2 = test_norm_2_sq / (test_norm_1 * test_norm_inf)
                if test_c2 < 0.1:  # Skip obviously bad candidates early
                    continue
                    
            # Compute final norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(amplitudes.tolist())
            
            # Skip if invalid
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                continue

            c2 = norm_2_sq / (norm_1 * norm_inf)
            
            # Keep the best function
            if c2 > best_c2:
                best_c2 = c2
                best_function = amplitudes.tolist()
                
        except Exception:
            # Skip invalid functions
            continue
    
    # Return the best function found or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a uniform distribution with some variation
        base_value = 1.0
        variation = 0.2
        return [base_value + np.random.uniform(-variation, variation) for _ in range(n_steps)]

def construct_function() -> list[float]:
    """
    Main function to construct optimized step function using adaptive approach.
    """
    return create_adaptive_step_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")