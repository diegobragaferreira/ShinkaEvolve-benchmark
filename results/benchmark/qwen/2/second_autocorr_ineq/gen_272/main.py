# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from numba import jit
import time
import warnings
from typing import List, Tuple

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

def create_adaptive_gaussian_function(n_steps: int, num_peaks: int = 50) -> list[float]:
    """
    Create an adaptive Gaussian-based step function with improved spacing and amplitude control.
    """
    # Domain setup
    domain_width = 0.5
    x_domain = np.linspace(-0.25, 0.25, n_steps)
    
    # Create peaks with adaptive spacing
    # Use logarithmic distribution to concentrate peaks near center
    peak_positions = []
    
    # Create multiple scales of peaks to avoid regularity
    scales = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    for scale in scales:
        # Number of peaks per scale inversely proportional to scale
        n_per_scale = max(1, int(10 * (0.2 / scale)))
        
        # Generate positions in this scale
        positions_in_scale = []
        for i in range(n_per_scale):
            # Use sine distribution for better center concentration
            ratio = (i + 1) / (n_per_scale + 1)
            pos = np.sin(ratio * np.pi) * scale * 0.25
            positions_in_scale.append(pos)
            
        # Add to main list
        peak_positions.extend(positions_in_scale)
    
    # Remove duplicate positions and sort
    peak_positions = sorted(list(set(peak_positions)))
    
    # Limit to desired number of peaks
    if len(peak_positions) > num_peaks:
        # Keep peaks closer to center for better autoconvolution
        distances_from_center = [abs(pos) for pos in peak_positions]
        sorted_indices = np.argsort(distances_from_center)
        peak_positions = [peak_positions[i] for i in sorted_indices[:num_peaks]]
    
    # Ensure we have enough peaks
    if len(peak_positions) < num_peaks:
        # Fill with additional peaks at strategic positions
        remaining = num_peaks - len(peak_positions)
        for i in range(remaining):
            # Place randomly in outer regions to maintain diversity
            pos = np.random.uniform(-0.25, 0.25)
            peak_positions.append(pos)
        peak_positions = sorted(peak_positions)
    
    # Create peak parameters with adaptive amplitudes
    peak_params = []
    
    for i, pos in enumerate(peak_positions):
        # Amplitude based on position - higher near center
        base_amp = 1.0 + 0.5 * np.exp(-10 * pos**2)
        # Add some randomness but keep amplitudes positive
        amplitude = max(0.1, np.random.normal(base_amp, 0.2))
        # Width inversely proportional to amplitude to maintain balance
        width = max(0.01, min(0.1, 0.05 * (1.0 / (amplitude + 0.1))))
        peak_params.extend([amplitude, pos, width])
    
    # Create function from Gaussian peaks
    f = np.zeros_like(x_domain)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        if width > 1e-6:
            f += amp * np.exp(-0.5 * ((x_domain - center) / width)**2)
    
    # Add sinusoidal modulation to break degeneracy
    modulation = 0.1 * np.sin(20 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)
    f += modulation
    
    # Add small amount of noise for robustness
    noise = np.random.normal(0, 0.01, n_steps)
    f += noise
    
    # Ensure non-negativity
    f = np.maximum(f, 0)
    
    # Normalize to reasonable scale
    if np.sum(f) > 0:
        f = f / np.sum(f) * 10
    
    return f.tolist()

def construct_function() -> list[float]:
    """
    Construct step function using improved adaptive Gaussian optimization approach.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    start_time = time.time()
    
    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    
    # Target: 5000 steps for better resolution and smoother function
    n_steps = 5000
    
    # Try several configurations to find the best one
    attempts = 0
    max_attempts = 25  # Reduced due to time constraints
    
    while attempts < max_attempts and (time.time() - start_time) < 85:
        try:
            # Create adaptive Gaussian function
            f_values = create_adaptive_gaussian_function(n_steps)
            
            # Evaluate the function
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)

            # Check for valid norms
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                attempts += 1
                continue

            c2 = norm_2_sq / (norm_1 * norm_inf)

            # Keep the best function
            if c2 > best_c2:
                best_c2 = c2
                best_function = f_values
                
        except Exception as e:
            # Skip invalid functions and continue
            pass
        
        attempts += 1
    
    # Return the best function found, or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a simpler construction with fewer steps
        n_steps = 1000
        f_values = [1.0] * n_steps
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")