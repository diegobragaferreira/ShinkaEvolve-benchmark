# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import warnings
from numba import jit
import time
import random
import logging
from typing import List, Tuple
import math

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

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

def balanced_gaussian_construction(n_points: int = 2000) -> List[float]:
    """
    Construct step function with balanced Gaussian peaks that respect minimum separation constraints.
    """
    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_points)

    # Initialize function
    f = np.zeros(n_points)

    # Place peaks with guaranteed minimum separation
    n_peaks = 12  # Fixed number for better consistency
    peak_positions = []
    peak_amplitudes = []
    
    # Generate positions with guaranteed minimum distance
    min_distance = 0.02  # Guaranteed minimum distance between peaks
    max_attempts = 50  # Maximum attempts per peak
    
    for i in range(n_peaks):
        attempts = 0
        placed = False
        while not placed and attempts < max_attempts:
            # Generate candidate position
            pos = np.random.uniform(-0.24, 0.24)  # Leave small margin
            
            # Check distance from existing peaks
            min_dist = float('inf')
            for existing_pos in peak_positions:
                min_dist = min(min_dist, abs(pos - existing_pos))
            
            # Accept if minimum distance is satisfied
            if min_dist >= min_distance:
                # Determine amplitude with exponential decay
                amp = np.random.exponential(1.0)
                
                peak_positions.append(pos)
                peak_amplitudes.append(amp)
                placed = True
            else:
                attempts += 1

    # Create Gaussian peaks with optimized widths
    for pos, amp in zip(peak_positions, peak_amplitudes):
        # Width of Gaussian - prefer moderate widths for better autoconvolution
        sigma = np.random.uniform(0.015, 0.035)
        gaussian = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
        f += gaussian

    # Ensure non-negativity
    f = np.maximum(f, 0.0)

    # Normalize to prevent extremely large values that cause numerical issues  
    max_val = np.max(f)
    if max_val > 0:
        f /= (max_val * 1.5)  # Slightly smaller scaling factor

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

    return f.tolist()

def spectral_peak_optimization(n_points: int = 2000) -> List[float]:
    """
    Optimize function in spectral domain to improve C2.
    """
    # Generate initial spectral representation
    # Start with a smooth, structured spectrum
    frequencies = np.fft.fftfreq(n_points, 0.5/n_points)
    
    # Create initial spectral profile that promotes good autoconvolution properties
    # Use band-limited approach with multiple frequency components
    magnitudes = np.zeros(n_points)
    
    # Add several peak frequencies with varying strengths
    # This favors flat autoconvolution profiles which maximize C2
    n_peaks = 10
    for i in range(n_peaks):
        # Position peaks logarithmically spaced in frequency domain
        freq_pos = 10**(np.log10(1) + i * (np.log10(n_points//2) - np.log10(1)) / (n_peaks - 1))
        freq_idx = int(freq_pos)
        if freq_idx < n_points//2:
            # Add energy at this frequency
            strength = np.random.gamma(2, 1.0)  # Gamma distribution for varied amplitudes
            magnitudes[freq_idx] = strength
            if freq_idx > 0:
                magnitudes[-freq_idx] = strength  # Conjugate symmetry
    
    # Add some low-frequency components for smoothness
    magnitudes[0] = np.random.gamma(1, 2.0)  # DC component
    
    # Add some random phase variations to avoid local minima
    phases = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert back to time domain
    fft_result = magnitudes * np.exp(1j * phases)
    
    # Use inverse FFT to get real-valued function
    f_real = np.real(np.fft.ifft(fft_result))
    
    # Ensure non-negativity and normalize
    f_real = np.maximum(f_real, 0.0)
    max_val = np.max(f_real)
    if max_val > 0:
        f_real = f_real / (max_val * 2.0)
    
    # Apply post-processing to enhance quality
    # Smooth with Gaussian kernel
    window_size = min(51, n_points // 10)
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/4)**2)
        kernel = kernel / np.sum(kernel)
        f_smooth = np.convolve(f_real, kernel, mode='same')
        f_real = f_smooth
    
    # Final clip to ensure non-negativity
    f_real = np.maximum(f_real, 0.0)
    
    return f_real.tolist()

def adaptive_gaussian_construction(n_points: int = 2000) -> List[float]:
    """
    Enhanced adaptive Gaussian construction with tighter parameter control.
    """
    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_points)

    # Initialize function
    f = np.zeros(n_points)

    # Place peaks strategically with stricter constraints
    n_peaks = 15  # Fixed number for better consistency
    peak_positions = []
    peak_amplitudes = []
    
    # Generate positions with better spacing strategy
    for i in range(n_peaks):
        # Prefer placing peaks away from edges for better smoothness
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

def mixed_strategy(n_points: int = 2000) -> List[float]:
    """
    Combine spectral and random approaches.
    """
    # Start with spectral approach
    base_func = spectral_peak_optimization(n_points)
    
    # Add some random noise for diversity
    noise_level = 0.005
    f_array = np.array(base_func)
    noise = np.random.normal(0, noise_level, len(f_array))
    f_array += noise
    f_array = np.maximum(f_array, 0.0)
    
    return f_array.tolist()

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value using optimized hybrid strategies.
    """
    best_c2 = -1
    best_f = None
    num_attempts = 30  # More attempts for better exploration
    
    start_time = time.time()
    
    # Strategy weights for weighted selection (based on performance analysis)
    strategy_weights = [0.3, 0.3, 0.25, 0.15]  # spectral, balanced_gaussian, mixed, adaptive_gaussian
    strategies = [
        spectral_peak_optimization,
        balanced_gaussian_construction,
        mixed_strategy,
        adaptive_gaussian_construction
    ]
    
    for attempt in range(num_attempts):
        if time.time() - start_time > 85:
            break
            
        try:
            # Select strategy based on weights
            strategy_func = random.choices(strategies, weights=strategy_weights)[0]
            
            # Use fixed size for consistent performance
            n_points = 2000
            
            # Generate function using selected strategy
            f_values = strategy_func(n_points)
            
            # Calculate norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)

            # Avoid division by zero with numerical stability
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
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
        # Fallback to simple uniform distribution
        return [0.5] * 1000

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
