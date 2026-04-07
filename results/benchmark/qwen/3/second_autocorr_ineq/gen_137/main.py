# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from typing import List
import time
import math

def compute_autoconvolution_norms(f_values: List[float]):
    """
    Compute the three norms needed for C₂ calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4]
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    dx = 0.5 / n_steps  # Step size

    # Create piecewise constant function from step heights
    f = np.array(f_values)
    
    # Use FFT-based convolution for efficiency
    # Pad to appropriate size for full convolution
    pad_size = 2 * n_steps - 1
    f_padded = np.pad(f, (0, pad_size - n_steps), mode='constant')
    
    # FFT-based convolution
    f_fft = fft(f_padded)
    g_fft = f_fft * f_fft.conj()  # Autoconvolution in frequency domain
    g = np.real(ifft(g_fft))[:pad_size]
    
    # Extract the central region corresponding to [-1/4, 1/4]
    central_start = (pad_size - n_steps) // 2
    central_end = central_start + n_steps
    g_centered = g[central_start:central_end]

    # Compute norms
    g_abs = np.abs(g_centered)
    
    # L2 norm squared
    norm_2_sq = np.sum(g_abs * g_abs) * dx
    
    # L1 norm
    norm_1 = np.sum(g_abs) * dx
    
    # L-infinity norm
    norm_inf = np.max(g_abs)
    
    return norm_2_sq, norm_1, norm_inf

def calculate_c2(f_values: List[float]) -> float:
    """
    Calculate the C2 constant from the step function values.
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except:
        return 0.0

def generate_geometric_pattern(n: int, base_rate: float = 0.8, decay: float = 0.9) -> List[float]:
    """Generate a geometrically decaying pattern that enhances convolution properties"""
    pattern = []
    for i in range(n):
        # Create a pattern that starts high and decays
        pos = i / (n - 1) if n > 1 else 0.5
        # Geometric decay with some oscillation
        amplitude = base_rate * (decay ** i)
        oscillation = 0.1 * math.sin(8 * math.pi * pos)
        value = max(0.0, amplitude + oscillation + 0.05)
        pattern.append(value)
    return pattern

def generate_sine_pattern(n: int, frequency: float = 4.0) -> List[float]:
    """Generate a sine-based modulation pattern"""
    pattern = []
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        # Sine wave with varying amplitude
        amplitude = 0.5 + 0.3 * math.sin(frequency * math.pi * pos)
        # Add some high-frequency noise for complexity
        noise = 0.05 * math.sin(20 * math.pi * pos) * math.cos(15 * math.pi * pos)
        value = max(0.0, amplitude + noise)
        pattern.append(value)
    return pattern

def generate_peak_pattern(n: int, num_peaks: int = 8) -> List[float]:
    """Generate a pattern with strategically placed peaks"""
    pattern = [0.0] * n
    
    # Place peaks at regular intervals
    peak_positions = []
    for i in range(num_peaks):
        pos = (i + 0.5) / num_peaks
        peak_positions.append(int(pos * n))
    
    # Create peak shapes using Gaussian-like functions
    for pos in peak_positions:
        if pos < n:
            # Create a bell-shaped peak
            for i in range(max(0, pos-10), min(n, pos+11)):
                distance = abs(i - pos)
                # Gaussian-like decay
                value = 1.0 * math.exp(-0.5 * (distance**2) / 25.0)
                pattern[i] = max(pattern[i], value)
    
    # Add some randomness to make it more interesting
    for i in range(n):
        if i % 3 == 0:  # Every third position
            pattern[i] = max(0.0, pattern[i] + 0.1 * (np.random.random() - 0.5))
    
    # Normalize and ensure minimum values
    max_val = max(pattern) if pattern else 1.0
    if max_val > 0:
        pattern = [val / max_val * 0.8 for val in pattern]
    
    # Add some base level
    for i in range(len(pattern)):
        pattern[i] = max(0.0, pattern[i] + 0.1)
    
    return pattern

def refine_pattern(pattern: List[float], iterations: int = 3) -> List[float]:
    """Refine pattern through iterative improvement heuristics"""
    refined = pattern.copy()
    
    for iter_num in range(iterations):
        # Try to improve by adjusting adjacent elements
        improved = refined.copy()
        
        for i in range(len(improved)):
            # Try small adjustments
            current_value = improved[i]
            best_value = current_value
            best_c2 = calculate_c2(improved)
            
            # Try small positive adjustment
            test_value = max(0.0, current_value + 0.02)
            if test_value != current_value:
                test_pattern = improved.copy()
                test_pattern[i] = test_value
                test_c2 = calculate_c2(test_pattern)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_value = test_value
            
            # Try small negative adjustment  
            test_value = max(0.0, current_value - 0.02)
            if test_value != current_value:
                test_pattern = improved.copy()
                test_pattern[i] = test_value
                test_c2 = calculate_c2(test_pattern)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_value = test_value
            
            improved[i] = best_value
        
        refined = improved
    
    return refined

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C₂ value using pattern-based optimization.
    Uses deterministic construction with mathematical patterns and refinement.
    """
    start_time = time.time()
    best_c2 = 0.0
    best_pattern = []
    
    # Different pattern types to try
    pattern_types = [
        ("geometric", lambda n: generate_geometric_pattern(n, 0.9, 0.85)),
        ("sine", lambda n: generate_sine_pattern(n, 6.0)),
        ("peaks", lambda n: generate_peak_pattern(n, 12)),
    ]
    
    # Try different sizes to find optimal resolution
    sizes = [200, 300, 400, 500, 600, 700, 800]
    
    for size in sizes:
        if time.time() - start_time > 85:  # Leave 5 seconds for final processing
            break
            
        for pattern_name, pattern_func in pattern_types:
            if time.time() - start_time > 85:
                break
                
            # Generate pattern
            pattern = pattern_func(size)
            
            # Refine the pattern
            refined_pattern = refine_pattern(pattern, 5)
            
            # Evaluate
            current_c2 = calculate_c2(refined_pattern)
            
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_pattern = refined_pattern.copy()
    
    # Additional local search refinement on the best pattern found
    if best_pattern and time.time() - start_time < 85:
        # Try a more aggressive refinement step
        final_pattern = refine_pattern(best_pattern, 10)
        final_c2 = calculate_c2(final_pattern)
        
        if final_c2 > best_c2:
            best_pattern = final_pattern
            best_c2 = final_c2
    
    # Final validation check
    if not best_pattern:
        # Fallback to a simple geometric pattern
        best_pattern = generate_geometric_pattern(500, 0.8, 0.9)
    
    # Ensure pattern length is reasonable and has good properties
    if len(best_pattern) < 100:
        best_pattern = generate_geometric_pattern(500, 0.8, 0.9)
    
    return best_pattern

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")