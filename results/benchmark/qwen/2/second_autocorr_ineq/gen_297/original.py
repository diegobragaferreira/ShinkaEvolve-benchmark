# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.fft import fft, ifft
import random
from typing import List, Tuple
import time
import math
from deap import base, creator, tools, algorithms
import warnings
from numba import jit

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

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
    
    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_structured_gaussian_population(n_individuals: int = 30, n_points: int = 2000) -> List[List[float]]:
    """
    Create diverse initial population with structured Gaussian peaks
    """
    population = []
    
    for i in range(n_individuals):
        # Create x-axis
        x = np.linspace(-0.25, 0.25, n_points)
        
        # Initialize function
        f = np.zeros(n_points)
        
        # Generate peaks with proper spacing
        n_peaks = random.randint(8, 16)
        peak_positions = []
        peak_amplitudes = []
        peak_widths = []
        
        # Place peaks with guaranteed minimum separation
        min_distance = 0.015  # Minimum distance between peaks
        max_attempts = 100
        
        for j in range(n_peaks):
            attempts = 0
            placed = False
            while not placed and attempts < max_attempts:
                # Generate candidate position (prefer center for better autoconvolution)
                pos = np.random.beta(1.5, 1.5) * 0.4 - 0.2  # Map to [-0.2, 0.2]
                
                # Check distance from existing peaks
                min_dist = float('inf')
                for existing_pos in peak_positions:
                    min_dist = min(min_dist, abs(pos - existing_pos))
                
                # Accept if minimum distance is satisfied
                if min_dist >= min_distance:
                    amp = np.random.exponential(1.0)
                    sigma = np.random.uniform(0.01, 0.04)
                    
                    peak_positions.append(pos)
                    peak_amplitudes.append(amp)
                    peak_widths.append(sigma)
                    placed = True
                else:
                    attempts += 1
            
            if not placed:
                # If couldn't place, just add a random peak
                pos = np.random.uniform(-0.24, 0.24)
                amp = np.random.exponential(1.0)
                sigma = np.random.uniform(0.01, 0.04)
                
                peak_positions.append(pos)
                peak_amplitudes.append(amp)
                peak_widths.append(sigma)

        # Create Gaussian peaks
        for pos, amp, sigma in zip(peak_positions, peak_amplitudes, peak_widths):
            gaussian = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            f += gaussian

        # Ensure non-negativity
        f = np.maximum(f, 0.0)

        # Normalize
        max_val = np.max(f)
        if max_val > 0:
            f /= (max_val * 2.0)

        # Apply smoothing with adaptive window size
        window_size = min(51, n_points // 12)
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/4)**2)
            kernel = kernel / np.sum(kernel)
            f_smooth = np.convolve(f, kernel, mode='same')
            f = f_smooth

        # Final clip
        f = np.maximum(f, 0.0)
        
        population.append(f.tolist())
    
    return population

def selective_differential_evolution_optimization(initial_func: List[float], max_time_seconds: int = 30) -> List[float]:
    """
    Perform differential evolution optimization specifically on peak parameters
    """
    try:
        # Extract peak information (simplified approach)
        x = np.linspace(-0.25, 0.25, len(initial_func))
        f_array = np.array(initial_func)
        
        # Simple peak detection 
        peaks = []
        for i in range(1, len(f_array)-1):
            if f_array[i] > f_array[i-1] and f_array[i] > f_array[i+1]:
                peaks.append((x[i], f_array[i]))
        
        # Keep top 8 peaks for refinement
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:min(8, len(peaks))]
        
        if len(top_peaks) < 2:
            # If not enough peaks, just return original
            return initial_func
        
        # Define optimization function for peak parameters only
        def peak_objective(params):
            # Reconstruct function based on peak parameters
            temp_func = np.zeros_like(f_array)
            
            # Use the provided peak positions but adjust amplitudes
            for i, (pos, orig_amp) in enumerate(top_peaks):
                if i < len(params):
                    # Scale amplitude based on parameter
                    scaled_amp = params[i] * orig_amp
                    sigma = np.random.uniform(0.015, 0.035)  # Keep width fixed
                    gaussian = scaled_amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                    temp_func += gaussian
            
            temp_func = np.maximum(temp_func, 0.0)
            max_val = np.max(temp_func)
            if max_val > 0:
                temp_func = temp_func / (max_val * 2.0)
            
            # Compute C2
            try:
                norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(temp_func.tolist())
                if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                    return 1e10
                c2 = norm_2_sq / (norm_1 * norm_inf)
                return -c2  # Negative because we minimize
            except Exception:
                return 1e10
        
        # Optimize peak amplitudes
        bounds = [(0.1, 3.0) for _ in range(len(top_peaks))]
        result = differential_evolution(
            peak_objective,
            bounds,
            maxiter=20,
            popsize=8,
            seed=42,
            disp=False,
            timeout=max_time_seconds
        )
        
        if result.success:
            # Reconstruct final function
            final_func = np.zeros_like(f_array)
            for i, (pos, orig_amp) in enumerate(top_peaks):
                if i < len(result.x):
                    scaled_amp = result.x[i] * orig_amp
                    sigma = np.random.uniform(0.015, 0.035)
                    gaussian = scaled_amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                    final_func += gaussian
            
            final_func = np.maximum(final_func, 0.0)
            max_val = np.max(final_func)
            if max_val > 0:
                final_func = final_func / (max_val * 2.0)
            
            return final_func.tolist()
        
    except Exception:
        pass
    
    return initial_func

def spectral_peak_constructor(n_points: int = 2000) -> List[float]:
    """
    Construct function using spectral domain approach with enhanced properties
    """
    # Create frequency domain representation
    magnitudes = np.zeros(n_points)
    
    # Create multiple clusters of frequency components
    n_clusters = 6
    cluster_centers = np.logspace(np.log10(2), np.log10(n_points//2), n_clusters)
    
    # Add components to clusters with more controlled energy distribution
    for i, center in enumerate(cluster_centers):
        n_components = np.random.randint(2, 5)
        for j in range(n_components):
            freq_offset = np.random.normal(0, center * 0.1)
            freq_idx = int(center + freq_offset)
            if 1 <= freq_idx < n_points//2:
                # Use gamma distribution for more varied strengths
                strength = np.random.gamma(2.0, 1.5)
                magnitudes[freq_idx] = strength
                if freq_idx > 0:
                    magnitudes[-freq_idx] = strength  # Conjugate symmetry
                    
    # Add DC component
    magnitudes[0] = np.random.gamma(1.5, 2.0)
    
    # Add phase information
    phases = np.random.uniform(0, 2*np.pi, n_points)
    
    # Create complex spectrum
    spectrum = magnitudes * np.exp(1j * phases)
    
    # Convert back to time domain
    f_real = np.real(ifft(spectrum))
    
    # Ensure non-negativity
    f_real = np.maximum(f_real, 0.0)
    
    # Normalize
    max_val = np.max(f_real)
    if max_val > 0:
        f_real = f_real / (max_val * 1.8)
    
    # Apply structured smoothing with Gaussian kernel
    window_size = min(51, n_points // 10)
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/4)**2)
        kernel = kernel / np.sum(kernel)
        f_smooth = np.convolve(f_real, kernel, mode='same')
        f_real = f_smooth
    
    # Final clip
    f_real = np.maximum(f_real, 0.0)
    
    return f_real.tolist()

def adaptive_peak_refinement(func: List[float], n_iterations: int = 10) -> List[float]:
    """
    Adaptively refine peak structure in function
    """
    f_array = np.array(func)
    x = np.linspace(-0.25, 0.25, len(f_array))
    
    # Apply iterative refinement
    for iteration in range(n_iterations):
        # Simple gradient-based refinement
        # Just smooth slightly and re-normalize
        window_size = min(31, len(f_array) // 20)
        if window_size % 2 == 0:
            window_size += 1
        if window_size > 1:
            kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/6)**2)
            kernel = kernel / np.sum(kernel)
            f_smooth = np.convolve(f_array, kernel, mode='same')
            f_array = f_smooth
        
        # Ensure non-negativity
        f_array = np.maximum(f_array, 0.0)
        
        # Re-normalize
        max_val = np.max(f_array)
        if max_val > 0:
            f_array = f_array / (max_val * 2.0)
    
    return f_array.tolist()

def hybrid_strategy(n_points: int = 2000) -> List[float]:
    """
    Hybrid approach combining spectral and peak-based strategies
    """
    # Start with spectral construction
    base_func = spectral_peak_constructor(n_points)
    
    # Apply peak refinement
    refined_func = adaptive_peak_refinement(base_func, n_iterations=5)
    
    return refined_func

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value using improved hybrid strategies.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    best_c2 = -1
    best_f = None
    num_attempts = 40  # More attempts for better exploration
    
    # Strategy weights for weighted selection - optimized based on prior performance
    strategy_weights = [0.35, 0.30, 0.25, 0.10]  # spectral, gaussian, hybrid, refined
    strategies = [
        spectral_peak_constructor,
        create_structured_gaussian_population,
        hybrid_strategy,
        adaptive_peak_refinement
    ]
    
    start_time = time.time()
    
    for attempt in range(num_attempts):
        if time.time() - start_time > 85:
            break
            
        try:
            # Select strategy based on weights
            strategy = random.choices(strategies, weights=strategy_weights)[0]
            
            # Use fixed size for consistent performance
            n_points = 2000
            
            # Generate function using selected strategy
            if strategy == create_structured_gaussian_population:
                # Special handling for population-based strategy
                population = strategy(n_individuals=1, n_points=n_points)
                f_values = population[0]
            else:
                f_values = strategy(n_points)
            
            # Compute C2 directly
            c2 = compute_c2(f_values)
            
            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values.copy()

        except Exception as e:
            warnings.warn(f"Attempt {attempt} failed with error: {str(e)}")
            continue
    
    # Final refinement if we found anything
    if best_f is not None and best_c2 > 0.9:
        try:
            # Apply selective optimization
            refined = selective_differential_evolution_optimization(best_f, max_time_seconds=10)
            refined_c2 = compute_c2(refined)
            if refined_c2 > best_c2:
                best_f = refined
        except Exception:
            pass
    
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
