# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.fftpack import fft, ifft
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def fast_trapezoidal_integration(y_vals):
    """Fast trapezoidal integration for numba compatibility"""
    if len(y_vals) < 2:
        return 0.0 if len(y_vals) == 0 else y_vals[0]
    
    integral = 0.0
    for i in range(len(y_vals) - 1):
        integral += (y_vals[i] + y_vals[i+1]) / 2.0
    return integral

@jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)

    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_piecewise(g_vals):
    """Compute norms using piecewise linear integration matching evaluator's method"""
    n = len(g_vals)

    if n <= 1:
        return 0.0, 0.0, 0.0

    # Compute L2 norm squared using trapezoidal-like integration
    # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    norm_2_sq = 0.0
    dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5

    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

    # Compute L1 norm (sum of absolute values)
    norm_1 = 0.0
    for i in range(n):
        norm_1 += abs(g_vals[i])

    # Compute L-infinity norm (maximum absolute value)
    norm_inf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > norm_inf:
            norm_inf = abs_val

    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f):
    """Compute the three norms needed for C2 calculation"""
    # Convert to numpy array
    f_arr = np.array(f, dtype=np.float64)

    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_arr)

    # Compute norms using piecewise integration
    norm_2_sq, norm_1, norm_inf = compute_norms_piecewise(g)

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f):
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_spectral_guided_function(n_points):
    """Create function guided by spectral properties for better C2"""
    # Create base frequency domain representation that favors good autoconvolution
    # This creates a "flat" spectrum that becomes concentrated in autoconvolution
    
    # Generate frequency domain pattern with specific characteristics
    freq_domain = np.zeros(n_points, dtype=complex)
    
    # Add multiple frequency components that promote favorable autoconvolution
    # These frequencies are chosen to create constructive interference in convolution
    
    # Base harmonics
    for k in range(1, min(10, n_points//4)):
        # Add frequency components with decreasing amplitudes
        amplitude = 1.0 / (k * k + 1)
        freq_domain[k] = amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi))
        if n_points - k >= 0:
            freq_domain[n_points - k] = amplitude * np.exp(-1j * np.random.uniform(0, 2*np.pi))
    
    # Add some low frequency components for energy concentration
    for i in range(3):
        freq_domain[i] = 0.5 * np.random.random() * np.exp(1j * np.random.uniform(0, 2*np.pi))
        freq_domain[n_points - 1 - i] = 0.5 * np.random.random() * np.exp(-1j * np.random.uniform(0, 2*np.pi))
    
    # Inverse FFT to get spatial domain function
    f_values = ifft(freq_domain).real
    
    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    
    # Normalize to reasonable scale
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / (max_val * 2.0)
    
    return f_values.tolist()

def create_multi_scale_peak_function(n_points):
    """Create function with multi-scale peak structure"""
    # Create domain
    x = np.linspace(-0.25, 0.25, n_points)
    f_values = np.zeros_like(x)
    
    # Multi-scale peak distribution to promote better autoconvolution
    # High frequency peaks (narrow, high amplitude) in center
    high_freq_count = min(5, n_points // 100)
    for i in range(high_freq_count):
        pos = np.random.uniform(-0.05, 0.05)
        amp = np.random.uniform(1.5, 3.0)
        width = np.random.uniform(0.005, 0.015)
        gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
        f_values += gaussian_peak
    
    # Medium frequency peaks (medium width, medium amplitude)
    med_freq_count = min(8, n_points // 50)
    for i in range(med_freq_count):
        pos = np.random.uniform(-0.15, 0.15)
        amp = np.random.uniform(0.8, 1.8)
        width = np.random.uniform(0.015, 0.04)
        gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
        f_values += gaussian_peak
    
    # Low frequency peaks (wide, low amplitude) at edges
    low_freq_count = min(6, n_points // 30)
    for i in range(low_freq_count):
        pos = np.random.uniform(-0.23, 0.23)
        amp = np.random.uniform(0.3, 0.8)
        width = np.random.uniform(0.03, 0.07)
        gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
        f_values += gaussian_peak
    
    # Apply smoothing to reduce sharp transitions
    if n_points > 100:
        window_size = max(3, min(21, int(n_points / 40)))
        if window_size % 2 == 0:
            window_size += 1
        f_values = signal.savgol_filter(f_values, window_size, 3)
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)
    
    # Normalize
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / (max_val * 1.5)
    
    return f_values.tolist()

def adaptive_peak_generation(n_points):
    """Generate peaks with strategic placement based on spectral analysis"""
    # Start with spectral-guided approach
    spectral_func = create_spectral_guided_function(n_points)
    spectral_array = np.array(spectral_func)
    
    # Create peaks from spectral function with some randomization
    f_values = np.zeros(n_points)
    
    # Sample from the spectral function to determine peak locations
    peak_indices = []
    for i in range(n_points):
        if spectral_array[i] > np.random.random() * 0.1:  # Threshold sampling
            peak_indices.append(i)
    
    # Ensure minimum spacing between peaks
    valid_indices = []
    if len(peak_indices) > 0:
        valid_indices.append(peak_indices[0])
        for idx in peak_indices[1:]:
            if abs(idx - valid_indices[-1]) > n_points // 20:  # Minimum spacing
                valid_indices.append(idx)
    
    # Create peaks from valid indices
    for idx in valid_indices[:min(15, len(valid_indices))]:  # Limit number of peaks
        pos = -0.25 + (idx / n_points) * 0.5
        amp = spectral_array[idx] * np.random.uniform(0.8, 1.5)
        width = np.random.uniform(0.01, 0.05)
        
        # Create Gaussian peak
        gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / width)**2)
        f_values += gaussian_peak
    
    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / (max_val * 1.5)
    
    return f_values.tolist()

def multi_objective_optimization(f_initial, max_iter=20):
    """Perform multi-objective optimization to improve C2"""
    # Objective function that balances C2 and other criteria
    def multi_objective(params):
        # Ensure non-negative
        params = np.maximum(params, 0)
        
        # Compute norms
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(params)
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        
        # Additional penalty terms for numerical stability
        penalty = 0
        if norm_inf > 1000:
            penalty += (norm_inf - 1000) * 0.01
        if norm_1 < 0.01:
            penalty += (0.01 - norm_1) * 100
            
        return -(c2 - penalty)
    
    # Use a simpler optimization with fewer iterations
    bounds = [(0, 1) for _ in range(len(f_initial))]
    
    try:
        result = differential_evolution(
            multi_objective,
            bounds,
            maxiter=max_iter,
            popsize=8,
            seed=42,
            strategy='best1bin'
        )
        
        if result.success:
            refined = np.maximum(result.x, 0).tolist()
            return refined
    except:
        pass
    
    return f_initial

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using spectral guidance"""
    start_time = time.time()
    
    # Try multiple strategies with different parameters
    best_c2 = 0.0
    best_function = []
    
    # Strategy 1: Spectral-guided function
    try:
        # Try with different point counts
        for n_points in [1000, 1500, 2000, 2500]:
            if time.time() - start_time > 80:
                break
                
            spectral_func = create_spectral_guided_function(n_points)
            c2 = compute_c2(spectral_func)
            
            if c2 > best_c2:
                best_c2 = c2
                best_function = spectral_func.copy()
                
    except Exception as e:
        warnings.warn(f"Spectral strategy failed: {str(e)}")
    
    # Strategy 2: Multi-scale peak function
    try:
        if time.time() - start_time > 80:
            pass
        else:
            multi_scale_func = create_multi_scale_peak_function(2000)
            c2 = compute_c2(multi_scale_func)
            
            if c2 > best_c2:
                best_c2 = c2
                best_function = multi_scale_func.copy()
                
    except Exception as e:
        warnings.warn(f"Multi-scale strategy failed: {str(e)}")
    
    # Strategy 3: Adaptive peak generation
    try:
        if time.time() - start_time > 80:
            pass
        else:
            adaptive_func = adaptive_peak_generation(2000)
            c2 = compute_c2(adaptive_func)
            
            if c2 > best_c2:
                best_c2 = c2
                best_function = adaptive_func.copy()
                
    except Exception as e:
        warnings.warn(f"Adaptive strategy failed: {str(e)}")
    
    # Final refinement using multi-objective optimization
    if best_function and time.time() - start_time < 85:
        try:
            refined_func = multi_objective_optimization(best_function, max_iter=15)
            refined_c2 = compute_c2(refined_func)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_function = refined_func
        except Exception as e:
            warnings.warn(f"Refinement failed: {str(e)}")
    
    # Final fallback if nothing worked well
    if not best_function:
        # Create a simple bell-shaped function
        n_points = 1000
        x = np.linspace(-0.25, 0.25, n_points)
        f_values = np.exp(-0.5 * (x / 0.1)**2)
        f_values = f_values / (np.max(f_values) * 1.5)
        best_function = f_values.tolist()
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")