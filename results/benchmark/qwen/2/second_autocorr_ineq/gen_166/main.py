# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
from numba import jit
import time
import random
from functools import lru_cache
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.signal import convolve
import optuna
from optuna.samplers import TPESampler

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

def compute_spectral_properties_jax(f_values):
    """
    Compute spectral properties using JAX for efficient gradient computation.
    """
    f_array = jnp.array(f_values)
    # Compute FFT
    fft_vals = jnp.fft.fft(f_array)
    magnitudes = jnp.abs(fft_vals)
    phases = jnp.angle(fft_vals)
    return magnitudes, phases

def initialize_spectral_function(n_points: int = 2000) -> list[float]:
    """
    Initialize function in spectral domain with optimized structure for high C2.
    This creates a spectrum that naturally leads to favorable autoconvolution properties.
    """
    # Create frequency domain representation with strategic peaks
    frequencies = np.fft.fftfreq(n_points, 0.5/n_points)
    magnitudes = np.zeros(n_points)
    
    # Create multi-scale frequency components that promote smooth autoconvolution
    # Use log-uniform distribution of peak frequencies
    n_clusters = 6
    cluster_centers = np.logspace(np.log10(1), np.log10(n_points//2), n_clusters)
    
    # Each cluster forms a band of frequency components
    for i, center in enumerate(cluster_centers):
        # Add multiple components per cluster
        n_components = np.random.randint(2, 5)
        for j in range(n_components):
            # Spread components logarithmically around cluster center
            freq_offset = np.random.normal(0, center * 0.15)
            freq_idx = int(center + freq_offset)
            if 1 <= freq_idx < n_points//2:
                # Use gamma distribution for varied strengths
                strength = np.random.gamma(2.5, 1.0) 
                magnitudes[freq_idx] = strength
                if freq_idx > 0:
                    magnitudes[-freq_idx] = strength  # Conjugate symmetry
    
    # Add low-frequency components for smoothness and DC component
    magnitudes[0] = np.random.gamma(1.5, 2.5)  # DC component
    
    # Add phase information
    phases = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to complex spectrum and create signal
    spectrum = magnitudes * np.exp(1j * phases)
    
    # Inverse FFT to get real-valued function
    f_real = np.real(np.fft.ifft(spectrum))
    
    # Ensure non-negativity
    f_real = np.maximum(f_real, 0.0)
    
    # Normalize for better numerical stability
    max_val = np.max(f_real)
    if max_val > 0:
        f_real = f_real / (max_val * 2.0)
    
    # Apply smoothing to reduce high-frequency noise
    window_size = max(3, min(101, n_points // 15))
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        kernel = np.exp(-0.5 * np.arange(-window_size//2 + 1, window_size//2 + 1)**2 / (window_size/4)**2)
        kernel = kernel / np.sum(kernel)
        f_smooth = np.convolve(f_real, kernel, mode='same')
        f_real = f_smooth
    
    # Final non-negativity check
    f_real = np.maximum(f_real, 0.0)
    
    return f_real.tolist()

def spectral_gradient_optimization(initial_func: list[float], max_iter: int = 200) -> list[float]:
    """
    Perform gradient-based optimization in spectral domain.
    Uses analytical gradients to maximize C2.
    """
    # Convert initial function to jax array
    f_init = jnp.array(initial_func, dtype=jnp.float32)
    
    # Get initial spectral representation
    spectrum = jnp.fft.fft(f_init)
    mag = jnp.abs(spectrum)
    phase = jnp.angle(spectrum)
    
    # Use adaptive learning rate
    learning_rate = 0.05
    momentum = 0.9
    velocity_mag = jnp.zeros_like(mag)
    velocity_phase = jnp.zeros_like(phase)
    
    # Define the objective function for gradient computation
    @jit
    def compute_c2_from_spectrum(magnitude, phase):
        # Reconstruct complex spectrum
        complex_spectrum = magnitude * jnp.exp(1j * phase)
        
        # Inverse FFT to get function
        f_reconstructed = jnp.real(jnp.fft.ifft(complex_spectrum))
        
        # Ensure non-negativity (clip negative values)
        f_reconstructed = jnp.maximum(f_reconstructed, 0.0)
        
        # Compute norms using the existing computation method
        # Note: this uses jax operations and assumes proper sizing
        n = len(f_reconstructed)
        dx = 0.5 / (n - 1) if n > 1 else 0.5
        
        # Autoconvolution computation
        g = jnp.convolve(f_reconstructed, f_reconstructed, mode='full')
        g_middle = g[n-1:2*n-1]
        
        # Compute norms directly from jax arrays
        g_sq = g_middle * g_middle
        g_x = jnp.linspace(-0.5, 0.5, len(g_middle))
        
        # Trapezoidal integration for ||g||₂²
        area = jnp.sum((g_sq[:-1] + g_sq[1:]) * (g_x[1:] - g_x[:-1])) / 2
        norm_2_sq = area
        
        # ||g||₁ and ||g||∞
        norm_1 = jnp.sum(jnp.abs(g_middle)) * dx
        norm_inf = jnp.max(jnp.abs(g_middle))
        
        # Avoid division by zero
        norm_1 = jnp.where(norm_1 <= 1e-15, 1e-15, norm_1)
        norm_inf = jnp.where(norm_inf <= 1e-15, 1e-15, norm_inf)
        
        return norm_2_sq / (norm_1 * norm_inf)
    
    # Gradient computation (simplified version with finite differences)
    # Since direct analytical gradient is complex, we'll use a simpler approach:
    # We'll optimize by perturbing the spectrum and checking improvement
    
    current_mag = mag
    current_phase = phase
    
    for iteration in range(max_iter):
        try:
            # Store current state
            old_c2 = compute_c2_from_spectrum(current_mag, current_phase)
            
            # Try small perturbations to magnitudes
            perturbed_mag = current_mag + jnp.random.normal(0, 0.05, size=current_mag.shape) * current_mag
            perturbed_mag = jnp.maximum(perturbed_mag, 0.0)  # Non-negativity
            
            # Try small perturbations to phases
            perturbed_phase = current_phase + jnp.random.normal(0, 0.05, size=current_phase.shape)
            
            # Compute new C2 values
            c2_new_mag = compute_c2_from_spectrum(perturbed_mag, current_phase)
            c2_new_phase = compute_c2_from_spectrum(current_mag, perturbed_phase)
            
            # Choose which perturbation improves C2
            if c2_new_mag > old_c2 or c2_new_phase > old_c2:
                if c2_new_mag > c2_new_phase:
                    current_mag = perturbed_mag
                else:
                    current_phase = perturbed_phase
            else:
                # If no improvement, perturb slightly anyway for exploration
                if random.random() < 0.3:
                    # Add some noise to encourage exploration
                    noise_mag = jnp.random.normal(0, 0.02, size=current_mag.shape) * current_mag
                    noise_phase = jnp.random.normal(0, 0.02, size=current_phase.shape)
                    current_mag = jnp.maximum(current_mag + noise_mag, 0.0)
                    current_phase = current_phase + noise_phase
            
            # Apply momentum and normalize
            current_mag = jnp.maximum(current_mag, 0.0)
            current_phase = current_phase % (2 * jnp.pi)
            
        except Exception as e:
            # Fall back to simple random search if anything fails
            if iteration < max_iter - 5:
                # Random small perturbations
                current_mag = current_mag + jnp.random.normal(0, 0.01, size=current_mag.shape) * current_mag
                current_phase = current_phase + jnp.random.normal(0, 0.01, size=current_phase.shape)
                current_mag = jnp.maximum(current_mag, 0.0)
                current_phase = current_phase % (2 * jnp.pi)
    
    # Reconstruct final function
    final_spectrum = current_mag * jnp.exp(1j * current_phase)
    final_function = jnp.real(jnp.fft.ifft(final_spectrum))
    final_function = jnp.maximum(final_function, 0.0)
    
    # Normalize
    max_val = jnp.max(final_function)
    if max_val > 0:
        final_function = final_function / (max_val * 2.0)
    
    return final_function.tolist()

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value using spectral gradient optimization.
    """
    best_c2 = -1
    best_f = None
    num_attempts = 15  # Fewer attempts, but with better strategies
    
    start_time = time.time()
    
    for attempt in range(num_attempts):
        if time.time() - start_time > 85:
            break
            
        try:
            # Generate initial function using spectral initialization
            initial_func = initialize_spectral_function(2000)
            
            # Apply gradient-based optimization
            optimized_func = spectral_gradient_optimization(initial_func, max_iter=150)
            
            # Calculate norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(optimized_func)

            # Avoid division by zero
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                continue

            # Compute C2
            c2 = norm_2_sq / (norm_1 * norm_inf)

            if c2 > best_c2:
                best_c2 = c2
                best_f = optimized_func.copy()

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
