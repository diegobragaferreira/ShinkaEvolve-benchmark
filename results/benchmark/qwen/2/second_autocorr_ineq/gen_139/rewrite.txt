# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)

    # Create step function on [-1/4, 1/4] with equal spacing
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
    # For each pair of adjacent points, integrate quadratic function
    g_sq = g**2
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

def evaluate_c2_score(f_values):
    """Compute C2 score for given function values"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

def generate_spectral_design(n_points, target_spectrum_type="smooth"):
    """
    Generate a spectral design with desired characteristics that will produce good autoconvolution.
    """
    # Create frequency grid
    freqs = fftfreq(n_points, 1.0/n_points)
    
    # Generate target magnitude spectrum based on type
    if target_spectrum_type == "smooth":
        # Smooth spectrum with gradual roll-off
        magnitude = np.exp(-0.1 * freqs**2) * (1 + 0.3 * np.sin(0.5 * freqs))
    elif target_spectrum_type == "peaked":
        # Spectrum with dominant low-frequency content and controlled high-frequency decay
        magnitude = np.exp(-0.05 * freqs**2) * (1 + 0.4 * np.cos(0.3 * freqs))
        magnitude = np.maximum(magnitude, 0.1)
    else:  # default smooth
        magnitude = np.exp(-0.1 * freqs**2) * (1 + 0.2 * np.sin(0.2 * freqs))
    
    # Add some noise to make it less artificial
    noise_level = 0.05
    magnitude += np.random.normal(0, noise_level, len(magnitude)) * magnitude
    
    # Ensure non-negative magnitude
    magnitude = np.maximum(magnitude, 0.01)
    
    # Create complex spectrum with random phases
    phase = np.random.uniform(0, 2*np.pi, len(freqs))
    spectrum = magnitude * np.exp(1j * phase)
    
    return spectrum, freqs

def spectral_to_time_domain(spectrum):
    """
    Convert spectral representation back to time domain function.
    """
    # Inverse FFT to get time-domain signal
    time_signal = ifft(spectrum).real
    
    # Normalize to reasonable range
    if np.std(time_signal) > 0:
        time_signal = time_signal / np.std(time_signal) * 2
    
    # Ensure non-negativity
    time_signal = np.maximum(time_signal, 0)
    
    # Normalize to mean around 1 (for better autoconvolution properties)
    if np.mean(time_signal) > 0:
        time_signal = time_signal / np.mean(time_signal) * 1.5
    
    return time_signal

def optimize_spectral_peak_position(spectrum, freqs, target_freq=0.0):
    """
    Adjust the spectrum to shift peak toward target frequency.
    """
    # Make a copy to avoid modifying original
    new_spectrum = spectrum.copy()
    
    # Create a shift in frequency domain
    shifts = np.exp(-2j * np.pi * target_freq * freqs / len(freqs))
    new_spectrum = new_spectrum * shifts
    
    # Apply a small amount of smoothing to maintain spectrum integrity
    # This helps avoid overfitting to specific frequencies
    window = signal.windows.hann(len(freqs))
    new_spectrum = new_spectrum * window
    
    return new_spectrum

def create_spectral_peak_function(n_steps=1000):
    """
    Create step function by designing its spectrum appropriately.
    """
    # Generate initial spectral design
    spectrum, freqs = generate_spectral_design(n_steps, "peaked")
    
    # Apply multiple optimizations to improve autoconvolution properties
    # Phase 1: Shift main energy to low frequencies
    spectrum = optimize_spectral_peak_position(spectrum, freqs, target_freq=0.0)
    
    # Phase 2: Add some controlled high-frequency content to maintain smoothness
    high_freq_weight = 0.3
    high_freq_spectrum = generate_spectral_design(n_steps, "smooth")[0]
    spectrum = (1 - high_freq_weight) * spectrum + high_freq_weight * high_freq_spectrum
    
    # Phase 3: Apply frequency-dependent filtering to optimize autoconvolution
    # Create filter that emphasizes frequencies beneficial for good C2
    filter_weights = np.exp(-0.02 * freqs**2) * (1 + 0.1 * np.cos(0.1 * freqs))
    spectrum = spectrum * filter_weights
    
    # Convert to time domain
    time_signal = spectral_to_time_domain(spectrum)
    
    # Ensure proper length
    if len(time_signal) > n_steps:
        time_signal = time_signal[:n_steps]
    elif len(time_signal) < n_steps:
        time_signal = np.pad(time_signal, (0, n_steps - len(time_signal)), 'constant')
    
    # Apply final smoothing to reduce extreme variations
    if n_steps > 100:
        window_size = min(51, n_steps // 5)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            time_signal = signal.savgol_filter(time_signal, window_size, 3)
    
    # Ensure non-negativity and normalize
    time_signal = np.maximum(time_signal, 0)
    if np.max(time_signal) > 0:
        time_signal = time_signal / np.max(time_signal) * 2.0
    
    return time_signal.tolist()

def spectral_peak_refinement(initial_function, n_iterations=20):
    """
    Refine the spectral peak function by iterative optimization.
    """
    current_function = np.array(initial_function)
    
    for iteration in range(n_iterations):
        # Create new function with slight variations
        new_function = current_function.copy()
        
        # Apply small random modifications
        modification_strength = 0.1 / (iteration + 1)  # Decreasing strength
        
        # Modify a random subset of points
        n_modify = max(1, min(len(new_function) // 10, 20))
        indices_to_modify = np.random.choice(len(new_function), n_modify, replace=False)
        
        for idx in indices_to_modify:
            # Apply multiplicative or additive changes
            if np.random.random() < 0.5:
                factor = 1 + np.random.normal(0, modification_strength)
                new_function[idx] = max(0, new_function[idx] * factor)
            else:
                delta = np.random.normal(0, modification_strength * max(1, new_function[idx]))
                new_function[idx] = max(0, new_function[idx] + delta)
        
        # Evaluate improvement
        current_score = evaluate_c2_score(current_function.tolist())
        new_score = evaluate_c2_score(new_function.tolist())
        
        # Accept improvement or sometimes accept worse solutions for exploration
        if new_score >= current_score:
            current_function = new_function
        elif np.random.random() < 0.1:  # 10% chance to accept worse solution
            current_function = new_function
    
    return current_function.tolist()

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses spectral peak optimization approach.
    """
    start_time = time.time()
    
    # Phase 1: Multiple spectral peak constructions
    best_result = []
    best_c2 = 0
    
    # Try several spectral peak constructions
    for attempt in range(10):
        if time.time() - start_time > 80:
            break
            
        try:
            # Create spectral peak function with adaptive parameters
            n_steps = np.random.randint(500, 1500)
            spectral_func = create_spectral_peak_function(n_steps)
            
            # Evaluate spectral function
            c2 = evaluate_c2_score(spectral_func)
            
            if c2 > best_c2:
                best_c2 = c2
                best_result = spectral_func.copy()
        except Exception as e:
            continue
    
    # Phase 2: Spectral refinement of best result
    if best_result and time.time() - start_time < 75:
        try:
            refined_result = spectral_peak_refinement(best_result, 30)
            refined_c2 = evaluate_c2_score(refined_result)
            
            if refined_c2 > best_c2:
                best_result = refined_result
                best_c2 = refined_c2
        except Exception as e:
            pass
    
    # Phase 3: Additional refinement with hybrid approach
    if best_result and time.time() - start_time < 70:
        try:
            # Apply some additional targeted adjustments
            final_result = best_result.copy()
            
            # Apply some local adjustments to enhance performance
            n_adjustments = max(1, len(final_result) // 20)
            adjustment_indices = np.random.choice(len(final_result), n_adjustments, replace=False)
            
            for idx in adjustment_indices:
                # Small multiplicative adjustments to enhance balance
                factor = 1.0 + np.random.normal(0, 0.03)  # Small adjustment
                final_result[idx] = max(0, final_result[idx] * factor)
            
            # Evaluate final result
            final_c2 = evaluate_c2_score(final_result)
            if final_c2 > best_c2:
                best_result = final_result
                best_c2 = final_c2
                
        except Exception as e:
            pass
    
    # Phase 4: Final fallback to robust construction
    if len(best_result) == 0 or best_c2 < 0.8:
        try:
            # Use a more conservative spectral approach with fixed parameters
            n_steps = 1000
            best_result = create_spectral_peak_function(n_steps)
        except Exception as e:
            # Last resort: simple symmetric function
            n_steps = 1000
            x = np.linspace(-1, 1, n_steps)
            base_shape = np.exp(-x**2 / 2)
            base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
            best_result = base_shape.tolist()
    
    # Final evaluation and validation
    if best_result:
        try:
            final_c2 = evaluate_c2_score(best_result)
            if final_c2 <= 0:
                # Fallback if final evaluation fails
                n_steps = 1000
                x = np.linspace(-1, 1, n_steps)
                base_shape = np.exp(-x**2 / 2)
                base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
                best_result = base_shape.tolist()
        except:
            # Final fallback
            n_steps = 1000
            x = np.linspace(-1, 1, n_steps)
            base_shape = np.exp(-x**2 / 2)
            base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
            best_result = base_shape.tolist()
    
    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")