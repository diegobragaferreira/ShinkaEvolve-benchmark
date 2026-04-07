# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import math
import random
from scipy.fft import fft, ifft

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)

    # Perform convolution (autoconvolution)
    # Using 'full' mode to get complete convolution result
    g = signal.convolve(f, f, mode='full')

    # Calculate the norms
    # ||g||₂² = sum(gᵢ²)
    g_squared = g * g
    norm_g_2_squared = np.sum(g_squared)

    # ||g||₁ = sum(|gᵢ|)
    norm_g_1 = np.sum(np.abs(g))

    # ||g||∞ = max(|gᵢ|)
    norm_g_inf = np.max(np.abs(g))

    return norm_g_2_squared, norm_g_1, norm_g_inf

def harmonic_envelope_construction(n_steps: int = None) -> list[float]:
    """
    Construct function using harmonic envelope approach that creates favorable autoconvolution properties.
    This builds a function based on mathematical principles that naturally lead to high C2 values.
    """
    if n_steps is None:
        n_steps = random.randint(800, 2000)
        
    # Generate base harmonic envelope using sine/cosine basis functions
    # This creates a smooth, well-behaved function with predictable autoconvolution properties
    
    # Create frequency domain representation that promotes good autoconvolution
    frequencies = np.fft.fftfreq(n_steps, 1.0/n_steps)
    # Apply a characteristic function that favors low-frequency content for smoother autoconvolution
    # This creates a natural "envelope" that promotes balanced peaks
    envelope = np.exp(-0.5 * (frequencies / (n_steps/20))**2) * np.exp(-0.1 * np.abs(frequencies))
    
    # Ensure DC component is positive
    envelope[0] = max(envelope[0], 0.1)
    
    # Generate random phase for diversity while maintaining the spectral shape
    phases = np.random.uniform(0, 2*np.pi, n_steps)
    phases[0] = 0  # Keep DC component real
    
    # Apply inverse FFT to get spatial domain function
    # Combine amplitude envelope with random phases
    complex_signal = envelope * np.exp(1j * phases)
    spatial_function = np.real(ifft(complex_signal))
    
    # Convert to step function with desired properties
    # First ensure non-negative values with some smoothing
    f_values = np.abs(spatial_function) + 0.01  # Add small constant to avoid zeros
    
    # Apply adaptive transformation to create optimal peak structure
    # Transform function to emphasize preferred regions
    transformed = np.power(f_values, 1.3)  # Slightly enhance peaks
    
    # Apply final smoothing to reduce noise while preserving structure
    if n_steps > 50:
        # Use Savitzky-Golay filter for better shape preservation
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            transformed = signal.savgol_filter(transformed, window_size, 3)
    
    # Ensure non-negativity again after filtering
    transformed = np.maximum(transformed, 0)
    
    # Normalize to reasonable scale
    max_val = np.max(transformed)
    if max_val > 0:
        transformed = transformed / max_val * 2.0
        
    return transformed.tolist()

def adaptive_peak_balancing(f_values: list[float]) -> list[float]:
    """Apply adaptive refinement to balance autoconvolution properties."""
    # Convert to numpy for processing
    f_array = np.array(f_values)
    
    # Compute current autoconvolution
    g = signal.convolve(f_array, f_array, mode='full')
    g = g[len(f_array)-1:2*len(f_array)-1]
    
    # Analyze the autoconvolution profile
    g_abs = np.abs(g)
    g_max = np.max(g_abs)
    g_mean = np.mean(g_abs)
    
    # If autoconvolution has too sharp peaks relative to mean, flatten it
    if g_max > 0 and g_mean > 0 and g_max / g_mean > 5.0:
        # Apply sigmoid transformation to flatten extreme values
        # This reduces the dominance of large autoconvolution values
        g_scaled = g_abs / g_max
        flattened = 1.0 / (1.0 + np.exp(-10 * (g_scaled - 0.5)))  # Sigmoid curve
        # Scale back to roughly preserve magnitude
        g_flattened = g_max * flattened
        
        # Reconstruct function to match flattened autoconvolution properties
        # This is an approximation - we'll focus on adjusting the original function
        adjustment_factor = 0.95
        adjusted_f = f_array * adjustment_factor
        return adjusted_f.tolist()
    
    return f_values

def enhanced_frequency_domain_construction(n_steps: int = None) -> list[float]:
    """
    Enhanced frequency domain approach that specifically targets optimal C2 properties.
    Uses a more sophisticated spectral shaping technique.
    """
    if n_steps is None:
        n_steps = random.randint(1000, 2500)
        
    # Create a more structured frequency domain envelope
    frequencies = np.fft.fftfreq(n_steps, 1.0/n_steps)
    
    # Build a custom envelope designed to promote good autoconvolution
    # This envelope has:
    # 1. Low frequency dominance (smooth structure)
    # 2. Controlled high frequency content (avoiding noise)
    # 3. Strategic nulls and peaks to create favorable convolution
    
    # Low frequency emphasis with controlled roll-off
    low_freq_envelope = np.exp(-0.2 * (frequencies / (n_steps/15))**2)
    
    # Add some controlled high-frequency content
    high_freq_component = 0.3 * np.exp(-0.5 * (frequencies / (n_steps/5))**2) * \
                          np.sin(0.5 * frequencies / (n_steps/100))
    
    # Combine envelopes
    combined_envelope = low_freq_envelope + 0.5 * high_freq_component
    
    # Add some random modulation to maintain diversity
    modulation = 0.1 * np.random.uniform(0.8, 1.2, n_steps)
    combined_envelope = combined_envelope * modulation
    
    # Ensure non-negativity and normalize
    combined_envelope = np.maximum(combined_envelope, 0.01)
    
    # Generate phases
    phases = np.random.uniform(0, 2*np.pi, n_steps)
    phases[0] = 0  # DC component remains real
    
    # Apply inverse FFT
    complex_signal = combined_envelope * np.exp(1j * phases)
    
    # Extract real part and ensure positivity
    spatial_function = np.real(ifft(complex_signal))
    f_values = np.abs(spatial_function) + 0.01  # Add minimum to avoid zero values
    
    # Apply adaptive smoothing and normalization
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_values = signal.savgol_filter(f_values, window_size, 3)
    
    # Final normalization
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / max_val * 2.0
    
    return f_values.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses a hybrid approach combining harmonic envelope construction with adaptive refinement.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Try multiple construction strategies and choose the best
    best_result = []
    best_c2 = 0
    
    # Strategy 1: Harmonic envelope construction
    try:
        harmonic_result = harmonic_envelope_construction()
        if harmonic_result:
            # Evaluate harmonic result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(harmonic_result)
            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = harmonic_result
    except Exception as e:
        pass

    # Strategy 2: Enhanced frequency domain construction
    try:
        enhanced_result = enhanced_frequency_domain_construction()
        if enhanced_result:
            # Evaluate enhanced result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(enhanced_result)
            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = enhanced_result
    except Exception as e:
        pass

    # Strategy 3: Adaptive peak balancing of the best so far
    if best_result:
        try:
            balanced_result = adaptive_peak_balancing(best_result)
            # Evaluate balanced result
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(balanced_result)
            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = balanced_result
        except Exception as e:
            pass

    # Fallback to basic approach if none work
    if not best_result:
        n_steps = 1000
        # Simple but effective approach: create a bell-curved function
        x = np.linspace(-1, 1, n_steps)
        base_shape = np.exp(-x**2 / 2)
        base_shape = 0.8 * (base_shape / np.max(base_shape)) + 0.2
        best_result = base_shape.tolist()

    # Final optimization step: mild local adjustment
    if best_result:
        try:
            # Apply small perturbations to see if we can improve
            final_result = best_result.copy()
            
            # Apply some transformations in case we're close to optimal
            for i in range(len(final_result)):
                if random.random() < 0.1:  # 10% chance to modify
                    # Small random adjustment
                    factor = random.uniform(0.95, 1.05)
                    final_result[i] = max(0, final_result[i] * factor)
            
            # Re-evaluate to see if this helped
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(final_result)
            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = final_result
                    
        except Exception as e:
            pass

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")