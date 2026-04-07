# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import time
from typing import List
import math

def construct_function() -> List[float]:
    """
    Construct a step function that maximizes C2 = ||g||₂² / (||g||₁ · ||g||∞)
    where g = f*f (autoconvolution) and f is the step function.
    
    Uses spectral-guided optimization approach:
    1. Start with a good frequency-domain template
    2. Apply adaptive spectral shaping to increase C2
    3. Use FFT for fast autoconvolution
    4. Direct spectral optimization to maximize target metric
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    start_time = time.time()
    max_time = 90.0
    
    # Parameters
    n_steps = 4000  # Fixed resolution for consistency and speed
    domain_width = 0.5  # [-0.25, 0.25]
    
    # Generate base spectral template (sinc-like pattern for good autoconvolution properties)
    x = np.linspace(-domain_width/2, domain_width/2, n_steps)
    
    # Create a base function with structured frequency content
    # Start with a modified sinc function which tends to produce good autoconvolutions
    base_freq = 10.0  # Base frequency for spectral shaping
    frequencies = fftfreq(n_steps, domain_width/n_steps)
    
    # Create a frequency spectrum that promotes good autoconvolution properties
    # This is designed to create peaks in the frequency domain that translate to 
    # desirable properties in the time domain
    spectrum = np.zeros(n_steps)
    
    # Add several peaks at different frequencies to encourage good spectral mixing
    peak_frequencies = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]  # Frequencies in cycles per domain
    peak_amplitudes = [1.0, 0.7, 0.5, 0.6, 0.4, 0.3]  # Decreasing amplitudes to avoid over-sharpening
    
    for freq, amp in zip(peak_frequencies, peak_amplitudes):
        # Convert frequency to index
        if freq == 0.0:
            # DC component
            spectrum[0] = amp
        else:
            # Find closest frequency bin
            idx = int(freq * n_steps / domain_width)
            if 0 < idx < n_steps//2:
                # Use Gaussian envelope for smoother transitions
                sigma = 0.5
                for i in range(max(0, idx-2), min(n_steps//2, idx+3)):
                    distance = abs(i - idx)
                    spectrum[i] += amp * np.exp(-distance**2 / (2*sigma**2))
                    if i != 0 and i != n_steps//2:
                        spectrum[n_steps-i] += amp * np.exp(-distance**2 / (2*sigma**2))
    
    # Ensure symmetry for real-valued output
    spectrum[0] = abs(spectrum[0])  # DC component should be real
    for i in range(1, n_steps//2):
        spectrum[n_steps-i] = np.conj(spectrum[i])  # Hermitian symmetry
    
    # Inverse FFT to get time domain function
    f = np.real(ifft(spectrum))
    
    # Normalize the function
    f = np.maximum(f, 0)  # Ensure non-negative
    if np.sum(f) > 0:
        f = f / np.sum(f) * 10  # Scale appropriately
    
    # Quick evaluation of initial function
    def compute_c2_direct(f_vals):
        """Direct C2 computation for fast evaluation"""
        try:
            # Use FFT for fast convolution
            f_fft = fft(f_vals)
            g_fft = f_fft * np.conj(f_fft)  # Autoconvolution in frequency domain
            g = np.real(ifft(g_fft))
            g = g[:len(f_vals)]  # Keep first part
            
            # Compute norms
            norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
            norm_2_sq = np.sum(g**2) / len(g)  # Simplified version
            norm_inf = np.max(np.abs(g))
            
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 0.0
                
            return norm_2_sq / (norm_1 * norm_inf)
        except Exception:
            return 0.0
    
    # Initial evaluation
    current_c2 = compute_c2_direct(f)
    
    # Spectral optimization loop
    max_iterations = 100  # Limit iterations to stay within time budget
    iteration = 0
    
    while iteration < max_iterations and (time.time() - start_time) < max_time * 0.9:
        if (time.time() - start_time) > max_time * 0.95:
            break
            
        # Create a candidate by modifying spectral components
        # This is a simplified spectral manipulation approach
        candidate_spectrum = spectrum.copy()
        
        # Randomly modify some frequency components
        num_modifications = max(1, n_steps // 50)
        for _ in range(num_modifications):
            # Pick a random frequency bin (avoid DC and Nyquist)
            bin_idx = np.random.randint(1, n_steps//2)
            
            # Modify amplitude slightly
            modification_factor = 1.0 + np.random.normal(0, 0.05)  # Small change
            candidate_spectrum[bin_idx] *= modification_factor
            
            # Keep symmetric for real output
            if bin_idx != n_steps - bin_idx:
                candidate_spectrum[n_steps - bin_idx] = np.conj(candidate_spectrum[bin_idx])
        
        # Make sure DC component is real
        candidate_spectrum[0] = abs(candidate_spectrum[0])
        
        # Inverse FFT to get candidate function
        try:
            candidate_f = np.real(ifft(candidate_spectrum))
            candidate_f = np.maximum(candidate_f, 0)  # Non-negative
            
            if np.sum(candidate_f) > 0:
                candidate_f = candidate_f / np.sum(candidate_f) * 10
            else:
                candidate_f = np.ones_like(candidate_f) * 0.1
            
            # Evaluate candidate
            candidate_c2 = compute_c2_direct(candidate_f)
            
            # Accept improvement or with probability based on difference
            if candidate_c2 > current_c2:
                current_c2 = candidate_c2
                f = candidate_f.copy()
                spectrum = candidate_spectrum.copy()
            else:
                # Occasionally accept worse solutions for escape from local optima
                delta_c2 = candidate_c2 - current_c2
                acceptance_prob = math.exp(delta_c2 * 10) if delta_c2 < 0 else 1.0
                if np.random.random() < acceptance_prob * 0.1:  # Lower chance to accept worse
                    current_c2 = candidate_c2
                    f = candidate_f.copy()
                    spectrum = candidate_spectrum.copy()
                    
        except Exception:
            pass
            
        iteration += 1
    
    # Final cleanup and return with added robustness
    final_f = np.maximum(f, 0)
    if np.sum(final_f) > 0:
        final_f = final_f / np.sum(final_f) * 10
    
    # Add small noise for robustness if needed
    noise_level = 0.01
    noisy_f = final_f + np.random.normal(0, noise_level, len(final_f))
    noisy_f = np.maximum(noisy_f, 0)
    
    # Convert to list
    result = noisy_f.tolist()
    
    # Ensure minimal structure for compatibility
    if len(result) < 100:
        result = [0.5] * 100
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")