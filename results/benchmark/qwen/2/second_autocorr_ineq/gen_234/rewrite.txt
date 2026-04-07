# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple
import time
import math

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞) where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Convert to numpy array
    f = np.array(f_values)

    # Compute autoconvolution g = f * f
    g = signal.convolve(f, f, mode='full')

    # Extract central portion (valid autoconvolution)
    half_len = len(f) - 1
    g = g[half_len:]  # Take right half

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)

    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_spectral_peak_function(n_steps: int) -> List[float]:
    """
    Create a function in spectral domain that inherently promotes good autoconvolution properties.
    Uses a combination of carefully positioned frequency components with varying strengths.
    """
    # Create frequency domain representation with multiple components
    frequencies = np.fft.fftfreq(n_steps, 0.5/n_steps)
    
    # Create magnitude spectrum with specific characteristics
    # Design spectrum to promote flat autoconvolution profiles which maximize C2
    magnitudes = np.zeros(n_steps)
    
    # Add components at different scales - use log-spaced frequencies
    n_components = min(20, n_steps // 4)
    
    # Logarithmic distribution of frequencies
    if n_components > 0:
        log_freqs = np.logspace(np.log10(1), np.log10(n_steps//2), n_components, base=10, dtype=int)
        log_freqs = log_freqs[log_freqs < n_steps//2]  # Filter out too high frequencies
        
        for i, freq_idx in enumerate(log_freqs):
            if freq_idx < n_steps//2:
                # Add energy at this frequency with decreasing strength
                strength = 1.0 / (1.0 + i * 0.3)  # Decreasing amplitude
                magnitudes[freq_idx] = strength
                if freq_idx > 0:
                    magnitudes[-freq_idx] = strength  # Conjugate symmetry
    
    # Add some low-frequency components for smoothness
    magnitudes[0] = 0.5  # DC component
    
    # Add some random phase to avoid local minima
    phases = np.random.uniform(0, 2*np.pi, n_steps)
    
    # Convert back to time domain
    fft_result = magnitudes * np.exp(1j * phases)
    
    # Use inverse FFT to get real-valued function
    f_real = np.real(np.fft.ifft(fft_result))
    
    # Ensure non-negativity and normalize
    f_real = np.maximum(f_real, 0.0)
    
    # Normalize to reasonable range
    max_val = np.max(f_real)
    if max_val > 0:
        f_real = f_real / (max_val * 2.0)
    
    # Add slight smoothing to reduce numerical artifacts
    if n_steps > 50:
        from scipy.ndimage import gaussian_filter1d
        f_real = gaussian_filter1d(f_real, sigma=0.8, mode='constant', cval=0.0)
        f_real = np.maximum(f_real, 0.0)
    
    return f_real.tolist()

def create_log_spaced_peaks(n_steps: int, n_peaks: int = None) -> List[float]:
    """
    Create a function with peaks distributed using logarithmic spacing for optimal coverage.
    """
    if n_peaks is None:
        n_peaks = min(15, max(3, n_steps // 100))
    
    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Initialize function
    f = np.zeros(n_steps)
    
    # Generate log-spaced peak positions
    if n_peaks > 0:
        # Create log-spaced positions across the domain
        log_min = np.log(0.01)  # Minimum relative position
        log_max = np.log(0.48)  # Maximum relative position (leaving margin)
        
        # Generate log-spaced positions (excluding boundaries)
        log_positions = np.logspace(log_min, log_max, n_peaks, base=np.e)
        
        # Map to actual positions in [-0.25, 0.25]
        total_range = 0.5
        offset = 0.03  # Minimum distance from edges
        
        peak_positions = []
        peak_heights = []
        peak_widths = []
        
        for i, log_pos in enumerate(log_positions):
            # Map to actual coordinate
            relative_pos = log_pos if i < len(log_positions) else 0.5
            pos = -0.25 + offset + relative_pos * (total_range - 2*offset)
            pos = np.clip(pos, -0.25 + offset, 0.25 - offset)
            peak_positions.append(pos)
            
            # Generate peak parameters with better statistical distribution
            width = np.random.uniform(0.01, 0.03)
            peak_widths.append(width)
            
            # Height inversely proportional to width for better control
            height = np.random.uniform(0.8, 2.0)
            peak_heights.append(height)
            
        # Create Gaussian peaks
        for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
            gaussian = height * np.exp(-0.5 * ((x - pos) / width) ** 2)
            f += gaussian
    
    # Ensure non-negativity
    f = np.maximum(f, 0.0)
    
    # Normalize appropriately
    max_val = np.max(f)
    if max_val > 0:
        f = f / (max_val * 1.2)
    
    # Apply smoothing to reduce numerical artifacts
    if n_steps > 50:
        from scipy.ndimage import gaussian_filter1d
        try:
            f = gaussian_filter1d(f, sigma=1.0, mode='constant', cval=0.0)
        except:
            pass
    f = np.maximum(f, 0.0)
    
    return f.tolist()

def refine_peak_parameters(best_function: List[float], n_steps: int) -> List[float]:
    """
    Perform targeted refinement of peak parameters to maximize C2.
    Focuses on adjusting peak heights and positions rather than full function space.
    """
    try:
        # Extract information about existing peaks
        x = np.linspace(-0.25, 0.25, n_steps)
        f_vals = np.array(best_function)
        
        # Identify dominant peaks through local maxima detection
        df = np.gradient(f_vals)
        ddf = np.gradient(df)
        
        # Find peaks - places where first derivative is zero and second is negative
        peaks = []
        for i in range(1, len(f_vals)-1):
            if df[i-1] > 0 and df[i] <= 0:  # Local maximum crossing
                if ddf[i] < 0:  # Actually a peak (not a trough)
                    peaks.append((i, f_vals[i], x[i]))
        
        # Sort by height to get the most significant peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:min(8, len(peaks))]
        
        if len(top_peaks) < 2:
            return best_function
            
        # Extract peak parameters for optimization
        peak_indices = [p[0] for p in top_peaks]
        peak_positions = [p[2] for p in top_peaks]  # x coordinates
        peak_heights = [p[1] for p in top_peaks]   # y values
        
        # Objective function for refinement - targets peak parameters
        def objective_function(params):
            # Create copy of original function
            temp_func = np.array(best_function)
            
            # Apply refined parameters to the identified peaks
            param_idx = 0
            for i, peak_idx in enumerate(peak_indices):
                if param_idx + 2 < len(params):
                    # Position adjustment (relative to original)
                    pos_shift = params[param_idx]
                    # Height adjustment (multiplier)
                    height_factor = params[param_idx + 1]
                    
                    # Apply adjustments
                    # Direct modification of nearby points
                    start_idx = max(0, peak_idx - 15)
                    end_idx = min(n_steps, peak_idx + 15)
                    
                    # Create adjustment based on Gaussian kernel around peak
                    peak_x = peak_positions[i]
                    new_x = peak_x + pos_shift
                    
                    # Apply smoothing effect around peak
                    for j in range(start_idx, end_idx):
                        dist = abs(x[j] - new_x)
                        # Gaussian influence
                        influence = np.exp(-0.5 * (dist / 0.02)**2)
                        # Modify value with weighted influence
                        temp_func[j] = max(0, temp_func[j] * (1.0 + (height_factor - 1.0) * influence * 0.5))
                    
                    param_idx += 2
            
            # Ensure non-negativity
            temp_func = np.maximum(temp_func, 0)
            
            try:
                c2_val = compute_c2(temp_func.tolist())
                return -c2_val  # Negative since we maximize
            except:
                return 1e10
                
        # Optimization bounds - allow reasonable adjustments
        bounds = []
        for i in range(len(peak_indices)):
            bounds.extend([(-0.02, 0.02), (0.7, 1.5)])  # Position shift, height factor
            
        # Perform optimization with limited iterations for speed
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=20,  # Reduced number of iterations for speed
            popsize=8, 
            seed=42,
            disp=False
        )
        
        if result.success:
            # Apply results to function
            temp_func = np.array(best_function)
            
            # Apply the refined peaks
            param_idx = 0
            for i, peak_idx in enumerate(peak_indices):
                if param_idx + 1 < len(result.x):
                    pos_shift = result.x[param_idx]
                    height_factor = result.x[param_idx + 1]
                    
                    # Apply to nearby points with smoothing
                    start_idx = max(0, peak_idx - 10)
                    end_idx = min(n_steps, peak_idx + 10)
                    
                    peak_x = peak_positions[i]
                    new_x = peak_x + pos_shift
                    
                    # Apply influence
                    for j in range(start_idx, end_idx):
                        dist = abs(x[j] - new_x)
                        influence = np.exp(-0.5 * (dist / 0.02)**2)
                        temp_func[j] = max(0, temp_func[j] * (1.0 + (height_factor - 1.0) * influence * 0.5))
                    
                    param_idx += 2
            
            temp_func = np.maximum(temp_func, 0)
            return temp_func.tolist()
            
    except Exception:
        pass
    
    return best_function

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value using spectral peak optimization.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    # Determine number of steps with time consideration
    n_steps = min(10000, max(100, 1000 + int(np.random.randint(0, 500) * 2)))
    
    # Strategy 1: Create function from spectral domain
    try:
        func1 = create_spectral_peak_function(n_steps)
        c2_1 = compute_c2(func1)
    except Exception:
        c2_1 = 0.0
        func1 = [0.5] * n_steps
    
    # Strategy 2: Create function with log-spaced peaks
    try:
        func2 = create_log_spaced_peaks(n_steps)
        c2_2 = compute_c2(func2)
    except Exception:
        c2_2 = 0.0
        func2 = [0.5] * n_steps
    
    # Strategy 3: Hybrid approach
    try:
        func3 = create_spectral_peak_function(n_steps)
        # Apply some targeted refinement
        func3 = refine_peak_parameters(func3, n_steps)
        c2_3 = compute_c2(func3)
    except Exception:
        c2_3 = 0.0
        func3 = [0.5] * n_steps
    
    # Choose the best among strategies
    candidates = [(c2_1, func1), (c2_2, func2), (c2_3, func3)]
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select best function and perform final refinement if time permits
    best_c2, best_function = candidates[0]
    
    # Final refinement if time allows
    if time.time() - start_time < 85:
        try:
            # Apply more targeted refinement on the best-known function
            refined = refine_peak_parameters(best_function, n_steps)
            refined_c2 = compute_c2(refined)
            
            if refined_c2 > best_c2:
                best_function = refined
        except Exception:
            pass
    
    # Final validation
    try:
        final_c2 = compute_c2(best_function)
        if final_c2 < 0.1:
            # If very poor performance, fall back to a structured approach
            best_function = create_log_spaced_peaks(n_steps)
    except Exception:
        best_function = [0.5] * n_steps
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")