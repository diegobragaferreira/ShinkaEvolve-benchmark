# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
from scipy.ndimage import gaussian_filter1d
import time

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation with improved numerical stability.
    """
    # Create step function with proper spacing
    n = len(f_values)
    if n < 1:
        return 0.0, 0.0, 0.0

    # Convert to numpy array
    f = np.array(f_values, dtype=np.float64)

    # Create the step function on [-1/4, 1/4] with equal spacing
    dx = 0.5 / (n - 1) if n > 1 else 0.5
    x = np.linspace(-0.25, 0.25, n)

    # Use fast convolution via FFT for better performance
    # Pad arrays to avoid circular convolution effects
    padded_len = 2 * n - 1
    f_padded = np.pad(f, (0, padded_len - n), 'constant', constant_values=0)
    
    # Compute autoconvolution using FFT
    f_fft = np.fft.fft(f_padded)
    g_fft = f_fft * np.conj(f_fft)
    g = np.fft.ifft(g_fft).real[:padded_len]
    
    # Keep only the middle part (proper autoconvolution)
    g = g[n-1:2*n-1]
    
    # Adjust for the fact that our function is defined on [-1/4, 1/4]
    # The actual convolution interval is [-1/2, 1/2]
    g_x = np.linspace(-0.5, 0.5, len(g))

    # Now compute the required norms
    # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
    g_sq = g * g
    # Trapezoidal rule: sum((y[i] + y[i+1])/2 * delta_x) 
    # But we're integrating over variable x, so we compute the correct area
    dx_vals = np.diff(g_x)
    if len(dx_vals) == 0:
        norm_2_sq = 0.0
    else:
        # For trapezoidal integration of g^2
        g_sq_areas = (g_sq[:-1] + g_sq[1:]) * dx_vals / 2.0
        norm_2_sq = np.sum(g_sq_areas)

    # ||g||₁ (L1 norm) - approximate via summation with proper scaling 
    norm_1 = np.sum(np.abs(g)) * dx  # dx is the step size

    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g))

    # Numerical stability checks
    if np.isnan(norm_2_sq) or np.isinf(norm_2_sq):
        norm_2_sq = 0.0
    if np.isnan(norm_1) or np.isinf(norm_1) or norm_1 <= 0:
        norm_1 = 1e-12
    if np.isnan(norm_inf) or np.isinf(norm_inf) or norm_inf <= 0:
        norm_inf = 1e-12

    return norm_2_sq, norm_1, norm_inf

def adaptive_gaussian_construction():
    """
    Construct step function based on adaptive Gaussian peaks with improved strategy.
    """
    # Generate initial parameters
    n_peaks = np.random.randint(8, 30)
    n_points = np.random.randint(1000, 8000)  # Increased resolution for better performance

    # Create x-axis evenly spaced
    x = np.linspace(-0.25, 0.25, n_points)

    # Initialize function
    f = np.zeros(n_points)

    # Place peaks with logarithmic spacing to prevent narrow autoconvolution
    # and avoid clustering
    peak_positions = []
    peak_amplitudes = []
    
    # Use logarithmic distribution for peak positions
    # This helps avoid clustering around center and edges
    log_min = np.log(0.02)
    log_max = np.log(0.12)
    log_spaced_positions = np.logspace(log_min, log_max, n_peaks)
    
    # Choose positions with appropriate spacing
    for i in range(n_peaks):
        # Pick a position with logarithmic spacing bias but add randomness
        base_pos = log_spaced_positions[i]
        # Alternate sides to distribute evenly
        side = 1 if i % 2 == 0 else -1
        pos = side * base_pos + np.random.uniform(-0.01, 0.01)
        
        # Clip to valid range
        pos = np.clip(pos, -0.23, 0.23)
        
        # Avoid too close proximity
        valid_position = True
        for existing_pos in peak_positions:
            if abs(pos - existing_pos) < 0.02:
                valid_position = False
                break
                
        if valid_position:
            peak_positions.append(pos)
            
    # Generate amplitudes with better distribution
    # Start with exponential distribution but adjust based on position 
    # to avoid very sharp peaks that cause instability
    for pos in peak_positions:
        # Use distribution that decreases with distance from center
        center_distance = abs(pos)
        # Base amplitude with decay factor based on distance from center
        base_amp = np.random.exponential(0.5) * np.exp(-center_distance * 5.0)
        # Add some variation
        amp = base_amp * np.random.uniform(0.5, 1.5)
        
        # Cap amplitude to prevent extreme values
        amp = min(1.0, amp)
        
        peak_amplitudes.append(amp)

    # Create Gaussian peaks with optimized widths
    for i, (pos, amp) in enumerate(zip(peak_positions, peak_amplitudes)):
        # Vary widths to create more complex profile
        # Wider at center, narrower at edges
        center_distance = abs(pos)
        base_sigma = 0.02 + 0.03 * np.exp(-center_distance * 3.0)  
        sigma = np.clip(base_sigma * np.random.uniform(0.7, 1.3), 0.005, 0.08)
        
        gaussian = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
        f += gaussian

    # Ensure non-negativity
    f = np.maximum(f, 0.0)

    # Apply better smoothing with Gaussian filter for more natural transitions
    # Use adaptive width based on function characteristics
    if n_points > 100:
        smooth_width = max(1, int(n_points * 0.01))
        f = gaussian_filter1d(f, smooth_width, mode='nearest')

    # Normalize to prevent extremely large values that cause numerical issues
    max_val = np.max(f)
    if max_val > 0:
        f /= (max_val * 1.5)  # Scale down gently for stability

    # Final check for any remaining negative values
    f = np.maximum(f, 0.0)

    return f.tolist()

def construct_function() -> list[float]:
    """
    Construct step-function with high C2 value using improved adaptive Gaussian method.
    """
    # Try multiple attempts to get a good function
    best_c2 = -1
    best_f = None
    num_attempts = 10

    start_time = time.time()
    for attempt in range(num_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break
            
        try:
            # Generate function using adaptive Gaussian construction
            f_values = adaptive_gaussian_construction()

            # Calculate norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 0 or norm_inf <= 0:
                continue

            # Compute C2
            c2 = norm_2_sq / (norm_1 * norm_inf)

            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values.copy()

        except Exception as e:
            warnings.warn(f"Attempt {attempt} failed with error: {str(e)}")
            continue

    # If no successful attempts, fallback to a simpler construction
    if best_f is None:
        # Fallback to original if nothing works reliably
        n_points = np.random.randint(500, 2000)
        return [np.random.random() * 0.5 for _ in range(n_points)]
        
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")