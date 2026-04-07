# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import time

def compute_autoconvolution_norms(f_values):
    """Compute the three norms for autoconvolution g = f*f"""
    if len(f_values) < 2:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution
    g = signal.convolve(f_values, f_values, mode='full')

    # Get the central portion corresponding to the actual convolution support
    # For two functions on [-1/4, 1/4], their convolution supports [-1/2, 1/2]
    # With n points each, we get 2*n-1 points in convolution
    center_start = len(g) // 2
    center_end = center_start + len(f_values) * 2 - 1
    g_center = g[center_start:center_end]

    # Compute the three norms correctly
    # ||g||₂² computed via trapezoidal integration of g²
    g_squared = g_center**2
    norm_2_squared = np.sum(g_squared)

    # ||g||₁ = sum(|g|) * dx where dx is the step size
    dx = 0.5 / (len(f_values) - 1) if len(f_values) > 1 else 0.5
    norm_1 = np.sum(np.abs(g_center)) * dx

    # ||g||∞ = max(|g|)
    norm_inf = np.max(np.abs(g_center)) if len(g_center) > 0 else 0.0

    return norm_2_squared, norm_1, norm_inf

def compute_c2(f_values):
    """Compute C2 value for given step function"""
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0

    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def adaptive_gaussian_construction(n_points=None):
    """Construct step function using adaptive Gaussian shaping with logarithmic peak spacing"""
    if n_points is None:
        n_points = np.random.randint(100, 5000)

    # Create a smooth Gaussian-like structure with controlled amplitude
    # This builds upon the insight that flatter autoconvolution profiles
    # lead to higher C₂ values

    # Generate x coordinates from -1/4 to 1/4
    x = np.linspace(-0.25, 0.25, n_points)

    # Create multiple Gaussian peaks with varying positions and amplitudes
    # This helps avoid local optima while promoting beneficial autoconvolution shapes
    f_values = np.zeros(n_points)

    # Add several peaks at strategically spaced locations using logarithmic distribution
    # This prevents clustering and ensures uniform coverage of the domain
    num_peaks = np.random.randint(3, 8)
    peak_width = 0.05  # Fixed peak width for consistency

    # Use logarithmic spacing to distribute peaks more evenly
    # Generate logarithmically spaced distances from center
    log_min = np.log(0.01)
    log_max = np.log(0.2)  # Max distance from center (0.25 but leave some margin)
    log_spaced_distances = np.logspace(log_min, log_max, num_peaks)

    # Distribute peaks around center with some randomness to avoid perfect symmetry
    peak_positions = []
    for i, dist in enumerate(log_spaced_distances):
        # Alternate sides and add some randomness
        side = (-1)**i  # Alternate left/right
        base_pos = side * dist * 0.5  # Scale to fit within domain
        # Add slight variation to avoid perfect symmetry
        peak_pos = base_pos + np.random.uniform(-0.01, 0.01)
        peak_positions.append(peak_pos)

    for peak_position in peak_positions:
        # Adjust amplitude based on position to avoid overly sharp autoconvolutions
        center_distance = abs(peak_position)
        if center_distance < 0.05:  # Near center
            amplitude_factor = 1.5
        elif center_distance < 0.1:  # Middle area
            amplitude_factor = 1.2
        else:  # Outer areas
            amplitude_factor = 0.8

        peak_amplitude = np.random.uniform(0.8, 1.5) * amplitude_factor

        # Create Gaussian peak
        gaussian_peak = peak_amplitude * np.exp(-0.5 * ((x - peak_position) / peak_width)**2)
        f_values += gaussian_peak

    # Normalize to control overall magnitude
    f_values = np.clip(f_values, 0, None)  # Ensure non-negative

    # Apply additional smoothing to make profile less sharp
    if len(f_values) > 10:
        # Simple moving average smoothing
        window_size = min(5, len(f_values) // 10)
        if window_size > 1:
            smoothed = np.convolve(f_values, np.ones(window_size)/window_size, mode='same')
            f_values = smoothed

    return f_values.tolist()

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Try several constructions and pick the best one
    best_c2 = 0.0
    best_f_values = None
    start_time = time.time()

    # Try multiple constructions with timeout
    for _ in range(100):  # Limit iterations to keep within time budget
        if time.time() - start_time > 85:  # Leave some margin
            break

        # Construct using adaptive Gaussian method
        try:
            f_values = adaptive_gaussian_construction()
            c2 = compute_c2(f_values)

            if c2 > best_c2:
                best_c2 = c2
                best_f_values = f_values
        except:
            continue

    # If we couldn't construct anything decent, fall back to random
    if best_c2 < 0.1 or best_f_values is None:
        n_points = np.random.randint(100, 1000)
        best_f_values = [np.random.random() for _ in range(n_points)]

    return best_f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")