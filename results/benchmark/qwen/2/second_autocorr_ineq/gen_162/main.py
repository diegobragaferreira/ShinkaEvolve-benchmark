# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from scipy import signal
import math

def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
    """Compute the autoconvolution g = f * f of step function f."""
    n = len(f_values)
    if n == 0:
        return np.array([])

    f_array = np.array(f_values)

    # Compute convolution using numpy's convolve (valid mode)
    g = np.convolve(f_array, f_array, mode='full')

    # Trim to appropriate size (should be 2*n-1 elements)
    g = g[n-1:-(n-1)] if n > 1 else g

    return g

def compute_norms(g_values: np.ndarray) -> tuple:
    """Compute the three required norms for C2 calculation."""
    if len(g_values) == 0:
        return 0.0, 0.0, 0.0

    # ||g||₂² using trapezoidal-like piecewise linear integration with proper spacing
    if len(g_values) <= 1:
        norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
    else:
        # Properly compute using trapezoidal-like integration with correct width
        # The convolution output corresponds to a spacing that needs to be accounted for
        norm_2_sq = 0.0
        # For consecutive points y_i and y_{i+1}, we integrate over width 1
        # The contribution is (1/3)(y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
        for i in range(len(g_values)-1):
            y1, y2 = g_values[i], g_values[i+1]
            norm_2_sq += (1.0/3.0) * (y1**2 + y1*y2 + y2**2)

    # ||g||₁: L1 norm, approximated as sum(|g|) / (len(g) + 1)
    if len(g_values) > 0:
        norm_1 = np.sum(np.abs(g_values)) / (len(g_values) + 1)
    else:
        norm_1 = 0.0

    # ||g||∞: Infinity-norm
    norm_inf = np.max(np.abs(g_values)) if len(g_values) > 0 else 0.0

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 for given step function values."""
    g = compute_autoconvolution(f_values)
    norm_2_sq, norm_1, norm_inf = compute_norms(g)

    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def adaptive_gaussian_construction(n_steps: int = 1000) -> List[float]:
    """
    Construct step function using adaptive Gaussian peak building approach.
    This method builds peaks strategically to maximize C2 while maintaining
    good numerical properties.
    """
    # Domain setup
    domain_width = 0.5
    domain_center = 0.0
    step_width = domain_width / n_steps

    # Start with a few well-placed Gaussian peaks
    # Using logarithmic spacing to distribute peaks effectively
    n_peaks = 5  # Start with 5 peaks

    # Initialize peaks with positions based on logarithmic distribution
    # to avoid clustering at edges
    peak_positions = []
    for i in range(n_peaks):
        # Logarithmic distribution to get more points near center and fewer at edges
        # Map from 0-1 to -0.25 to 0.25 with logarithmic spacing
        ratio = (i + 1) / (n_peaks + 1)
        # Apply exponential mapping to concentrate points near center
        pos = domain_center + (ratio ** 1.5) * domain_width/2
        if i % 2 == 0:
            pos = -pos  # Alternate sides to balance
        peak_positions.append(pos)

    # Initialize peak parameters [amplitude, center, width]
    peak_params = []
    for i in range(n_peaks):
        amplitude = 50.0
        center = peak_positions[i]
        width = 0.05  # Initial width
        peak_params.extend([amplitude, center, width])

    # Build the function progressively
    domain_points = np.linspace(-domain_width/2, domain_width/2, n_steps)

    # Helper function to generate Gaussian peaks
    def gaussian_peak_function(x: np.ndarray, params: List[float]) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, center, width = params[i], params[i+1], params[i+2]
            width = max(width, 1e-6)  # Ensure positive width
            result += amp * np.exp(-0.5 * ((x - center) / width)**2)
        return result

    # Local search refinement function
    def local_search(current_params, domain_points, target_c2):
        """
        Apply local search to improve parameters by finding better neighbors
        """
        # Try several local modifications to find improvements
        best_params = list(current_params)
        best_c2 = target_c2

        # Make multiple small modifications to find local optima
        for _ in range(50):  # Try up to 50 local changes
            test_params = list(current_params)

            # Choose which parameter to modify
            param_idx = np.random.randint(len(test_params))

            # Apply different types of modifications based on parameter type
            if param_idx % 3 == 0:  # amplitude
                # Multiply by factor near 1.0
                factor = 1.0 + np.random.uniform(-0.1, 0.1)
                test_params[param_idx] *= factor
            elif param_idx % 3 == 1:  # center
                # Small shift
                test_params[param_idx] += np.random.uniform(-0.01, 0.01)
            else:  # width
                # Multiply by factor
                factor = 1.0 + np.random.uniform(-0.1, 0.1)
                test_params[param_idx] = max(0.001, test_params[param_idx] * factor)

            # Ensure non-negative
            test_params[param_idx] = max(0, test_params[param_idx])

            # Apply spacing constraint
            enforce_peak_spacing(test_params, domain_width)

            # Evaluate
            func_values = gaussian_peak_function(domain_points, test_params)
            test_c2 = compute_c2(func_values.tolist())

            if test_c2 > best_c2:
                best_c2 = test_c2
                best_params = list(test_params)

        return best_params, best_c2

    # Adaptive optimization loop
    best_c2 = -1.0
    best_params = list(peak_params)
    best_function = None

    max_iterations = 500
    for iteration in range(max_iterations):
        # Create function from current peak parameters
        func_values = gaussian_peak_function(domain_points, peak_params)

        # Convert to step function by taking samples
        step_values = func_values.tolist()

        # Compute C2
        c2_val = compute_c2(step_values)

        # Check if this is our best result so far
        if c2_val > best_c2:
            best_c2 = c2_val
            best_params = list(peak_params)
            best_function = step_values.copy()

        # Stop if we're getting close to convergence
        if iteration > 10 and abs(c2_val - best_c2) < 1e-8:
            break

        # Apply adaptive adjustments to peak parameters
        # Only adjust the parameters that are most likely to improve C2
        for i in range(0, len(peak_params), 3):
            # Perturb amplitude slightly
            if i < len(peak_params):  # amplitude
                old_amp = peak_params[i]
                # Small random change with bias toward increase for C2 maximization
                change_factor = 1.0 + np.random.normal(0, 0.1)
                new_amp = max(0, old_amp * change_factor)
                peak_params[i] = new_amp

            # Perturb width slightly
            if i+2 < len(peak_params):  # width
                old_width = peak_params[i+2]
                # Change width by a factor
                change_factor = 1.0 + np.random.normal(0, 0.1)
                new_width = max(0.001, old_width * change_factor)
                peak_params[i+2] = new_width

        # Enforce minimum spacing between peaks to avoid narrow autoconvolution
        # This helps maintain a flatter autoconvolution profile which improves C2
        enforce_peak_spacing(peak_params, domain_width)

        # Apply local search refinement periodically
        if iteration % 30 == 0 and iteration > 0:
            refined_params, refined_c2 = local_search(peak_params, domain_points, c2_val)
            if refined_c2 > c2_val:
                peak_params = list(refined_params)
                c2_val = refined_c2

        # If we're approaching a plateau, make larger adjustments
        if iteration > 50 and iteration % 20 == 0:
            # Occasionally do larger changes to explore more space
            for i in range(0, len(peak_params), 3):
                if i < len(peak_params):  # amplitude
                    change_factor = 1.0 + np.random.normal(0, 0.2)  # Larger change
                    new_amp = max(0, peak_params[i] * change_factor)
                    peak_params[i] = new_amp

        # Occasionally reduce amplitudes if we see signs of growing ||g||_∞ too much
        # This prevents overly sharp peaks that hurt C2
        if iteration > 20 and np.random.rand() < 0.1:
            # Check if we're getting very peaked g (high ||g||_∞)
            func_values = gaussian_peak_function(domain_points, peak_params)
            g = compute_autoconvolution(func_values.tolist())
            if len(g) > 0:
                norm_inf = np.max(np.abs(g))
                if norm_inf > 200:  # If autoconvolution is too peaked
                    # Reduce all peak amplitudes moderately
                    for j in range(0, len(peak_params), 3):
                        if j < len(peak_params):
                            peak_params[j] *= 0.95

    # Final refinement - use the best parameters found
    final_func_values = gaussian_peak_function(domain_points, best_params)
    return final_func_values.tolist()

def enforce_peak_spacing(peak_params: List[float], domain_width: float, min_distance_ratio: float = 0.1):
    """
    Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution.
    """
    if len(peak_params) < 3:
        return

    # Group peaks by their parameters [amp, center, width]
    peaks = []
    for i in range(0, len(peak_params), 3):
        peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

    # Sort by center position
    peaks.sort(key=lambda x: x[1])

    # Ensure minimum spacing
    min_distance = min_distance_ratio * domain_width
    for i in range(1, len(peaks)):
        prev_center = peaks[i-1][1]
        curr_center = peaks[i][1]
        distance = abs(curr_center - prev_center)

        if distance < min_distance:
            # Adjust position of current peak
            # Move it away from the previous one
            offset = min_distance - distance
            if curr_center > prev_center:
                peaks[i][1] += offset
            else:
                peaks[i][1] -= offset

    # Put them back into flat list
    for i, (amp, center, width) in enumerate(peaks):
        peak_params[i*3] = amp
        peak_params[i*3 + 1] = center
        peak_params[i*3 + 2] = width

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses adaptive Gaussian peak construction.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Try different strategies
    best_c2 = 0.0
    best_function = []

    # Strategy 1: Adaptive Gaussian construction with various sizes
    try:
        # Try different step counts to find optimal resolution
        step_counts = [1000, 1500, 2000, 2500]
        for n_steps in step_counts:
            func = adaptive_gaussian_construction(n_steps)
            c2_val = compute_c2(func)

            if c2_val > best_c2:
                best_c2 = c2_val
                best_function = func
    except Exception as e:
        pass

    # If nothing worked, fallback to a simple uniform distribution
    if len(best_function) == 0:
        # Simple uniform function that often works well
        best_function = [10.0] * 500
        best_c2 = compute_c2(best_function)

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")