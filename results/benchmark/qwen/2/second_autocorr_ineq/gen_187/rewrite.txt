# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from scipy import signal
from scipy.optimize import minimize
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

def gaussian_peak_function(x: np.ndarray, peak_params: List[float]) -> np.ndarray:
    """Generate a function composed of multiple Gaussian peaks."""
    result = np.zeros_like(x)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        width = max(width, 1e-6)  # Ensure positive width
        result += amp * np.exp(-0.5 * ((x - center) / width)**2)
    return result

def chebyshev_nodes(n: int, a: float = -0.25, b: float = 0.25) -> np.ndarray:
    """Generate Chebyshev nodes for better distribution."""
    if n <= 0:
        return np.array([])
    
    # Chebyshev nodes in [-1, 1], then map to [a, b]
    k = np.arange(1, n + 1)
    theta = (2*k - 1) * np.pi / (2*n)
    nodes = np.cos(theta)
    
    # Map from [-1, 1] to [a, b]
    return 0.5 * (b - a) * nodes + 0.5 * (b + a)

def enforce_peak_spacing(peak_params: List[float], domain_width: float, min_distance_ratio: float = 0.08):
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

def adaptive_gaussian_quadratic(n_steps: int = 1000) -> List[float]:
    """
    Hybrid approach combining adaptive Gaussian construction with quadratic optimization principles.
    """
    # Domain setup
    domain_width = 0.5
    domain_center = 0.0

    # Use Chebyshev nodes for optimal peak distribution (from quadratic optimization)
    n_peaks = 7  # Based on empirical analysis
    
    # Generate Chebyshev nodes for peak positions
    peak_positions = chebyshev_nodes(n_peaks)
    
    # Alternate sides for better symmetry
    for i in range(len(peak_positions)):
        if i % 2 == 0:
            peak_positions[i] = -peak_positions[i]
    
    # Initialize peak parameters with adaptive amplitudes
    peak_params = []
    for i, center in enumerate(peak_positions):
        # Adaptive amplitudes based on position (higher at center, lower at edges)
        amplitude = 50.0 + 30.0 * (1 - abs(center) / (domain_width/2))
        amplitude = max(10.0, amplitude)  # Minimum amplitude
        
        # Widths that are inversely proportional to amplitude for balance
        width = 0.03 + 0.02 * (1 - amplitude/80.0)
        
        peak_params.extend([amplitude, center, width])

    # Domain points for evaluation
    domain_points = np.linspace(-domain_width/2, domain_width/2, n_steps)
    
    # Create function that computes C2 for given parameters
    def objective(params):
        """Compute negative C2 (since we want to maximize C2)"""
        # Extract parameters
        amps = params[::3]
        centers = params[1::3] 
        widths = params[2::3]
        
        # Reconstruct parameter list
        peak_list = []
        for i in range(len(amps)):
            peak_list.extend([amps[i], centers[i], widths[i]])
        
        # Generate function
        func_values = gaussian_peak_function(domain_points, peak_list)
        step_values = func_values.tolist()
        
        # Compute C2
        c2_val = compute_c2(step_values)
        
        # Return negative because we minimize
        return -c2_val
    
    # Simple optimization using scipy minimize with bounds
    bounds = []
    for i in range(n_peaks):
        # Amplitude bounds: 10 to 100
        bounds.append((10.0, 100.0))
        # Center bounds: [-0.25, 0.25]
        bounds.append((-0.25, 0.25))
        # Width bounds: 0.01 to 0.2
        bounds.append((0.01, 0.2))
    
    # Try optimization with different methods
    try:
        result = minimize(objective, peak_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 100})
        if result.success:
            # Use optimized parameters
            optimized_params = result.x
            peak_params = optimized_params.tolist()
    except:
        # Fall back to simple refinement if optimization fails
        pass
    
    # Final function generation with peak spacing enforcement
    func_values = gaussian_peak_function(domain_points, peak_params)
    step_values = func_values.tolist()
    
    # Apply final post-processing to ensure good quality
    # Ensure non-negative, smooth the function slightly
    step_values = [max(0, x) for x in step_values]
    
    # Smooth a bit by averaging with neighbors
    smoothed = []
    window = 5
    for i in range(len(step_values)):
        start_idx = max(0, i - window//2)
        end_idx = min(len(step_values), i + window//2 + 1)
        avg_val = np.mean(step_values[start_idx:end_idx])
        smoothed.append(avg_val)
    
    step_values = smoothed
    
    return step_values

def local_search_refinement(current_params: List[float], domain_points: np.ndarray, 
                           domain_width: float, initial_c2: float) -> tuple:
    """
    Enhanced local search for parameter refinement
    """
    best_params = list(current_params)
    best_c2 = initial_c2

    # Try several local modifications to find improvements
    for _ in range(30):  # Reduced iterations but more targeted
        test_params = list(current_params)

        # Choose which parameter to modify
        param_idx = np.random.randint(len(test_params))

        # Apply different types of modifications based on parameter type
        if param_idx % 3 == 0:  # amplitude
            # Apply multiplicative change with adaptive factor
            factor = 1.0 + np.random.normal(0, 0.1)
            test_params[param_idx] *= factor
        elif param_idx % 3 == 1:  # center
            # Small shift with higher probability for small changes
            test_params[param_idx] += np.random.normal(0, 0.005)
        else:  # width
            # Multiplicative change
            factor = 1.0 + np.random.normal(0, 0.1)
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

def adaptive_gaussian_construction_advanced(n_steps: int = 1000) -> List[float]:
    """
    Advanced adaptive Gaussian construction with multiple refinement strategies
    """
    # Domain setup
    domain_width = 0.5
    domain_center = 0.0

    # Start with a few well-placed Gaussian peaks using Chebyshev distribution
    n_peaks = 7

    # Generate Chebyshev nodes for peak positions
    peak_positions = chebyshev_nodes(n_peaks)
    
    # Alternate sides for better symmetry
    for i in range(len(peak_positions)):
        if i % 2 == 0:
            peak_positions[i] = -peak_positions[i]
    
    # Initialize peak parameters with adaptive amplitudes
    peak_params = []
    for i, center in enumerate(peak_positions):
        # Adaptive amplitudes based on position
        amplitude = 50.0 + 30.0 * (1 - abs(center) / (domain_width/2))
        amplitude = max(10.0, amplitude)
        
        # Widths that are inversely proportional to amplitude for balance
        width = 0.03 + 0.02 * (1 - amplitude/80.0)
        
        peak_params.extend([amplitude, center, width])

    # Build the function progressively
    domain_points = np.linspace(-domain_width/2, domain_width/2, n_steps)

    # Adaptive optimization loop with hybrid strategy
    best_c2 = -1.0
    best_params = list(peak_params)
    best_function = None

    max_iterations = 300  # Reduced iterations for time efficiency
    
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
        # Apply different strategies for different parameter types
        for i in range(0, len(peak_params), 3):
            # Perturb amplitude slightly with bias toward increasing
            if i < len(peak_params):  # amplitude
                old_amp = peak_params[i]
                # Bias toward increase for C2 maximization
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

        # Enforce minimum spacing between peaks
        enforce_peak_spacing(peak_params, domain_width)

        # Apply local search refinement periodically
        if iteration % 20 == 0 and iteration > 0:
            refined_params, refined_c2 = local_search_refinement(
                peak_params, domain_points, domain_width, c2_val
            )
            if refined_c2 > c2_val:
                peak_params = list(refined_params)
                c2_val = refined_c2

        # Occasionally do larger changes to explore more space
        if iteration > 50 and iteration % 25 == 0:
            for i in range(0, len(peak_params), 3):
                if i < len(peak_params):  # amplitude
                    change_factor = 1.0 + np.random.normal(0, 0.15)  # Larger change
                    new_amp = max(0, peak_params[i] * change_factor)
                    peak_params[i] = new_amp

        # Occasionally reduce amplitudes if autoconvolution is too peaked
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
                            peak_params[j] *= 0.97

    # Final refinement - use the best parameters found
    final_func_values = gaussian_peak_function(domain_points, best_params)
    return final_func_values.tolist()

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses hybrid optimization approach.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Try different strategies
    best_c2 = 0.0
    best_function = []

    # Strategy 1: Quadratic optimization approach with Chebyshev nodes
    try:
        result = adaptive_gaussian_quadratic(1000)
        c2_val = compute_c2(result)
        
        if c2_val > best_c2:
            best_c2 = c2_val
            best_function = result
    except Exception as e:
        pass

    # Strategy 2: Advanced adaptive Gaussian construction
    if len(best_function) == 0:
        try:
            result = adaptive_gaussian_construction_advanced(1000)
            c2_val = compute_c2(result)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_function = result
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