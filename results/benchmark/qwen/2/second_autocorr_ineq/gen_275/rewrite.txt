# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import warnings
import random
from numba import jit

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation with improved numerical stability"""
    # Convert to numpy array
    f = np.array(f_values, dtype=np.float64)

    if len(f) < 1:
        return 0.0, 1e-12, 1e-12

    # Compute autoconvolution using fast numba implementation
    g = compute_autoconvolution_fast(f)
    
    # Compute norms using piecewise integration
    norm_2_squared, norm_1, norm_inf = compute_norms_piecewise(g)

    # Numerical stability checks
    if np.isnan(norm_2_squared) or np.isinf(norm_2_squared):
        norm_2_squared = 0.0
    if np.isnan(norm_1) or np.isinf(norm_1) or norm_1 <= 0:
        norm_1 = 1e-12
    if np.isnan(norm_inf) or np.isinf(norm_inf) or norm_inf <= 0:
        norm_inf = 1e-12

    return norm_2_squared, norm_1, norm_inf

def harmonic_peak_construction():
    """Construct function using harmonic peak patterns optimized for autoconvolution"""
    np.random.seed(42)  # For reproducibility

    # Parameters for better peak distribution
    min_peaks = 12
    max_peaks = 40
    n_points = 3000  # Increase for better resolution

    # Try several configurations and pick the best
    best_c2 = 0
    best_f = None

    # Generate multiple candidate functions with different parameters
    for attempt in range(25):  # Increase attempts for better exploration
        try:
            # Determine number of peaks (between min_peaks and max_peaks)
            n_peaks = np.random.randint(min_peaks, max_peaks + 1)

            # Create domain
            x = np.linspace(-0.25, 0.25, n_points)
            f_values = np.zeros_like(x)

            # Place peaks with logarithmic spacing to prevent clustering
            peak_positions = []
            peak_amplitudes = []
            peak_widths = []

            # Use logarithmic distribution of peak positions
            # This helps avoid clustering around center and edges
            log_min = np.log(0.015)
            log_max = np.log(0.15)
            log_spaced_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)

            # Place peaks on both sides of center
            for i in range(n_peaks):
                # Alternate sides to distribute evenly
                side = 1 if i % 2 == 0 else -1
                if i < len(log_spaced_positions):
                    pos = side * log_spaced_positions[i // 2] + np.random.uniform(-0.008, 0.008)
                else:
                    # Fill remaining peaks with uniform distribution
                    pos = np.random.uniform(-0.23, 0.23)

                # Clip to valid range
                pos = np.clip(pos, -0.23, 0.23)

                # Avoid too close proximity
                valid_position = True
                for existing_pos in peak_positions:
                    if abs(pos - existing_pos) < 0.015:  # Minimum gap of 0.015
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
                base_amp = np.random.exponential(0.6) * np.exp(-center_distance * 4.0)
                # Add some variation
                amp = base_amp * np.random.uniform(0.6, 1.4)

                # Cap amplitude to prevent extreme values
                amp = min(1.2, amp)

                peak_amplitudes.append(amp)

            # Create Gaussian peaks with optimized widths
            for i, (pos, amp) in enumerate(zip(peak_positions, peak_amplitudes)):
                # Vary widths to create more complex profile
                # Wider at center, narrower at edges
                center_distance = abs(pos)
                base_sigma = 0.015 + 0.035 * np.exp(-center_distance * 2.5)
                sigma = np.clip(base_sigma * np.random.uniform(0.75, 1.25), 0.003, 0.09)

                gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                f_values += gaussian_peak

            # Apply better smoothing with adaptive width
            if n_points > 100:
                smooth_width = max(1, int(n_points * 0.008))
                f_values = gaussian_filter1d(f_values, smooth_width, mode='nearest')

            # Ensure non-negativity
            f_values = np.maximum(f_values, 0.0)

            # Normalize for better numerical behavior
            max_val = np.max(f_values)
            if max_val > 0:
                f_values /= (max_val * 1.2)  # Scale down gently for stability

            # Final check for any remaining negative values
            f_values = np.maximum(f_values, 0.0)

            # Compute C2
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

            # Avoid division by zero or extreme values
            if norm_1 > 1e-12 and norm_inf > 1e-12:
                c2 = norm_2_sq / (norm_1 * norm_inf)

                # Only consider reasonable solutions
                if 0.75 < c2 < 2.0:
                    if c2 > best_c2:
                        best_c2 = c2
                        best_f = f_values.tolist()

        except Exception as e:
            warnings.warn(f"Attempt {attempt} failed with error: {str(e)}")
            continue

    # If we didn't find anything good, fallback to a basic approach
    if best_f is None:
        # Fallback to a slightly more structured approach with more peaks
        n_points = 2000
        x = np.linspace(-0.25, 0.25, n_points)
        # Create a more complex bell-shaped function with multiple components
        f_values = np.zeros_like(x)
        # Add multiple harmonic components
        for i in range(8):
            center = np.random.uniform(-0.2, 0.2)
            sigma = np.random.uniform(0.01, 0.03)
            height = np.random.uniform(0.5, 1.5)
            gaussian = height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
            f_values += gaussian
        # Normalize
        f_values = f_values / (np.max(f_values) * 1.5)
        # Ensure non-negativity
        f_values = np.maximum(f_values, 0.0)
        best_f = f_values.tolist()

    # Final refinement with local optimization
    try:
        # Use only the best function for refinement if we have one
        if best_f is not None:
            # Simple gradient-free optimization to refine the best found solution
            def objective(params):
                # Re-compute with refined parameters
                norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(params)
                if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                    return 0.0
                c2 = norm_2_sq / (norm_1 * norm_inf)
                return -c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0

            # Use a more efficient optimization approach with fewer iterations
            # Only optimize every 3rd point for computational efficiency
            sample_indices = np.arange(0, len(best_f), 3)
            if len(sample_indices) > 10:
                reduced_bounds = [(0, 1.5)] * len(sample_indices)
                reduced_params = [best_f[i] for i in sample_indices]
                
                # Run differential evolution on reduced parameters
                refined_result = differential_evolution(objective,
                                                       bounds=reduced_bounds,
                                                       maxiter=25, popsize=5,
                                                       seed=42)
                
                # Apply refined parameters back to original function
                final_f = best_f.copy()
                for i, idx in enumerate(sample_indices):
                    if i < len(refined_result.x):
                        final_f[idx] = max(0, refined_result.x[i])
                
                # Re-compute with refined parameters
                _, norm_1, norm_inf = compute_autoconvolution_norms(final_f)
                if norm_1 > 1e-12 and norm_inf > 1e-12:
                    norm_2_sq, _, _ = compute_autoconvolution_norms(final_f)
                    final_c2 = norm_2_sq / (norm_1 * norm_inf)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
                        best_f = final_f
    except Exception as e:
        warnings.warn(f"Final refinement failed with error: {str(e)}")

    # If still no good solution, return a reasonable fallback
    if best_f is None:
        n_points = 800
        best_f = [np.random.random() * 0.5 for _ in range(n_points)]

    return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Use harmonic peak construction for better results
    return harmonic_peak_construction()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")