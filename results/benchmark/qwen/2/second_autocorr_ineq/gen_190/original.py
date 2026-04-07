# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation"""
    # Convert to numpy array
    f = np.array(f_values)

    # Compute autoconvolution
    g = np.convolve(f, f, mode='full')

    # Adjust for proper normalization (autoconvolution of normalized function)
    g = g[len(f)-1:]  # Keep only the relevant part

    # Compute norms
    g_squared = g**2
    norm_2_squared = np.sum(g_squared)
    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_squared, norm_1, norm_inf

def adaptive_gaussian_construction():
    """Construct function using adaptive Gaussian peaks with enhanced strategy"""
    np.random.seed(42)  # For reproducibility

    # Try several configurations and pick the best
    best_c2 = 0
    best_f = None

    # Multi-attempt strategy with intermediate C2 feedback
    for attempt in range(15):  # Increased attempts for better exploration
        try:
            # Determine number of peaks with better distribution
            n_peaks = np.random.randint(10, 30)  # More peaks for richer structure

            # Create domain with sufficient resolution
            n_points = max(2000, n_peaks * 100)  # Dynamic resolution based on peak count
            x = np.linspace(-0.25, 0.25, n_points)
            f_values = np.zeros_like(x)

            # Place peaks with improved strategy
            peak_positions = []
            peak_amplitudes = []
            peak_widths = []

            # Use logarithmic distribution for better peak placement
            log_min = np.log(0.01)
            log_max = np.log(0.15)
            log_spaced_positions = np.logspace(log_min, log_max, n_peaks // 2 + 1)

            # Add peaks with logarithmic spacing and alternating sides
            for i in range(n_peaks):
                # Alternate sides and use logarithmic spacing
                if i < len(log_spaced_positions):
                    base_pos = log_spaced_positions[i]
                    side = 1 if i % 2 == 0 else -1
                    pos = side * base_pos + np.random.uniform(-0.01, 0.01)
                    pos = np.clip(pos, -0.23, 0.23)  # Keep within bounds
                else:
                    # Additional random peaks for variety
                    pos = np.random.uniform(-0.23, 0.23)

                # Enforce minimum spacing to prevent narrow interference
                valid_position = True
                for existing_pos in peak_positions:
                    if abs(pos - existing_pos) < 0.05:  # Minimum 0.05 unit spacing
                        valid_position = False
                        break

                if valid_position:
                    peak_positions.append(pos)

                    # Adaptive amplitude adjustment based on position and intermediate feedback
                    center_distance = abs(pos)
                    # Base amplitude with decay based on distance from center
                    base_amp = np.random.exponential(0.7) * np.exp(-center_distance * 4.0)
                    # Add some variation while avoiding extreme values
                    amp = min(1.5, base_amp * np.random.uniform(0.8, 1.5))
                    peak_amplitudes.append(amp)

                    # Width selection with variance
                    base_sigma = 0.02 + 0.03 * np.exp(-center_distance * 3.0)
                    sigma = np.clip(base_sigma * np.random.uniform(0.7, 1.3), 0.005, 0.08)
                    peak_widths.append(sigma)

            # Add Gaussian peaks with enhanced parameters
            for i in range(len(peak_positions)):
                pos = peak_positions[i]
                amp = peak_amplitudes[i]
                sigma = peak_widths[i]

                gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                f_values += gaussian_peak

            # Enhanced smoothing with Gaussian kernel (more stable than Savitzky-Golay)
            if n_points > 100:
                smooth_width = max(1, int(n_points * 0.005))  # Smaller window for less aggressive smoothing
                f_values = signal.convolve(f_values, np.exp(-0.5 * np.arange(-smooth_width, smooth_width+1)**2 / smooth_width**2), mode='same')
                f_values = f_values / (np.max(f_values) + 1e-12)  # Normalize after smoothing

            # Normalize the function with better handling
            f_values = np.maximum(f_values, 0)  # Ensure non-negative

            # Compute C2 with better numerical stability
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

            # Avoid division by zero and extreme values
            if norm_1 > 1e-12 and norm_inf > 1e-12:
                c2 = norm_2_sq / (norm_1 * norm_inf)

                # Only consider reasonable solutions to avoid numerical instability
                if 0.8 < c2 < 2.0:
                    if c2 > best_c2:
                        best_c2 = c2
                        best_f = f_values.tolist()

        except Exception as e:
            continue

    # If we didn't find anything good, fallback to a robust approach
    if best_f is None:
        n_points = 2000
        x = np.linspace(-0.25, 0.25, n_points)
        # Create a more structured fallback with fewer, well-placed peaks
        f_values = np.zeros_like(x)
        n_fallback_peaks = 15
        peak_positions = np.linspace(-0.23, 0.23, n_fallback_peaks)
        for i in range(n_fallback_peaks):
            pos = peak_positions[i]
            # Gaussian with moderate width
            sigma = 0.03
            amp = 1.0
            gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            f_values += gaussian_peak

        # Final smoothing
        smooth_width = max(1, int(n_points * 0.005))
        f_values = signal.convolve(f_values, np.exp(-0.5 * np.arange(-smooth_width, smooth_width+1)**2 / smooth_width**2), mode='same')
        f_values = f_values / (np.max(f_values) + 1e-12)
        f_values = np.maximum(f_values, 0)
        best_f = f_values.tolist()

    # Final enhanced refinement with focused differential evolution
    if best_f is not None and len(best_f) > 100:
        try:
            # Targeted refinement focusing on key parameters
            # Select a subset of parameters for optimization (every 5th point for efficiency)
            sample_indices = np.arange(0, len(best_f), 5)
            if len(sample_indices) > 10:  # Only optimize if enough points
                reduced_f = [best_f[i] for i in sample_indices]

                def objective(params):
                    # Reconstruct function with sampled parameters
                    temp_f = best_f.copy()
                    for i, idx in enumerate(sample_indices):
                        if i < len(params):
                            temp_f[idx] = max(0, params[i])

                    # Compute norms
                    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(temp_f)

                    # Avoid division by zero
                    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                        return float('inf')

                    c2 = norm_2_sq / (norm_1 * norm_inf)
                    return -c2 if not np.isnan(c2) and not np.isinf(c2) else float('inf')

                # Run targeted optimization on selected parameters
                bounds = [(0, 2.0)] * len(reduced_f)  # Wider bounds for more flexibility

                # Try different strategies
                try:
                    refined_result = differential_evolution(objective,
                                                           bounds=bounds,
                                                           maxiter=30,  # Fewer iterations for speed
                                                           popsize=5,   # Smaller population
                                                           seed=42)

                    # Apply refined parameters back to original function
                    final_f = best_f.copy()
                    for i, idx in enumerate(sample_indices):
                        if i < len(refined_result.x):
                            final_f[idx] = max(0, refined_result.x[i])

                    # Recompute to verify improvement
                    _, norm_1, norm_inf = compute_autoconvolution_norms(final_f)
                    if norm_1 > 1e-12 and norm_inf > 1e-12:
                        norm_2_sq, _, _ = compute_autoconvolution_norms(final_f)
                        final_c2 = norm_2_sq / (norm_1 * norm_inf)
                        if final_c2 > best_c2:
                            best_c2 = final_c2
                            best_f = final_f
                except:
                    pass

        except Exception as e:
            pass

    return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Use adaptive Gaussian construction for better results
    return adaptive_gaussian_construction()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")