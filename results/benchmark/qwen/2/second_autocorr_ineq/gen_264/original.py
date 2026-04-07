# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced adaptive Gaussian optimization."""
    np.random.seed(42)  # For reproducibility

    # Determine number of steps with higher resolution
    n_steps = np.random.randint(1000, 10000)  # Increased range for better resolution

    # Create x-axis points in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, n_steps)

    # Initialize with base multi-peak Gaussian structure
    base_function = np.zeros_like(x)

    # Use logarithmic spacing for peak positions to avoid clustering
    num_peaks = np.random.randint(10, 30)

    # Generate logarithmically spaced peak positions to avoid narrow interference
    log_positions = np.logspace(np.log10(0.01), np.log10(0.24), num_peaks)
    peak_positions = np.concatenate([log_positions, -log_positions[::-1]])
    peak_positions = peak_positions[peak_positions <= 0.25]
    peak_positions = peak_positions[peak_positions >= -0.25]

    # Ensure minimum gap between peaks to prevent narrow autoconvolution
    min_gap = 0.1 * 0.5  # 10% of domain width
    safe_positions = []
    for pos in sorted(peak_positions):
        if not safe_positions or abs(pos - safe_positions[-1]) >= min_gap:
            safe_positions.append(pos)

    num_peaks = len(safe_positions)

    # Construct peaks with optimized parameters
    for i in range(num_peaks):
        # Set peak center
        peak_center = safe_positions[i]

        # Adjust peak height based on position to avoid very sharp peaks
        # Peaks near edges get reduced height to prevent excessive autoconvolution peaks
        if abs(peak_center) > 0.15:
            peak_height = np.random.uniform(1.0, 1.5)
        else:
            peak_height = np.random.uniform(1.2, 2.0)

        # Use narrower widths for central peaks, wider for outer ones
        if abs(peak_center) < 0.05:
            peak_width = np.random.uniform(0.015, 0.03)
        elif abs(peak_center) < 0.15:
            peak_width = np.random.uniform(0.025, 0.05)
        else:
            peak_width = np.random.uniform(0.03, 0.07)

        # Create Gaussian peak
        gaussian_peak = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width)**2)
        base_function += gaussian_peak

    # Add some additional structure with controlled randomness
    # This helps create better autoconvolution properties
    for i in range(0, len(x), max(1, len(x)//20)):  # More frequent bumps
        if np.random.random() > 0.8:  # 20% chance to add small bump
            bump_center = x[i]
            bump_height = np.random.uniform(0.05, 0.3)
            bump_width = np.random.uniform(0.005, 0.015)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            base_function += bump

    # Ensure non-negative values
    base_function = np.maximum(base_function, 0)

    # Normalize to avoid extreme values that might hurt the C2 calculation
    if np.max(base_function) > 0:
        base_function = base_function / np.max(base_function) * 1.5

    # Apply light noise for robustness (but preserve structure)
    noise_level = 0.02  # Reduced noise level
    noisy_function = base_function + np.random.normal(0, noise_level, len(base_function))
    noisy_function = np.maximum(noisy_function, 0)

    # Smooth the function to reduce sharp transitions that could hurt C2
    # Use a simple moving average smoothing
    window_size = max(1, n_steps // 200)  # Dynamic window size
    if window_size % 2 == 0:
        window_size += 1  # Must be odd for symmetric filter
    if window_size > 1:
        smoothed_function = signal.savgol_filter(noisy_function, window_size, 1)
        smoothed_function = np.maximum(smoothed_function, 0)
        noisy_function = smoothed_function

    # Convert to step values ensuring proper format
    step_values = noisy_function.tolist()

    # Phase 2: Improved Local Optimization of Peak Parameters
    def optimize_peaks(initial_func, n_steps):
        # Extract peak information with better peak detection
        x = np.linspace(-0.25, 0.25, n_steps)

        # Better peak detection using second derivative analysis
        peaks = []
        # Look for local maxima with minimum height threshold
        for i in range(1, len(initial_func)-1):
            if (initial_func[i] > initial_func[i-1] and
                initial_func[i] > initial_func[i+1] and
                initial_func[i] > 0.1):  # Only consider significant peaks
                peaks.append((i, initial_func[i]))

        # Take top peaks based on height
        peaks.sort(key=lambda x: x[1], reverse=True)
        selected_peaks = peaks[:min(8, len(peaks))]

        if len(selected_peaks) == 0:
            return np.array(initial_func)

        # Use a hybrid optimization approach
        # First, try a simpler approach with adaptive parameters
        def objective_with_adaptive_bounds(params):
            # Reconstruct function with given params
            temp_func = np.zeros_like(x)
            for i, (pos_idx, height) in enumerate(selected_peaks):
                # Adaptive bounds based on peak position and height
                max_pos_adjust = min(0.05, 0.1 * (1 - abs(x[pos_idx])/0.25))  # Smaller adjustments near edges
                max_height_adjust = 0.3 * (height / np.max([p[1] for p in selected_peaks]))  # Relative adjustment

                center_pos = x[pos_idx] + (params[i*2] - 0.5) * max_pos_adjust
                peak_height = height * (1.0 + params[i*2+1] * max_height_adjust)

                # Determine appropriate width based on position and height
                width = max(0.01, 0.02 + 0.01 * (1 - abs(center_pos)/0.25))
                temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)

            c2 = compute_c2(temp_func)
            return -c2  # Negative because we minimize

        # Try multiple optimization approaches
        best_params = None
        best_c2 = -np.inf
        best_func = np.array(initial_func)

        # Approach 1: Differential Evolution with adaptive settings
        try:
            # Create bounds that are adaptive to peak positions and heights
            bounds = []
            for i, (pos_idx, height) in enumerate(selected_peaks):
                max_pos_adjust = min(0.05, 0.1 * (1 - abs(x[pos_idx])/0.25))
                max_height_adjust = 0.3 * (height / np.max([p[1] for p in selected_peaks]))
                bounds.extend([(-0.5, 0.5)])  # Position adjustment
                bounds.extend([(-0.5, 0.5)])  # Height adjustment

            # Run with more iterations for better convergence
            result = differential_evolution(objective_with_adaptive_bounds,
                                          bounds=bounds,
                                          maxiter=50,
                                          popsize=10,
                                          seed=42,
                                          polish=True)

            # Evaluate the result
            temp_func = np.zeros_like(x)
            for i, (pos_idx, height) in enumerate(selected_peaks):
                max_pos_adjust = min(0.05, 0.1 * (1 - abs(x[pos_idx])/0.25))
                max_height_adjust = 0.3 * (height / np.max([p[1] for p in selected_peaks]))
                center_pos = x[pos_idx] + (result.x[i*2] - 0.5) * max_pos_adjust
                peak_height = height * (1.0 + result.x[i*2+1] * max_height_adjust)
                width = max(0.01, 0.02 + 0.01 * (1 - abs(center_pos)/0.25))
                temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)

            c2_result = compute_c2(temp_func)
            if c2_result > best_c2:
                best_c2 = c2_result
                best_func = temp_func.copy()
                best_params = result.x.copy()

        except Exception:
            pass

        # If no good result from DE, fall back to simple local search with careful bounds
        if best_params is None:
            # Simple simulated annealing approach for peak refinement
            current_params = np.zeros(len(selected_peaks) * 2)
            current_func = np.zeros_like(x)
            for i, (pos_idx, height) in enumerate(selected_peaks):
                center_pos = x[pos_idx]
                peak_height = height
                width = max(0.01, 0.02 + 0.01 * (1 - abs(center_pos)/0.25))
                current_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)

            current_c2 = compute_c2(current_func)
            best_c2 = current_c2
            best_func = current_func.copy()

            # Fine-grained optimization with cooling schedule
            temp = 1.0
            cooling_rate = 0.995
            max_iterations = 100

            for _ in range(max_iterations):
                if temp < 1e-6:
                    break

                # Generate neighbor
                neighbor_params = current_params.copy()
                for i in range(len(neighbor_params)):
                    if np.random.random() < 0.7:  # 70% chance to perturb
                        # Adaptive perturbation based on parameter importance
                        max_adjust = 0.2 if i % 2 == 0 else 0.1  # Position vs height
                        neighbor_params[i] += np.random.normal(0, max_adjust * temp)
                        # Clamp to reasonable bounds
                        if i % 2 == 0:
                            neighbor_params[i] = np.clip(neighbor_params[i], -0.5, 0.5)
                        else:
                            neighbor_params[i] = np.clip(neighbor_params[i], -0.5, 0.5)

                # Calculate new function
                temp_func = np.zeros_like(x)
                for i, (pos_idx, height) in enumerate(selected_peaks):
                    max_pos_adjust = min(0.05, 0.1 * (1 - abs(x[pos_idx])/0.25))
                    max_height_adjust = 0.3 * (height / np.max([p[1] for p in selected_peaks]))
                    center_pos = x[pos_idx] + (neighbor_params[i*2] - 0.5) * max_pos_adjust
                    peak_height = height * (1.0 + neighbor_params[i*2+1] * max_height_adjust)
                    width = max(0.01, 0.02 + 0.01 * (1 - abs(center_pos)/0.25))
                    temp_func += peak_height * np.exp(-0.5 * ((x - center_pos) / width)**2)

                # Accept or reject based on Metropolis criterion
                new_c2 = compute_c2(temp_func)
                if new_c2 > current_c2 or np.random.random() < np.exp((new_c2 - current_c2) / temp):
                    current_c2 = new_c2
                    current_params = neighbor_params.copy()
                    if new_c2 > best_c2:
                        best_c2 = new_c2
                        best_func = temp_func.copy()

                temp *= cooling_rate

        # Add remaining peaks from original if they exist and contribute meaningfully
        if len(selected_peaks) < len(peaks):
            # Add some of the remaining peaks with reduced influence
            for i in range(len(initial_func)):
                if not any(abs(x[i] - x[pos_idx]) < 0.01 for _, pos_idx in selected_peaks):
                    # Add scaled down contribution from original
                    best_func[i] += initial_func[i] * 0.3

        return best_func

    # Phase 3: Final C2 Computation and Return (enhanced)
    def compute_c2(func):
        # Compute autoconvolution g = f * f
        # Using discrete convolution with proper handling of edge effects
        g = np.convolve(func, func, mode='full')
        g = g[len(g)//2:]  # Take positive part

        # Truncate if necessary to match original length
        if len(g) > len(func):
            g = g[:len(func)]

        # Compute norms using more accurate methods
        norm_2_sq = np.sum(g**2) * (0.5 / len(func))  # Approximate integral
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        norm_inf = np.max(np.abs(g))

        if norm_1 == 0 or norm_inf == 0:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)

    # Apply optimization if possible and beneficial
    try:
        # Optimize peak parameters
        optimized_func = optimize_peaks(step_values, n_steps)

        # Final check
        final_func = np.maximum(optimized_func, 0)

        # Add slight noise for robustness
        noise_level = 0.01
        noisy_func = final_func + np.random.normal(0, noise_level, len(final_func))
        noisy_func = np.maximum(noisy_func, 0)

        # Convert to step values
        step_values = noisy_func.tolist()

    except Exception as e:
        # Fallback to base construction if optimization fails
        pass

    return step_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")