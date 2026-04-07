# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from numba import jit
import time

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    n_steps = len(f)

    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n_steps

    # Compute autoconvolution using discrete convolution
    g = np.convolve(f, f, mode='full')
    # Trim g to the correct size (this accounts for the convolution)
    g = g[len(f)-1:2*len(f)-1]

    # Compute L2 norm squared using piecewise linear integration
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        # Trapezoidal-like integration for quadratic function
        # Using formula for integral of ax^2 + bx + c over [x0,x1]
        # But here we approximate with piecewise linear segments
        # So we use: (dx/3)(y0^2 + y0*y1 + y1^2)
        y0, y1 = g[i], g[i+1]
        norm_2_squared += (dx/3) * (y0**2 + y0*y1 + y1**2)

    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g))

    # Infinity norm
    norm_inf = np.max(np.abs(g))

    # Handle numerical edge cases
    if norm_1 <= 1e-15:
        norm_1 = 1e-15
    if norm_inf <= 1e-15:
        norm_inf = 1e-15

    return norm_2_squared, norm_1, norm_inf

def adaptive_gaussian_construction(n_steps: int) -> list[float]:
    """
    Construct a step function with enhanced Gaussian-based peak placement using logarithmic spacing
    and multi-scale peak distribution to improve C2 values.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Multi-scale approach: create peaks at different scales to cover various frequency ranges
    domain_width = 0.5

    # Determine number of peaks based on resolution
    n_peaks = max(20, min(200, n_steps // 25))

    # Use logarithmic distribution to avoid clustering in narrow regions
    # Create multiple scales of peaks
    scales = np.logspace(np.log10(0.01), np.log10(0.25), 5)  # 5 different scales

    all_positions = []

    for scale in scales:
        # Determine how many peaks per scale (more peaks at smaller scales)
        n_per_scale = max(2, int(n_peaks * scale / 0.25))
        # Generate positions with logarithmic spacing within this scale
        positions = np.logspace(np.log10(scale * 0.1), np.log10(scale), n_per_scale)
        # Mirror to create symmetric distribution
        positions = np.concatenate([-positions[::-1], positions])
        # Filter to domain
        positions = positions[(positions >= -0.25) & (positions <= 0.25)]
        all_positions.extend(positions)

    # Remove duplicates and sort
    all_positions = np.unique(all_positions)

    # Ensure we don't exceed our target number of peaks
    if len(all_positions) > n_peaks:
        # Select a subset using weighted sampling toward center
        weights = np.exp(-10 * all_positions**2)  # Higher weight for center positions
        weights = weights / np.sum(weights)
        selected_indices = np.random.choice(len(all_positions), size=n_peaks, p=weights, replace=False)
        all_positions = all_positions[selected_indices]

    # Enhanced minimum spacing enforcement - calculate adaptive spacing based on distribution
    if len(all_positions) > 1:
        # Calculate adaptive minimum distance based on peak density
        sorted_positions = np.sort(all_positions)
        distances = np.diff(sorted_positions)
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            min_spacing = max(0.01 * domain_width, avg_distance * 0.7)  # Adaptive spacing
        else:
            min_spacing = 0.02 * domain_width  # Default spacing

        # Improved spacing enforcement algorithm
        filtered_positions = [sorted_positions[0]]
        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] - filtered_positions[-1] >= min_spacing:
                filtered_positions.append(sorted_positions[i])
        all_positions = np.array(filtered_positions)

    # Generate amplitudes that decay based on distance from center but with adaptive scaling
    amplitudes = np.exp(-15 * all_positions**2)  # Stronger center concentration

    # Add structured noise to break symmetry while maintaining overall structure
    # Use a combination of Gaussian noise and sinusoidal modulation
    noise = np.random.normal(0, 0.2, len(all_positions))
    # Add sinusoidal modulation to avoid degenerate cases
    modulation = 0.1 * np.sin(10 * np.pi * all_positions) * np.exp(-all_positions**2/0.05)
    amplitudes += noise + modulation

    # Ensure non-negative amplitudes
    amplitudes = np.maximum(amplitudes, 0)

    # Apply adaptive amplitude scaling to prevent autoconvolution spikes
    # Generate test function and check autoconvolution quality
    test_amplitudes = amplitudes.copy()
    if np.sum(test_amplitudes) > 0:
        test_amplitudes = test_amplitudes / np.sum(test_amplitudes) * 10

        # Quick test of autoconvolution
        test_g = np.convolve(test_amplitudes, test_amplitudes, mode='full')
        test_g = test_g[len(test_amplitudes)-1:2*len(test_amplitudes)-1]
        test_norm_1 = np.sum(np.abs(test_g))
        test_norm_inf = np.max(np.abs(test_g)) if np.max(np.abs(test_g)) > 0 else 1e-15

        # If the autoconvolution shows signs of being too spiked, reduce amplitudes
        if test_norm_inf > 5 * test_norm_1 and test_norm_1 > 0:
            # Scale down amplitudes to reduce peakiness
            amplitudes = amplitudes * 0.7

    # Final normalization with additional care to preserve structure
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes) * 10

    # Convert to step function by interpolating to desired resolution
    x_domain = np.linspace(-0.25, 0.25, n_steps)

    # Create step function using Gaussian interpolation with more sophisticated approach
    step_function = np.zeros(n_steps)

    # For better peak distribution control, we'll build peaks with adaptive widths
    peak_params = []  # Store peak parameters [amplitude, center, width]

    # Calculate adaptive widths based on peak positions and densities
    peak_densities = np.ones(len(all_positions))
    if len(all_positions) > 1:
        distances = np.diff(np.sort(np.abs(all_positions)))
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            # Higher density = narrower widths, lower density = wider widths
            peak_densities = np.clip(1.0 / (distances + 1e-8), 0.5, 2.0)
            # Extend to full length
            peak_densities = np.append(peak_densities, peak_densities[-1] if len(peak_densities) > 0 else 1.0)
        else:
            peak_densities = np.ones(len(all_positions))

    # Create peak parameters with adaptive widths
    for i, (pos, amp) in enumerate(zip(all_positions, amplitudes)):
        # Base width calculation: central peaks narrower, outer peaks wider
        base_width = 0.02 + 0.06 * (1.0 - np.abs(pos) / 0.25)
        # Apply density-based adjustment
        width = base_width / (peak_densities[i] * 0.5 + 0.5)
        width = np.clip(width, 0.01, 0.15)  # Keep within reasonable bounds
        peak_params.extend([amp, pos, width])

    # Now actually build the function with adaptive widths
    for i in range(0, len(peak_params), 3):
        amp = peak_params[i]
        pos = peak_params[i+1]
        width = peak_params[i+2]
        step_function += amp * np.exp(-0.5 * ((x_domain - pos) / width)**2)

    # Apply smoothing to reduce numerical artifacts while preserving structure
    if n_steps > 100:
        try:
            from scipy.signal import gaussian, convolve
            # Create a Gaussian kernel for smoothing
            kernel_size = min(21, n_steps // 20 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = gaussian(kernel_size, std=kernel_size/6.0)
            kernel = kernel / np.sum(kernel)  # Normalize
            # Apply convolution to smooth without losing peak information
            step_function = convolve(step_function, kernel, mode='same')
        except:
            # Fallback to simple moving average if scipy unavailable
            window_size = min(15, n_steps // 10)
            if window_size % 2 == 0:
                window_size += 1
            if window_size > 1:
                step_function = np.convolve(step_function, np.ones(window_size)/window_size, mode='same')

    # Add fine-grained modulation to break any remaining symmetries
    # Use more diverse frequencies to avoid degenerate cases
    fine_modulation = 0.03 * np.sin(40 * np.pi * x_domain) * np.exp(-x_domain**2/0.04)
    fine_modulation += 0.02 * np.cos(25 * np.pi * x_domain) * np.exp(-x_domain**2/0.06)
    step_function += fine_modulation

    # Ensure non-negativity and normalize
    step_function = np.maximum(step_function, 0)
    if np.sum(step_function) > 0:
        # Use a more conservative normalization to avoid over-amplification
        step_function = step_function / np.sum(step_function) * 8

    return step_function.tolist()

def construct_function() -> list[float]:
    """
    Construct step function with optimized Gaussian-based peak placement.
    """
    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()

    # Set maximum attempts to balance quality vs. time constraints
    max_attempts = 50

    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break

        # Try different number of steps to find optimal
        n_steps = np.random.randint(1000, 5000)

        # Generate function
        f_values = adaptive_gaussian_construction(n_steps)

        # Evaluate the function
        try:
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                continue
            c2 = norm_2_sq / (norm_1 * norm_inf)

            # Keep the best function
            if c2 > best_c2:
                best_c2 = c2
                best_function = f_values

        except Exception:
            # Skip invalid functions
            continue

    # Return the best function found, or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a simpler construction
        n_steps = 1000
        f_values = [1.0] * n_steps
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")