# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
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

def create_log_spaced_step_function(n_steps: int = 3000) -> list[float]:
    """
    Create a step function with logarithmic spacing of peak positions to improve C2.
    This approach uses logarithmic distribution to avoid clustering of peaks,
    and adaptive amplitude control to manage autoconvolution spikes.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create positions using logarithmic distribution to avoid clustering
    # This helps avoid the formation of overly sharp autoconvolution peaks
    n_peaks = min(200, n_steps // 5)  # Dynamic number of peaks, fewer than before

    # Create logarithmic spacing for peak positions
    # Use logspace to put more peaks near the center where they're more effective
    log_positions = np.logspace(np.log10(0.001), np.log10(0.25), n_peaks//2)
    # Mirror and combine for symmetric distribution
    positions = np.concatenate([-log_positions[::-1], log_positions])

    # Ensure we don't exceed our step count
    positions = positions[:n_steps]

    # Add some randomness to break systematic patterns but maintain structure
    noise_level = 0.015
    positions += np.random.normal(0, noise_level, len(positions))

    # Clip to valid range and sort
    positions = np.clip(positions, -0.25, 0.25)
    positions = np.sort(positions)

    # Generate amplitudes with adaptive behavior
    # Peaks near center get higher amplitude, with natural decay
    amplitudes = np.exp(-8 * positions**2)  # Stronger center concentration

    # Add adaptive noise with dynamic scaling
    adaptive_noise = np.random.normal(0, 0.1, len(positions))
    amplitudes += adaptive_noise

    # Ensure non-negative amplitudes
    amplitudes = np.maximum(amplitudes, 0)

    # Apply constraint-aware normalization to prevent extreme autoconvolution spikes
    # We try to balance between having enough energy and avoiding spikes that hurt C2
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes) * 120  # Scaled appropriately

        # Quick check to avoid functions that would have problematic autoconvolution
        # Compute a quick test to see if we're getting reasonable values
        test_f = amplitudes.copy()
        test_g = np.convolve(test_f, test_f, mode='full')
        test_g = test_g[len(test_f)-1:2*len(test_f)-1]
        test_norm_1 = np.sum(np.abs(test_g))
        test_norm_inf = np.max(np.abs(test_g)) if np.max(np.abs(test_g)) > 0 else 1e-15

        # If there are issues with the preliminary test, adjust
        if test_norm_inf > 10 * test_norm_1 and test_norm_1 > 0:
            # Reduce amplitude scaling to prevent too dominant peaks
            amplitudes = amplitudes * 0.8

    # Final normalization
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes) * 100

    return amplitudes.tolist()

def construct_function() -> list[float]:
    """
    Main function to construct optimized step function using logarithmic spacing approach.
    This approach is designed to beat the benchmark of 0.962 by creating more effective
    step function profiles with better autoconvolution characteristics.
    """
    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()

    # Set maximum attempts to balance quality vs. time constraints
    max_attempts = 100

    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break

        # Try different number of steps with preference toward larger ones
        # Larger step counts typically give better resolution for the autoconvolution
        n_steps = np.random.randint(1500, 5000)

        # Generate function using logarithmic spacing approach
        f_values = create_log_spaced_step_function(n_steps)

        # Evaluate the function
        try:
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)

            # Check for valid norms
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