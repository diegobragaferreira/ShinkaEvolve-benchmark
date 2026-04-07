# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import time

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)

    # Create step function on [-1/4, 1/4] with equal spacing
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
    # For each pair of adjacent points, integrate quadratic function
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

def construct_function() -> list[float]:
    """
    Construct step function with enhanced Gaussian-based peak optimization.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of steps with better distribution
    n_steps = np.random.randint(2000, 5000)

    # Multi-scale approach: create peaks at different scales to cover various frequency ranges
    domain_width = 0.5

    # Determine number of peaks based on resolution but favor more peaks for better optimization
    n_peaks = max(30, min(200, n_steps // 20))

    # Use logarithmic distribution to avoid clustering in narrow regions
    # Create multiple scales of peaks distributed via log-space
    scales = np.logspace(np.log10(0.01), np.log10(0.25), 6)  # 6 different scales

    all_positions = []

    for scale in scales:
        # Determine how many peaks per scale (more peaks at smaller scales)
        n_per_scale = max(3, int(n_peaks * scale / 0.25))
        # Generate positions with logarithmic spacing within this scale
        # We use a log-uniform distribution to space peaks more evenly
        positions = np.random.loguniform(scale * 0.1, scale, n_per_scale)
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

    # Ensure minimum spacing to prevent narrow autoconvolution interference
    if len(all_positions) > 1:
        min_spacing = 0.015 * domain_width  # 1.5% of domain width
        sorted_positions = np.sort(all_positions)
        filtered_positions = [sorted_positions[0]]

        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] - filtered_positions[-1] >= min_spacing:
                filtered_positions.append(sorted_positions[i])

        all_positions = np.array(filtered_positions)

    # Generate amplitudes that decay based on distance from center but with adaptive scaling
    amplitudes = np.exp(-12 * all_positions**2)  # Stronger center concentration

    # Apply adaptive amplitude scaling with feedback from preliminary evaluation
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

    # Add structured noise to break symmetry while maintaining overall structure
    # Use a combination of Gaussian noise and sinusoidal modulation
    noise = np.random.normal(0, 0.15, len(all_positions))
    # Add sinusoidal modulation to avoid degenerate cases
    modulation = 0.08 * np.sin(15 * np.pi * all_positions) * np.exp(-all_positions**2/0.05)
    amplitudes += noise + modulation

    # Ensure non-negative amplitudes
    amplitudes = np.maximum(amplitudes, 0)

    # Final normalization with additional care to preserve structure
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes) * 10

    # Convert to step function by interpolating to desired resolution
    x_domain = np.linspace(-0.25, 0.25, n_steps)
    peak_x = np.linspace(-0.25, 0.25, len(all_positions))

    # Create step function using Gaussian interpolation with slightly wider kernels
    step_function = np.zeros(n_steps)
    for i, (pos, amp) in enumerate(zip(all_positions, amplitudes)):
        # Use a wider Gaussian to spread the influence and create smoother transitions
        width = 0.04  # Slightly wider than previous version
        step_function += amp * np.exp(-0.5 * ((x_domain - pos) / width)**2)

    # Add fine-grained modulation to break any remaining symmetries
    fine_modulation = 0.06 * np.sin(30 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)
    step_function += fine_modulation

    # Ensure non-negativity and normalize
    step_function = np.maximum(step_function, 0)
    if np.sum(step_function) > 0:
        step_function = step_function / np.sum(step_function) * 10

    # Return as list
    return step_function.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")