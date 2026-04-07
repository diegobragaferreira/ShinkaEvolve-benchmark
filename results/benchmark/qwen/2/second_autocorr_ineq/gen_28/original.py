# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal

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
    g_sq = g**2
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
    Construct step function with optimized Gaussian-based peak placement.
    """
    # Number of steps
    n_steps = np.random.randint(1000, 5000)

    # Generate positions using Gaussian distribution around center
    # This creates a bell-curve shaped distribution of peak heights
    positions = np.random.normal(0, 0.15, n_steps)
    # Clip to [-0.25, 0.25] to stay within domain
    positions = np.clip(positions, -0.25, 0.25)

    # Sort positions for proper step function construction
    positions = np.sort(positions)

    # Generate amplitudes based on distance from center
    # Peaks closer to center get higher amplitude
    amplitudes = np.exp(-20 * positions**2)

    # Add some controlled noise to break symmetry but preserve structure
    noise_level = 0.1
    amplitudes += np.random.normal(0, noise_level, n_steps)

    # Ensure non-negative amplitudes
    amplitudes = np.maximum(amplitudes, 0)

    # Normalize amplitudes to make them reasonable
    if np.sum(amplitudes) > 0:
        amplitudes = amplitudes / np.sum(amplitudes) * 10

    # Return as list
    return amplitudes.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")