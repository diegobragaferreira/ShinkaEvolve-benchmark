# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random

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
    Enhanced construct function that uses improved Gaussian peak placement and
    adaptive optimization to maximize C2.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None

    # Try multiple construction strategies
    max_attempts = 30

    for attempt in range(max_attempts):
        # Try different step counts for optimal resolution
        n_steps = np.random.randint(1500, 4000)

        # Enhanced Gaussian peak construction with strategic positioning
        # Use logarithmic spacing to distribute peaks more evenly
        peak_count = np.random.randint(10, 50)

        # Create log-distributed positions to avoid clustering
        positions = np.logspace(np.log10(0.001), np.log10(0.25), peak_count)
        # Mirror positions to create symmetric peaks
        positions = np.concatenate([-positions[::-1], positions])
        # Take only what we need
        positions = positions[:n_steps]

        # Add randomness to break systematic patterns
        positions = positions + np.random.normal(0, 0.01, len(positions))
        positions = np.clip(positions, -0.25, 0.25)
        positions = np.sort(positions)

        # Generate amplitudes with peak concentration near center
        amplitudes = np.exp(-15 * positions**2)

        # Add adaptive noise with variance that depends on peak density
        # More noise in sparse regions, less in dense regions
        noise_level = 0.15
        amplitudes += np.random.normal(0, noise_level, n_steps)

        # Ensure non-negative amplitudes and normalize
        amplitudes = np.maximum(amplitudes, 0)

        if np.sum(amplitudes) > 0:
            amplitudes = amplitudes / np.sum(amplitudes) * 100

        # Check if this function has good C2 before returning
        try:
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(amplitudes.tolist())

            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)

                # Keep the best function found
                if c2 > best_c2:
                    best_c2 = c2
                    best_function = amplitudes.tolist()

        except Exception:
            continue

    # Fallback if nothing worked
    if best_function is None:
        n_steps = 1000
        best_function = [1.0] * n_steps

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")