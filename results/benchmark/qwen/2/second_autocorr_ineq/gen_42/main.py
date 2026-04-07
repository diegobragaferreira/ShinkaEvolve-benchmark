# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import time
from numba import jit

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

def create_optimized_step_function() -> list[float]:
    """
    Create an optimized step function designed to maximize C2.
    Uses logarithmic spacing for peak positions and adaptive amplitude selection.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Parameters for function construction
    n_steps = 2000  # More steps for better resolution
    max_attempts = 100  # Maximum attempts to find good function

    best_c2 = -1
    best_function = None

    start_time = time.time()

    # Try multiple strategies
    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break

        # Strategy 1: Logarithmic distributed peaks with varying amplitudes
        # Create positions using logarithmic distribution to avoid clustering
        positions = np.logspace(np.log10(0.001), np.log10(0.25), n_steps//2)
        positions = np.concatenate([-positions[::-1], positions])  # Symmetric around center

        # Randomly shuffle positions to break any systematic patterns
        np.random.shuffle(positions)
        positions = np.clip(positions, -0.25, 0.25)
        positions = np.sort(positions)

        # Create amplitudes that decay from center outward to prevent sharp peaks
        amplitudes = np.exp(-10 * positions**2)  # Gaussian-like decay

        # Add some randomness to break symmetry but keep structure
        noise_factor = 0.3
        noise = np.random.normal(0, noise_factor, n_steps)
        amplitudes += noise

        # Ensure non-negative amplitudes
        amplitudes = np.maximum(amplitudes, 0)

        # Normalize amplitudes to have reasonable magnitude
        if np.sum(amplitudes) > 0:
            amplitudes = amplitudes / np.sum(amplitudes) * 100

        # Apply final checks and adjustments
        try:
            # Compute norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(amplitudes.tolist())

            # Skip if invalid
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                continue

            c2 = norm_2_sq / (norm_1 * norm_inf)

            # Keep the best function
            if c2 > best_c2:
                best_c2 = c2
                best_function = amplitudes.tolist()

        except Exception:
            # Skip invalid functions
            continue

    # Return the best function found or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a uniform distribution
        return [1.0] * n_steps

def construct_function() -> list[float]:
    """
    Main function to construct optimized step function.
    """
    return create_optimized_step_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")