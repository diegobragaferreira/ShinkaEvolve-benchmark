# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import math

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation.
    f_values: list of step heights
    Returns: ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)

    # Perform convolution (autoconvolution)
    # Using 'full' mode to get complete convolution result
    g = signal.convolve(f, f, mode='full')

    # The convolution result has length 2*n - 1 where n is length of f
    # We're interested in the central portion that corresponds to the actual autoconvolution
    # For our problem setup, we expect the result to be centered

    # Calculate the norms
    # ||g||₂² = sum(gᵢ²)
    g_squared = g * g
    norm_g_2_squared = np.sum(g_squared)

    # ||g||₁ = sum(|gᵢ|)
    norm_g_1 = np.sum(np.abs(g))

    # ||g||∞ = max(|gᵢ|)
    norm_g_inf = np.max(np.abs(g))

    return norm_g_2_squared, norm_g_1, norm_g_inf

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses adaptive peak shaping with controlled noise injection.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Choose a reasonable number of steps (between 100 and 1000)
    n_steps = np.random.randint(100, 1000)

    # Create a base structure that promotes good C2 behavior
    # Start with a symmetric distribution that balances peak and spread
    x = np.linspace(-1, 1, n_steps)

    # Create a base shape with controlled peak characteristics
    # Use a modified bell curve that avoids extreme peaks
    base_shape = np.exp(-x**2 / 2)

    # Scale and shift to create a reasonable base distribution
    base_shape = 0.7 * (base_shape / np.max(base_shape)) + 0.15

    # Apply adaptive noise injection based on the target behavior
    f_values = []

    # Compute initial norms to understand the behavior
    test_f = base_shape.copy()
    _, _, norm_g_inf_initial = compute_autoconvolution_norms(test_f.tolist())

    # Use adaptive noise scaling based on expected behavior
    # If the initial peak is too dominant, reduce noise to prevent over-shaping
    # If the peak is moderate, allow more exploration through noise
    noise_scale_factor = max(0.01, min(0.2, 0.1 / (norm_g_inf_initial + 1e-10)))

    for i in range(n_steps):
        base_val = base_shape[i]

        # Adaptive noise injection - scale based on current behavior
        noise_amplitude = noise_scale_factor * (0.1 + 0.1 * np.sin(i * 0.1))
        noise = np.random.normal(0, noise_amplitude)

        # Apply noise and ensure non-negativity
        val = max(0, base_val + noise)
        f_values.append(val)

    # Post-process to balance the function further
    # Check for overly dominant peaks and adjust accordingly
    _, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

    # If the infinity norm is too dominant relative to L1 norm,
    # reduce peak values slightly to improve the C2 ratio
    if norm_g_1 > 0 and norm_g_inf / norm_g_1 > 0.5:
        # Reduce peaks moderately if they dominate the distribution
        f_values = [max(0, val * 0.9) for val in f_values]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")