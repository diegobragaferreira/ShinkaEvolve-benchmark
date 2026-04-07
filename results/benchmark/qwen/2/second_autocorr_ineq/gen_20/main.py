# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal

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
    Combines structured initialization with adaptive refinement.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Choose a reasonable number of steps (between 100 and 1000)
    n_steps = np.random.randint(100, 1000)

    # Create a structured base shape inspired by successful patterns
    # Start with a symmetric bell-shaped distribution
    x = np.linspace(-1, 1, n_steps)
    
    # Create a bell curve with controlled peak
    base_shape = np.exp(-x**2 / 2)
    
    # Scale and shift to create a reasonable base distribution
    # Add some structure but avoid extreme peaks that could dominate norms
    base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
    
    # Initialize with the base structure
    f_values = base_shape.tolist()

    # Apply adaptive refinement based on initial performance
    _, _, norm_g_inf_initial = compute_autoconvolution_norms(f_values)

    # Adjust noise level based on initial behavior
    # If initial peak is too high, use less noise to prevent over-shaping
    # If it's moderate, allow more exploration
    noise_level = max(0.02, min(0.15, 0.08 / (norm_g_inf_initial + 1e-10)))
    
    # Apply adaptive noise injection
    for i in range(n_steps):
        # Base value from our structured shape
        base_val = base_shape[i]
        
        # Add adaptive noise
        noise_amplitude = noise_level * (0.1 + 0.1 * np.sin(i * 0.1))
        noise = np.random.normal(0, noise_amplitude)
        
        # Apply noise and ensure non-negativity
        val = max(0, base_val + noise)
        f_values[i] = val

    # Post-processing to improve the final ratio
    # Recompute norms to check current state
    _, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)
    
    # If the infinity norm dominates too much relative to L1 norm,
    # slightly reduce all values to improve C2 ratio
    if norm_g_1 > 0 and norm_g_inf / norm_g_1 > 0.6:
        # Reduce values moderately if they dominate the distribution
        f_values = [max(0, val * 0.92) for val in f_values]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")