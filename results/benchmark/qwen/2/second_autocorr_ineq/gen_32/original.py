# EVOLVE-BLOCK-START

import numpy as np

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    np.random.seed(42)  # For reproducibility

    # Determine number of steps
    n_steps = np.random.randint(200, 5000)

    # Create x-axis points in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, n_steps)

    # Initialize with base multi-peak Gaussian structure
    base_function = np.zeros_like(x)

    # Create multiple peaks with varying characteristics
    num_peaks = np.random.randint(3, 8)

    for _ in range(num_peaks):
        # Random peak parameters
        peak_center = np.random.uniform(-0.15, 0.15)
        peak_height = np.random.uniform(0.8, 2.0)
        peak_width = np.random.uniform(0.02, 0.08)

        # Create Gaussian peak
        gaussian_peak = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width)**2)
        base_function += gaussian_peak

    # Add some additional structure with controlled randomness
    # This helps create better autoconvolution properties
    for i in range(0, len(x), max(1, len(x)//10)):
        if np.random.random() > 0.7:  # 30% chance to add small bump
            bump_center = x[i]
            bump_height = np.random.uniform(0.1, 0.5)
            bump_width = np.random.uniform(0.005, 0.02)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            base_function += bump

    # Ensure non-negative values
    base_function = np.maximum(base_function, 0)

    # Normalize to avoid extreme values that might hurt the C2 calculation
    if np.max(base_function) > 0:
        base_function = base_function / np.max(base_function) * 1.5

    # Apply light noise for robustness (but preserve structure)
    noise_level = 0.03
    noisy_function = base_function + np.random.normal(0, noise_level, len(base_function))
    noisy_function = np.maximum(noisy_function, 0)

    # Convert to step values ensuring proper format
    step_values = noisy_function.tolist()

    return step_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")