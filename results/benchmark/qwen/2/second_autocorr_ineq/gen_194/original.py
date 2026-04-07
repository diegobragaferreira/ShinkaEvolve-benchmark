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
    Uses enhanced Gaussian peak construction with strategic placement and adaptive scaling.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Choose a reasonable number of steps (between 500 and 2000 for better resolution)
    n_steps = np.random.randint(500, 2000)

    # Create multiple Gaussian peaks with strategic placement
    f_values = np.zeros(n_steps)

    # Determine number of peaks - more peaks for better control of autoconvolution shape
    n_peaks = max(3, min(15, n_steps // 100))

    # Define peak positions with minimum spacing to avoid overly sharp autoconvolution
    peak_positions = []

    # First peak (left side)
    peak_positions.append(int(n_steps * 0.15))

    # Middle peaks with minimum spacing
    if n_peaks > 2:
        spacing = max(50, n_steps // (n_peaks + 1))
        for i in range(n_peaks - 2):
            pos = int(peak_positions[0] + (i + 1) * spacing)
            if pos >= n_steps - 50:  # Keep away from right edge
                pos = n_steps - 50
            peak_positions.append(pos)

    # Last peak (right side)
    peak_positions.append(int(n_steps * 0.85))

    # Ensure minimum gap between consecutive peaks
    adjusted_positions = [peak_positions[0]]
    for i in range(1, len(peak_positions)):
        min_gap = max(30, n_steps // 25)  # Minimum 30 steps gap or 4% of domain
        prev_pos = adjusted_positions[-1]
        new_pos = max(prev_pos + min_gap, peak_positions[i])
        adjusted_positions.append(new_pos)

    peak_positions = adjusted_positions

    # Generate peaks with optimized widths and heights
    for i, center_pos in enumerate(peak_positions):
        # Width decreases with peak position to maintain overall shape balance
        width = max(15, min(80, n_steps // (10 + i*2)))

        # Height inversely proportional to width to keep area balanced
        # but increase slightly for outer peaks to encourage better autoconvolution
        height = 1.0 + 0.3 * (i == 0 or i == len(peak_positions)-1)  # Outer peaks higher

        # Create Gaussian peak
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center_pos) / width) ** 2)
        f_values += gaussian

    # Apply adaptive smoothing to reduce noise while preserving peak structure
    if n_steps > 100:
        window_size = min(51, n_steps // 5)  # Smaller window for higher resolution
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_values = signal.savgol_filter(f_values, window_size, 3)

    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)

    # Normalize to reasonable range
    if np.max(f_values) > 0:
        f_values = f_values / np.max(f_values) * 2.0

    # Apply constraint-aware normalization to prevent extreme autoconvolution spikes
    _, _, norm_g_inf = compute_autoconvolution_norms(f_values.tolist())
    if norm_g_inf > 0:
        # Cap extreme values to keep autoconvolution manageable
        max_allowed = np.percentile(f_values, 90) if len(f_values) > 10 else 1.0
        if max_allowed > 0:
            f_values = np.minimum(f_values, max_allowed * 3.0)

    # Convert back to list
    result = f_values.tolist()

    # Final refinement: apply small adjustments to optimize C2
    # Recompute norms to check quality
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(result)

    # If autoconvolution has very high peak relative to L1 (which hurts C2),
    # adjust values to flatten the profile
    if norm_1 > 0 and norm_inf / norm_1 > 0.7:
        # Reduce all values to decrease peak dominance
        result = [max(0, val * 0.95) for val in result]

    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")