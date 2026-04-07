# EVOLVE-BLOCK-START

import numpy as np
from numba import njit

@njit
def compute_autoconvolution_norms(f_values):
    """Compute autoconvolution and norms efficiently with numba JIT"""
    n = len(f_values)
    # Compute autoconvolution g = f * f
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared)

    # Approximate L1 norm using trapezoidal rule
    norm_g1_approx = np.sum(np.abs(g)) / (len(g) + 1)

    # Infinity norm
    norm_ginf = np.max(np.abs(g))

    return norm_g2_squared, norm_g1_approx, norm_ginf

@njit
def calculate_c2(f_values):
    """Calculate C2 value for given step function values"""
    norm_g2_squared, norm_g1_approx, norm_ginf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_g1_approx < 1e-15 or norm_ginf < 1e-15:
        return 0.0

    c2 = norm_g2_squared / (norm_g1_approx * norm_ginf)
    return c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Generate a more sophisticated initial function
    n = np.random.randint(1000, 5000)
    f_values = np.random.random(n) * 2  # Scale to [0,2]
    # Apply some smoothing to avoid extreme peaks
    f_values = np.clip(f_values, 0, 1)
    return f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")