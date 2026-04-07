# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
import time
from numba import jit
import torch
import torch.nn.functional as F

# Global constants
MAX_STEPS = 50000
MIN_STEPS = 100
SEED = 42
TIME_LIMIT = 85  # seconds

# Set seed for reproducibility
np.random.seed(SEED)

@jit(nopython=True)
def compute_convolution_norms_numba(f_values):
    """Compute the three norms needed for C2 calculation using numba JIT acceleration"""
    # Convert to numpy array
    f = np.array(f_values)
    n_steps = len(f)

    # Create step function on [-1/4, 1/4] with proper spacing
    step_width = 0.5 / n_steps
    step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)

    # For autoconvolution, we'll work with a finer grid
    # Create piecewise constant function on refined grid
    x_fine = np.linspace(-0.25, 0.25, 1000)  # Fine grid for convolution
    dx = x_fine[1] - x_fine[0]

    # Build piecewise constant function
    f_func = np.zeros_like(x_fine)
    for i in range(n_steps):
        pos = step_positions[i]
        left = pos - step_width/2
        right = pos + step_width/2
        # Find indices where x_fine falls in this step
        mask = (x_fine >= left) & (x_fine <= right)
        f_func[mask] = f[i]

    # Perform autoconvolution manually (simplified version) - but more carefully
    # We compute g = f * f where * is convolution
    g = np.zeros(len(x_fine))
    for i in range(len(x_fine)):
        total = 0.0
        for j in range(len(x_fine)):
            if i - j >= 0 and i - j < len(x_fine):
                total += f_func[j] * f_func[i-j]
        g[i] = total

    # Normalize for proper scaling
    g = g * dx

    # Compute the required norms
    g_squared = g**2
    g_abs = np.abs(g)

    # ||g||₂² (using trapezoidal rule for integration)
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        norm_2_squared += (dx/3) * (g_squared[i] + g_squared[i+1] + g_squared[i]*g_squared[i+1])

    # ||g||₁ (L1 norm)
    norm_1 = 0.0
    for i in range(len(g)-1):
        norm_1 += (dx/2) * (g_abs[i] + g_abs[i+1])

    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs)

    return norm_2_squared, norm_1, norm_inf

def calculate_c2(f_values):
    """Calculate C₂ from step function values"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_convolution_norms_numba(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        return 0.0

def generate_multiscale_gaussian_function(n_steps):
    """
    Generate a step function with multi-scale Gaussian bumps,
    creating a structured initialization that improves optimization performance.
    """
    # Create multi-scale Gaussian pattern
    step_width = 0.5 / n_steps
    step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)

    # Generate multi-scale Gaussian components with different widths and amplitudes
    heights = np.zeros(n_steps)

    # Base scale - large scale features
    scales = [0.1, 0.05, 0.025]  # Different width scales
    amplitudes = [1.0, 0.7, 0.3]  # Different amplitudes

    for scale, amp in zip(scales, amplitudes):
        # Create Gaussian bump positions
        bump_centers = np.linspace(-0.2, 0.2, 5)  # 5 main bumps
        for center in bump_centers:
            # Generate Gaussian with specific scale and amplitude
            gaussian = amp * np.exp(-0.5 * ((step_positions - center) / scale)**2)
            heights += gaussian

    # Add some random variation to make it less rigid
    noise_factor = 0.1
    heights += np.random.normal(0, noise_factor * np.mean(heights), n_steps)

    # Ensure all values are non-negative
    heights = np.maximum(heights, 0)

    return heights.tolist()

def objective_and_gradient(params):
    """Compute C2 and its gradient w.r.t. step heights"""
    # params contains the step function heights
    f_values = params.tolist()

    # Compute C2
    c2 = calculate_c2(f_values)

    # Gradient calculation: This is a simplified version that assumes we can
    # approximate the gradient using finite differences (not very efficient but works).
    # In a perfect world, we'd compute analytical gradients, but that's complex.
    # Here we'll just use finite difference approximation for demonstration.

    # For simplicity, we'll return only the function value, and let optimization
    # framework handle the gradient computation or use a built-in method instead

    return -c2  # Negative because we want to maximize C2, but optimizers minimize

def direct_gradient_ascent():
    """Direct gradient ascent optimization approach"""
    # Start with a good initialization
    n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
    initial_heights = generate_multiscale_gaussian_function(n_steps)

    # Convert to tensor for easier manipulation
    x0 = np.array(initial_heights, dtype=np.float64)
    x0 = np.maximum(x0, 0.0)  # Ensure non-negative

    # Use scipy.optimize.minimize with L-BFGS-B which handles bounds well
    # We are minimizing -C2, so maximizing C2
    try:
        result = minimize(
            lambda x: -calculate_c2(x.tolist()),
            x0,
            method='L-BFGS-B',
            bounds=[(0, None) for _ in range(len(x0))],
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9},
            callback=lambda x: print(f"Iteration completed, current C2: {calculate_c2(x.tolist()):.6f}")
        )

        if result.success:
            best_solution = result.x
        else:
            best_solution = x0
    except Exception as e:
        # Fallback to initial solution
        best_solution = x0

    # Ensure non-negativity
    best_solution = np.maximum(best_solution, 0.0)

    return best_solution.tolist()

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using direct gradient ascent."""
    start_time = time.time()

    # Run direct gradient ascent optimization
    result = direct_gradient_ascent()

    end_time = time.time()
    eval_time = end_time - start_time

    print(f"Evaluated in {eval_time:.2f} seconds")
    print(f"Best C2 found: {calculate_c2(result):.6f}")

    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")