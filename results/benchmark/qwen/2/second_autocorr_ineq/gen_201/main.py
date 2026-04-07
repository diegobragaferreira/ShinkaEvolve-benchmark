# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
from typing import List, Tuple

def construct_function() -> list[float]:
    """
    Spectral-guided step function optimizer that leverages frequency domain analysis
    to construct optimal step functions for maximizing C₂ constant.
    """
    np.random.seed(42)

    # Determine number of steps with reasonable resolution
    n_steps = 5000  # Fixed at 5000 to match AlphaEvolve benchmark

    # Create x-axis points in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, n_steps)

    # Phase 1: Spectral-guided initialization
    # Analyze what spectral properties lead to good autoconvolution behavior
    spectral_peaks = generate_spectral_guided_peaks(n_steps)

    # Build function from spectral-guided peaks
    base_function = np.zeros_like(x)

    # Apply the generated peaks with optimized parameters
    for peak_info in spectral_peaks:
        pos, height, width = peak_info
        gaussian_peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
        base_function += gaussian_peak

    # Add supplementary structure for better autoconvolution properties
    add_supplementary_structure(base_function, x)

    # Ensure non-negative values and normalize
    base_function = np.maximum(base_function, 0)
    if np.max(base_function) > 0:
        base_function = base_function / np.max(base_function) * 2.0

    # Apply smoothing to reduce sharp transitions
    window_size = min(51, max(3, n_steps // 100))
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        window = np.ones(window_size) / window_size
        base_function = np.convolve(base_function, window, mode='same')

    # Convert to list and apply final refinement
    step_values = base_function.tolist()

    # Phase 2: Multi-scale refinement using targeted optimization
    try:
        refined_values = refine_function_spectral_guided(step_values, n_steps)
        step_values = refined_values
    except Exception as e:
        warnings.warn(f"Refinement failed: {str(e)}")
        pass

    # Add final robustness noise
    final_noise = np.random.normal(0, 0.005, len(step_values))
    final_func = np.array(step_values) + final_noise
    final_func = np.maximum(final_func, 0)

    return final_func.tolist()

def generate_spectral_guided_peaks(n_steps: int) -> List[Tuple[float, float, float]]:
    """
    Generate peaks guided by spectral analysis to optimize autoconvolution properties.
    """
    x = np.linspace(-0.25, 0.25, n_steps)
    peaks = []

    # Create peaks with strategic distribution based on spectral optimization principles
    # Central region - dense peaks for smooth autoconvolution
    central_count = 12
    for i in range(central_count):
        pos = np.random.uniform(-0.1, 0.1)
        height = np.random.uniform(1.8, 2.5)
        width = np.random.uniform(0.015, 0.035)
        peaks.append((pos, height, width))

    # Mid-region - moderate density
    mid_count = 8
    for i in range(mid_count):
        pos = np.random.choice([-0.2, -0.18, -0.16, -0.14, -0.12, -0.1,
                               0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
        height = np.random.uniform(1.5, 2.2)
        width = np.random.uniform(0.025, 0.045)
        peaks.append((pos, height, width))

    # Outer region - sparse peaks
    outer_count = 5
    for i in range(outer_count):
        pos = np.random.choice([-0.24, -0.22, -0.2, 0.2, 0.22, 0.24])
        height = np.random.uniform(1.2, 1.8)
        width = np.random.uniform(0.035, 0.06)
        peaks.append((pos, height, width))

    return peaks

def add_supplementary_structure(f_values: np.ndarray, x: np.ndarray):
    """
    Add supplementary structure to enhance autoconvolution properties.
    """
    n_steps = len(f_values)
    for i in range(0, n_steps, max(1, n_steps//30)):
        if np.random.random() > 0.85:
            bump_center = x[i]
            bump_height = np.random.uniform(0.05, 0.2)
            bump_width = np.random.uniform(0.008, 0.018)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            f_values += bump

def refine_function_spectral_guided(initial_func: List[float], n_steps: int) -> List[float]:
    """
    Global optimization refinement approach using differential evolution.
    """
    x = np.linspace(-0.25, 0.25, n_steps)

    # Convert to array for easier manipulation
    initial_array = np.array(initial_func)

    # Define objective function for optimization
    def objective(params):
        # Reshape the flat parameters back to function values
        func_values = np.array(params)
        # Ensure non-negativity
        func_values = np.maximum(func_values, 0)
        return -compute_c2(func_values)  # Negative because we minimize

    # Use differential evolution for global optimization
    # Set bounds for each parameter (function values)
    bounds = [(0, np.max(initial_array) * 3) for _ in range(len(initial_array))]

    # Run differential evolution with appropriate settings for time constraints
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,      # Limit iterations to stay within time constraints
            popsize=10,      # Population size
            mutation=(0.5, 1.0),  # Mutation factor
            recombination=0.7,    # Crossover probability
            seed=42,
            polish=True,     # Improve final result
            disp=False       # Suppress output
        )

        # Return the optimized function values
        optimized_func = np.maximum(result.x, 0)
        return optimized_func.tolist()

    except Exception:
        # Fallback to original if optimization fails
        return initial_func

def compute_c2(func_vals: List[float]) -> float:
    """
    Compute C₂ value with optimized numerical integration.
    """
    f = np.array(func_vals)

    # Autoconvolution using convolution
    g = np.convolve(f, f, mode='full')
    g = g[len(g)//2:]  # Take middle portion

    # Adjust for correct length
    if len(g) > len(f):
        g = g[:len(f)]

    # Compute norms with improved numerical accuracy
    # ||g||₂² (L2 norm squared) - using piecewise quadratic integration
    dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5
    norm_2_sq = 0
    for i in range(len(g)-1):
        # Trapezoidal rule for better precision
        area = dx * (g[i]**2 + g[i+1]**2) / 2
        norm_2_sq += area

    # ||g||₁ (L1 norm)
    norm_1 = np.sum(np.abs(g)) * dx

    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g))

    # Prevent division by zero or extremely small values
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_2_sq / (norm_1 * norm_inf)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")