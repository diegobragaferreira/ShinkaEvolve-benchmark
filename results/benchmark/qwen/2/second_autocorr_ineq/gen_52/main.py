# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
from typing import List
import warnings

def compute_autoconvolution_norms(f: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f_arr = np.array(f)

    # Compute autoconvolution g = f * f
    g = signal.convolve(f_arr, f_arr, mode='full')

    # Adjust indexing for correct convolution
    g = g[len(f_arr)-1:]

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)

    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)

    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def gaussian_peak_function(x: np.ndarray, peak_positions: np.ndarray, peak_heights: np.ndarray) -> np.ndarray:
    """
    Create a function as sum of Gaussian peaks
    """
    result = np.zeros_like(x)
    sigma = 0.02  # Fixed standard deviation for Gaussian peaks

    for pos, height in zip(peak_positions, peak_heights):
        result += height * np.exp(-0.5 * ((x - pos) / sigma) ** 2)

    return result

def build_gaussian_step_function(peak_positions: np.ndarray, peak_heights: np.ndarray,
                               n_steps: int = 1000) -> List[float]:
    """
    Build a step function approximation from Gaussian peaks on [-1/4, 1/4]
    """
    # Define domain
    x = np.linspace(-0.25, 0.25, n_steps)

    # Generate Gaussian function
    func_values = gaussian_peak_function(x, peak_positions, peak_heights)

    # Ensure non-negativity
    func_values = np.maximum(func_values, 0)

    # Enhanced smoothing with optimized Gaussian kernel
    # Use larger sigma for better numerical stability while preserving shape
    func_values = gaussian_filter1d(func_values, sigma=1.0)

    return func_values.tolist()

def objective_function(params, n_steps: int = 1000) -> float:
    """
    Objective function for optimization - negative C2 to minimize
    """
    # Parse parameters: alternating peak positions and heights
    n_peaks = len(params) // 2
    peak_positions = params[:n_peaks]
    peak_heights = params[n_peaks:]

    # Scale peak positions to [-0.25, 0.25]
    peak_positions = np.array(peak_positions) * 0.5 - 0.25

    # Ensure peak heights are non-negative
    peak_heights = np.maximum(peak_heights, 0)

    # Build function
    try:
        func_values = build_gaussian_step_function(peak_positions, peak_heights, n_steps)

        # Compute C2
        c2 = compute_c2(func_values)

        # Minimize negative C2
        return -c2
    except Exception:
        # If there's an error, return very negative value
        return -1e10

def adaptive_gaussian_construction(n_peaks: int, n_steps: int = 1000, num_attempts: int = 5) -> List[float]:
    """
    Construct Gaussian peaks with adaptive strategies to maximize C2
    """
    best_c2 = -np.inf
    best_function = None

    for attempt in range(num_attempts):
        # Initialize random peak positions with strategic spacing
        # Ensure minimum distance between peaks to prevent interference
        peak_positions = np.random.uniform(-0.25, 0.25, n_peaks)

        # Enforce minimum spacing between peaks (at least 0.1 * domain width)
        min_spacing = 0.05  # 0.1 * 0.5
        if n_peaks > 1:
            for i in range(1, n_peaks):
                # Try to place peak i such that it's not too close to previous peaks
                while True:
                    new_pos = np.random.uniform(-0.25, 0.25)
                    min_dist = min([abs(new_pos - old_pos) for old_pos in peak_positions[:i]])
                    if min_dist >= min_spacing:
                        peak_positions[i] = new_pos
                        break

        # Initialize heights with adaptive scaling based on attempt number
        # Later attempts scale down heights to prevent overfitting to local maxima
        height_scale = 1.0 / (1.0 + 0.1 * attempt)
        initial_heights = np.random.exponential(1.0, n_peaks) * height_scale

        # Combine into single parameter vector
        initial_params = np.concatenate([peak_positions, initial_heights])

        # Set bounds: positions [-0.25, 0.25], heights [0, 20]
        bounds = []
        for _ in range(n_peaks):
            bounds.extend([(-0.25, 0.25)])  # Position bounds
        for _ in range(n_peaks):
            bounds.extend([(0, 20)])        # Height bounds

        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                args=(n_steps,),
                maxiter=150,
                popsize=15,
                tol=1e-6,
                seed=42 + attempt  # Different seed per attempt
            )

            if result.success:
                # Extract optimal parameters
                opt_positions = result.x[:n_peaks]
                opt_heights = result.x[n_peaks:]

                # Build final function
                final_func = build_gaussian_step_function(opt_positions, opt_heights, n_steps)

                # Evaluate C2 for this attempt
                c2 = compute_c2(final_func)

                if c2 > best_c2:
                    best_c2 = c2
                    best_function = final_func

        except Exception:
            continue

    # Fallback: if no optimization succeeded, create a structured approach
    if best_function is None:
        # Create a simpler structure with fewer peaks in a symmetric way
        x = np.linspace(-0.25, 0.25, n_steps)
        # Use a smooth symmetric distribution
        func_values = np.exp(-x**2 / 0.02)
        func_values = np.maximum(func_values, 0)
        # Smooth with a wider kernel for better numerical properties
        func_values = gaussian_filter1d(func_values, sigma=1.5)
        best_function = func_values.tolist()

    return best_function

def construct_function() -> List[float]:
    """
    Construct step function with high C2 value using enhanced Gaussian peak optimization
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Try different numbers of peaks to find good configuration
    best_c2 = 0.0
    best_function = []

    # Test different peak configurations - higher numbers for better optimization
    peak_configs = [5, 7, 10, 15, 20]

    for n_peaks in peak_configs:
        # Try with different numbers of steps to balance quality vs speed
        n_steps = min(2000, max(800, 1000 + n_peaks * 150))

        try:
            # Use adaptive construction approach with multiple attempts
            func = adaptive_gaussian_construction(n_peaks, n_steps, num_attempts=3)

            # Evaluate C2
            c2 = compute_c2(func)

            if c2 > best_c2:
                best_c2 = c2
                best_function = func

        except Exception as e:
            continue

    # If we didn't find anything, fall back to a simple approach
    if not best_function:
        # Create a simple symmetric step function
        n_steps = 1000
        func_values = np.ones(n_steps)
        # Apply a bit of smoothing and make it non-negative
        func_values = gaussian_filter1d(func_values, sigma=2.0)
        func_values = np.maximum(func_values, 0)
        best_function = func_values.tolist()

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")