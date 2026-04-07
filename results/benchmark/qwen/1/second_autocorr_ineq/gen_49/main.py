# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import differential_evolution
import time

def compute_autoconvolution_norms(f_values):
    """
    Compute the autoconvolution g = f*f and associated norms
    """
    try:
        # Convert to numpy array and ensure non-negative values
        f = np.array(f_values)
        f = np.maximum(f, 0)  # Clip negative values to 0

        if len(f) == 0:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4]
        step_width = 0.5 / len(f)

        # Autoconvolution using scipy's convolve (valid mode)
        g = convolve(f, f, mode='full')

        # Adjust indices for proper interval mapping
        # g corresponds to [-1/2, 1/2] interval, so we map to [-1/4, 1/4]
        g_center = len(g) // 2
        half_len = len(f)
        g_trimmed = g[g_center - half_len : g_center + half_len]

        # Compute norms
        # ||g||_2^2 using trapezoidal rule for piecewise linear integration
        g_abs = np.abs(g_trimmed)
        if len(g_abs) < 2:
            norm_2_squared = 0.0
        else:
            # Trapezoidal integration formula for piecewise linear segments
            # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
            widths = np.full(len(g_abs)-1, step_width)
            y1 = g_abs[:-1]
            y2 = g_abs[1:]
            norm_2_squared = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)

        # ||g||_1 = sum of absolute values divided by number of elements for normalization
        norm_1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-12

        # ||g||_∞ = max absolute value
        norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-12

        return norm_2_squared, norm_1, norm_inf

    except Exception as e:
        # Fallback to minimal values in case of computation errors
        return 0.0, 1e-12, 1e-12

def evaluate_c2(f_values):
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Prevent division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def sophisticated_initialization(n_steps):
    """
    Create initial population with sophisticated approaches:
    1. Alternating high/low pattern
    2. Gaussian-weighted peaks
    3. Uniform distribution
    """
    # Approach 1: Alternating pattern
    pattern1 = []
    for i in range(n_steps):
        pattern1.append(1.0 if i % 2 == 0 else 0.1)

    # Approach 2: Gaussian-shaped peaks with smooth transitions
    x = np.linspace(-1, 1, n_steps)
    peak1 = np.exp(-((x - 0.3)**2) / 0.1)
    peak2 = np.exp(-((x + 0.3)**2) / 0.1)
    pattern2 = np.maximum(peak1, peak2)

    # Approach 3: Simple uniform distribution
    pattern3 = [1.0] * n_steps

    # Return the best one based on initial evaluation
    patterns = [pattern1, pattern2.tolist(), pattern3]
    best_pattern = pattern3  # Default fallback
    best_score = -1.0

    for p in patterns:
        score = evaluate_c2(p)
        if score > best_score:
            best_score = score
            best_pattern = p

    return best_pattern

def evolutionary_optimization():
    """
    Main evolutionary optimization routine
    """
    # Parameters for optimization
    n_steps = np.random.randint(500, 3000)  # Variable length within reasonable bounds

    # Initialize population with sophisticated strategies
    initial_population = []
    for _ in range(50):  # Population size
        individual = sophisticated_initialization(n_steps)
        # Add some noise to diversity
        noise_factor = np.random.uniform(0.8, 1.2)
        individual = [max(0.0, val * noise_factor) for val in individual]
        initial_population.append(individual)

    # Define bounds for each variable (non-negative)
    bounds = [(0.0, 5.0)] * n_steps

    # Use differential evolution with custom bounds and settings
    try:
        # Set up problem with objective function
        def objective(x):
            return -evaluate_c2(x)  # Minimize negative to maximize C2

        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=50,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False
        )

        # Return the best solution found
        return result.x.tolist()

    except Exception:
        # Fallback to initial population with simple random search
        best_f = None
        best_c2 = -1.0

        # Try several random solutions
        for _ in range(100):
            f_values = sophisticated_initialization(n_steps)
            c2 = evaluate_c2(f_values)
            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values

        return best_f if best_f is not None else sophisticated_initialization(n_steps)

def construct_function() -> list[float]:
    """
    Main function to construct optimized step-function for high C2 value
    """
    # Allow some time budget for computation
    start_time = time.time()

    try:
        # Run evolutionary optimization
        f_values = evolutionary_optimization()

        # Final validation and cleanup
        f_values = np.array(f_values)
        f_values = np.maximum(f_values, 0)  # Ensure non-negative
        f_values = f_values.tolist()

        # If too long, truncate to reasonable size
        if len(f_values) > 5000:
            f_values = f_values[:5000]

        return f_values

    except Exception as e:
        # Return a fallback solution in case of any failure
        n_steps = 500
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")