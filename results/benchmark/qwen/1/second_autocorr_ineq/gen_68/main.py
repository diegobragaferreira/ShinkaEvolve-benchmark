# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
import numba
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# JIT compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    n = len(f_values)

    # Compute autoconvolution g = f * f using discrete convolution
    # The resulting g will have length 2*n - 1 where n is the length of f
    g_length = 2 * n - 1
    g = np.zeros(g_length)

    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Compute the norms with improved integration
    # ||g||₂² = sum of (g[i]^2 + g[i]*g[i+1] + g[i+1]^2)/3
    # This is a more accurate trapezoidal-like integration for quadratic function
    norm_g_2_squared = 0.0

    # Trapezoidal integration for g^2: for interval [x_i, x_{i+1}] with values g_i, g_{i+1}
    # integral of g^2 dx ≈ (Δx/3)(g_i^2 + g_i*g_{i+1} + g_{i+1}^2) where Δx = 1
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # For better numerical stability, also consider the actual trapezoidal rule for ||g||₁
    # But since we're dealing with sums, we'll keep the original approach but with better accuracy
    norm_g_1 = 0.0
    for i in range(g_length):
        norm_g_1 += abs(g[i])

    # ||g||∞ = max(|g[i]|)
    norm_g_inf = 0.0
    for i in range(g_length):
        abs_g = abs(g[i])
        if abs_g > norm_g_inf:
            norm_g_inf = abs_g

    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂² as specified in requirements
    """
    f = np.array(f_values)

    # Use the fast JIT-compiled version
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_fast(f)

    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception as e:
        # Fallback in case of any numerical issues
        return 0.0

def generate_initial_population(n_individuals: int, n_steps: int) -> np.ndarray:
    """
    Generate diverse initial population for evolutionary algorithm
    """
    population = []

    # Create various types of initial configurations
    for i in range(n_individuals):
        # Type 1: Gaussian-like distribution
        if i % 4 == 0:
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.2 + np.random.random() * 0.3
            mu = np.random.random() * 0.5 - 0.25  # Center around -0.25 to 0.25
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            # Normalize
            f = f / np.sum(f)
            population.append(f)

        # Type 2: Uniform distribution with some randomness
        elif i % 4 == 1:
            f = np.random.random(n_steps)
            # Add some structure
            f = np.clip(f, 0, 1)
            f = f / np.sum(f)
            population.append(f)

        # Type 3: Peak centered distribution
        elif i % 4 == 2:
            f = np.zeros(n_steps)
            center = n_steps // 2
            width = max(1, n_steps // 8 + np.random.randint(-2, 3))
            f[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0
            # Add some noise
            f += np.random.normal(0, 0.03, n_steps)
            f = np.clip(f, 0, None)
            f = f / np.sum(f)
            population.append(f)

        # Type 4: Alternating segments with smooth transitions
        else:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 6)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    f[j:end_idx] = 0.7 + np.random.random(end_idx - j) * 0.3
                else:
                    f[j:end_idx] = 0.2 + np.random.random(end_idx - j) * 0.2

            # Smooth with Gaussian
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.3)**2)
            f = f * gaussian * 0.4 + gaussian * 0.6

            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f)
            population.append(f)

    return np.array(population)

def evolutionary_optimization() -> List[float]:
    """
    Use evolutionary algorithm to optimize step function
    """
    n_steps = 500  # Reasonable size for exploration

    # Define bounds for each parameter (step height)
    bounds = [(0, 1.0) for _ in range(n_steps)]

    def objective(x):
        # Return negative because we want to maximize C2
        return -evaluate_c2(x.tolist())

    # Use differential evolution for global optimization
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,  # Reduced iterations for faster execution
            popsize=12,   # Smaller population for speed
            seed=42,
            disp=False
        )

        if result.success:
            optimized_f = np.maximum(result.x, 0)
            # Normalize to ensure good scaling
            if np.sum(optimized_f) > 0:
                optimized_f = optimized_f / np.sum(optimized_f)

            # Post-process with local optimization to refine the solution
            try:
                # Local refinement with L-BFGS
                def local_objective(f_vals):
                    return -evaluate_c2(f_vals.tolist())

                bounds_local = [(0, 1.0) for _ in range(n_steps)]
                local_result = minimize(
                    local_objective,
                    optimized_f,
                    method='L-BFGS-B',
                    bounds=bounds_local,
                    options={'maxiter': 20}
                )

                if local_result.success:
                    refined_f = np.maximum(local_result.x, 0)
                    if np.sum(refined_f) > 0:
                        refined_f = refined_f / np.sum(refined_f)
                    return refined_f.tolist()
            except:
                pass

            return optimized_f.tolist()
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Return default if optimization fails
    return [1.0/n_steps] * n_steps

def sophisticated_initialization() -> List[float]:
    """
    Generate a sophisticated initial configuration based on mathematical intuition
    """
    n_steps = 500

    # Create a step function that tries to balance flatness with sufficient mass
    # Based on mathematical insights: we want to create a function that when convolved
    # produces a relatively flat profile but with enough energy to achieve high C2

    # Start with alternating high/low regions with some randomness
    f = np.zeros(n_steps)

    # First create a base alternating pattern with some randomness
    segment_size = max(1, n_steps // 10)
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region
            f[i:end_idx] = 0.8 + np.random.random(end_idx - i) * 0.2
        else:
            # Low region
            f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.1

    # Add Gaussian-based smoothing for more natural transitions
    x = np.linspace(-1, 1, n_steps)
    gaussian = np.exp(-0.5 * (x / 0.25)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4

    # Add some noise to break symmetry
    noise = np.random.normal(0, 0.02, n_steps)
    f = f + noise

    # Ensure non-negativity
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using improved methods
    """
    try:
        # Try sophisticated initialization first
        initial_f = sophisticated_initialization()
        c2_initial = evaluate_c2(initial_f)

        # Then run evolutionary optimization
        optimized_f = evolutionary_optimization()
        c2_optimized = evaluate_c2(optimized_f)

        # Return the better of the two
        if c2_optimized > c2_initial:
            return optimized_f
        else:
            return initial_f

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")