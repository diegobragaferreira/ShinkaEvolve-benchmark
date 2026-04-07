# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import time
from typing import List, Tuple
import numba
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_numba(f_values: np.ndarray) -> np.ndarray:
    """
    Fast computation of autoconvolution using numba-compiled loop
    """
    n = len(f_values)
    g = np.zeros(2*n - 1)

    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    return g

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the autoconvolution g = f*f and associated norms efficiently
    """
    try:
        # Convert to numpy array and ensure non-negative values
        f = np.array(f_values, dtype=np.float64)
        f = np.maximum(f, 0)  # Clip negative values to 0

        if len(f) == 0:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4]
        step_width = 0.5 / len(f)

        # Autoconvolution using fast numba-compiled function
        g = compute_autoconvolution_numba(f)

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

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Prevent division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def sophisticated_initialization(n_steps: int) -> List[float]:
    """
    Create initial population with sophisticated approaches:
    1. Alternating high/low pattern
    2. Gaussian-shaped peaks with smooth transitions
    3. Uniform distribution
    4. Optimized alternating pattern with better spacing
    5. Multi-peak pattern
    """
    # Approach 1: Alternating pattern with varying amplitudes
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

    # Approach 4: Optimized alternating pattern with better spacing
    pattern4 = []
    for i in range(n_steps):
        if i % 4 == 0 or i % 4 == 1:
            pattern4.append(1.0)
        else:
            pattern4.append(0.1)

    # Approach 5: Multi-peak pattern with varied heights
    pattern5 = np.zeros(n_steps)
    peak_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for pos in peak_positions:
        pattern5 += np.exp(-((x - pos)**2) / 0.05) * 0.8

    patterns = [pattern1, pattern2.tolist(), pattern3, pattern4, pattern5.tolist()]
    best_pattern = pattern3  # Default fallback
    best_score = -1.0

    for p in patterns:
        score = evaluate_c2(p)
        if score > best_score:
            best_score = score
            best_pattern = p

    return best_pattern

def adaptive_population_sizing(n_steps: int, iteration: int) -> int:
    """
    Dynamically adjust population size based on problem characteristics and iteration
    """
    base_pop = min(50, max(10, n_steps // 20))
    # Increase population size slightly in early iterations for better exploration
    if iteration < 2:
        return min(100, base_pop * 2)
    else:
        return base_pop

def advanced_refinement_strategy(best_solution: List[float], n_steps: int, bounds: List[Tuple[float, float]]) -> List[float]:
    """
    Enhanced refinement that combines multiple optimization techniques
    """
    try:
        # First, try L-BFGS-B with the current solution as starting point
        def objective(x):
            return -evaluate_c2(x)

        # Local refinement with L-BFGS-B
        ref_result = minimize(
            objective,
            best_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30}
        )

        refined_solution = ref_result.x.tolist()

        # Additional refinement with Nelder-Mead if L-BFGS didn't converge well
        if evaluate_c2(refined_solution) < 0.8 * evaluate_c2(best_solution):
            try:
                nm_result = minimize(
                    objective,
                    best_solution,
                    method='Nelder-Mead',
                    options={'maxiter': 20}
                )
                if evaluate_c2(nm_result.x.tolist()) > evaluate_c2(refined_solution):
                    refined_solution = nm_result.x.tolist()
            except:
                pass  # Keep previous result if Nelder-Mead fails

        return refined_solution

    except Exception:
        return best_solution

def evolutionary_optimization(max_time_seconds: float = 85.0) -> List[float]:
    """
    Main evolutionary optimization routine with enhanced features
    """
    start_time = time.time()

    # Parameters for optimization - wider range for more exploration
    n_steps_range = [200, 3000]
    n_steps = np.random.randint(n_steps_range[0], n_steps_range[1])

    # Define bounds for each variable (non-negative)
    bounds = [(0.0, 5.0)] * n_steps

    # Initialize population with multi-scale approach
    initial_population = []
    population_size = min(50, max(10, n_steps // 20))  # Base population size

    # Create diverse starting points with multi-scale initialization
    for _ in range(population_size):
        individual = sophisticated_initialization(n_steps)
        # Add controlled noise for diversity
        noise_factor = np.random.uniform(0.8, 1.2)
        individual = [max(0.0, val * noise_factor) for val in individual]
        initial_population.append(individual)

    # Use differential evolution with enhanced parameters and adaptive strategies
    try:
        # Set up problem with objective function
        def objective(x):
            return -evaluate_c2(x)  # Minimize negative to maximize C2

        remaining_time = max_time_seconds - (time.time() - start_time)
        maxiter = max(100, int(remaining_time / 2))

        # Multi-start differential evolution with adaptive parameters
        best_solution = None
        best_score = -1.0

        # Try multiple random starts to improve chances of finding global optimum
        max_starts = min(8, max(2, int(remaining_time / 8)))
        for start_iter in range(max_starts):
            if time.time() - start_time > max_time_seconds - 2.0:
                break

            # Adaptive population sizing based on iteration
            popsize = adaptive_population_sizing(n_steps, start_iter)

            # Vary mutation rate based on iteration
            mutation_rate = max(0.3, min(1.0, 0.6 + start_iter * 0.1))

            # Randomize the seed for different starts
            np.random.seed(int(time.time()) + start_iter * 1000)

            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                mutation=(mutation_rate, 1.0),
                recombination=0.7,
                seed=None,  # Let it use random seed
                disp=False
            )

            # Check if this is better
            current_score = evaluate_c2(result.x)
            if current_score > best_score:
                best_score = current_score
                best_solution = result.x.tolist()

        # If we run out of time, use the best individual from initial population
        if time.time() - start_time > max_time_seconds - 1.0:
            # Get best from initial population
            best_initial = max(initial_population, key=evaluate_c2)
            return best_initial

        # Advanced refinement for the best solution
        if best_solution is not None and time.time() - start_time < max_time_seconds - 3.0:
            refined_solution = advanced_refinement_strategy(best_solution, n_steps, bounds)
            # Use refined solution if it performs better
            if evaluate_c2(refined_solution) > best_score:
                best_solution = refined_solution

        # Return the best solution found
        if best_solution is not None:
            return best_solution

        # Fallback to initial population if no good solution found
        best_initial = max(initial_population, key=evaluate_c2)
        return best_initial

    except Exception as e:
        # Fallback to initial population with simple random search
        best_f = None
        best_c2 = -1.0

        # Try several random solutions
        attempts = min(200, n_steps * 3)
        for _ in range(attempts):
            if time.time() - start_time > max_time_seconds - 1.0:
                break
            f_values = sophisticated_initialization(n_steps)
            c2 = evaluate_c2(f_values)
            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values

        if best_f is not None:
            return best_f
        else:
            return sophisticated_initialization(n_steps)

def construct_function() -> List[float]:
    """
    Main function to construct optimized step-function for high C2 value
    """
    # Allow some time budget for computation (leave 5 seconds for cleanup)
    start_time = time.time()

    try:
        # Run evolutionary optimization with time constraint
        f_values = evolutionary_optimization(max_time_seconds=85.0)

        # Final validation and cleanup
        f_values = np.array(f_values, dtype=np.float64)
        f_values = np.maximum(f_values, 0)  # Ensure non-negative
        f_values = f_values.tolist()

        # If too long, truncate to reasonable size but maintain minimum length
        if len(f_values) > 5000:
            f_values = f_values[:5000]
        elif len(f_values) < 100:
            f_values = f_values + [0.0] * (100 - len(f_values))

        return f_values

    except Exception as e:
        # Return a fallback solution in case of any failure
        n_steps = 500
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")