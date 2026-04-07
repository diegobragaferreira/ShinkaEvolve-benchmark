# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
import numba
from numba import jit
import time
from typing import List

# JIT compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    Implements proper piecewise linear integration as required:
    for consecutive pairs of points (y1, y2) with unit spacing:
    integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
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
    
    # Compute the norms using the specified piecewise integration method
    # ||g||₂² = sum of (1/3)(y1² + y1*y2 + y2²) for consecutive pairs
    norm_g_2_squared = 0.0
    
    # For piecewise linear integration, we use trapezoidal-like approach:
    # for consecutive pairs of points (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    
    # ||g||₁ = sum(|g[i]|) - normalized by the length for consistent comparison
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
    except Exception:
        # Fallback in case of any numerical issues
        return 0.0

def generate_diverse_initializations(n_steps: int) -> List[List[float]]:
    """
    Generate diverse initial configurations to enhance exploration
    """
    initializations = []
    
    # Pattern 1: Multi-peak Gaussian with varying scales and positions
    x = np.linspace(-1, 1, n_steps)
    pattern1 = np.zeros(n_steps)
    for i in range(3):  # Three peaks
        center = -0.5 + i * 0.5
        width = 0.15 + np.random.random() * 0.2
        height = 0.7 + np.random.random() * 0.5
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    initializations.append(pattern1.tolist())

    # Pattern 2: Alternating high/low segments with smooth transitions
    pattern2 = []
    for i in range(n_steps):
        base_val = 1.0 if i % 3 == 0 else 0.2 if i % 3 == 1 else 0.6
        pattern2.append(base_val + np.random.random() * 0.3)
    initializations.append(pattern2)

    # Pattern 3: Center-heavy with Gaussian decay
    pattern3 = []
    center = n_steps // 2
    for i in range(n_steps):
        distance = abs(i - center) / (n_steps // 2)
        val = max(0.0, 1.0 * np.exp(-2 * distance**2))
        pattern3.append(val + 0.1 * np.random.random())
    initializations.append(pattern3)

    # Pattern 4: Sinusoidal with modulation
    pattern4 = []
    for i in range(n_steps):
        x_pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        base = 0.9 + 0.2 * np.sin(6 * np.pi * x_pos)
        mod = 0.15 * np.cos(10 * np.pi * x_pos)
        pattern4.append(max(0.0, base + mod))
    initializations.append(pattern4)

    # Pattern 5: Sparse spikes with varying heights
    pattern5 = np.zeros(n_steps)
    positions = [0.1, 0.25, 0.5, 0.75, 0.9]
    for pos in positions:
        pattern5 += np.exp(-((x - pos)**2) / 0.015) * (1.0 + np.random.random() * 0.5)
    initializations.append(pattern5.tolist())

    return initializations

def adaptive_evolutionary_search(n_steps: int, max_time: float = 85.0) -> List[float]:
    """
    Multi-phase evolutionary optimization with adaptive parameters
    """
    start_time = time.time()
    best_score = -np.inf
    best_individual = None

    # Create diverse initial population
    initializations = generate_diverse_initializations(n_steps)

    # Evaluate all initializations and find the best
    for init in initializations:
        try:
            score = evaluate_c2(init)
            if score > best_score:
                best_score = score
                best_individual = init.copy()
        except:
            continue

    # If no valid initialization found, use a simple pattern
    if best_individual is None:
        best_individual = [1.0] * n_steps

    # Phase 1: Global search with differential evolution (coarse)
    def objective(f_vals):
        return -evaluate_c2(np.array(f_vals))  # Negative because we minimize

    bounds = [(0.0, 2.0) for _ in range(n_steps)]

    # Adaptive evolutionary parameters based on iteration and elapsed time
    iteration = 0
    max_iterations = 100
    time_left = max_time - 5  # Reserve some time for cleanup

    while time.time() - start_time < time_left and iteration < max_iterations:
        # Adaptive population size calculation
        base_popsize = max(10, min(50, n_steps // 20))
        adaptive_popsize = base_popsize + int(iteration * 2)  # Gradually increase population size

        # Adaptive mutation rate that decreases over time
        adaptive_mutation = max(0.3, 0.8 - (iteration * 0.02))

        # Adaptive maxiter based on time left
        remaining_time = max_time - (time.time() - start_time)
        adaptive_maxiter = max(10, min(50, int(remaining_time / 3)))

        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=int(time.time() + iteration * 1000),
                maxiter=adaptive_maxiter,
                popsize=adaptive_popsize,
                mutation=(adaptive_mutation, 1.0),
                recombination=0.7,
                tol=1e-6,
                disp=False
            )

            current_score = evaluate_c2(result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = result.x.tolist()

        except Exception as e:
            pass

        iteration += 1

    # Phase 2: Refinement with multiple techniques
    if time.time() - start_time < max_time - 3:
        # Try L-BFGS-B for local refinement
        try:
            refined_result = minimize(
                objective,
                best_individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            current_score = evaluate_c2(refined_result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = refined_result.x.tolist()
        except Exception as e:
            pass

        # Try Nelder-Mead as backup
        try:
            nm_result = minimize(
                objective,
                best_individual,
                method='Nelder-Mead',
                options={'maxiter': 30}
            )
            current_score = evaluate_c2(nm_result.x)
            if current_score > best_score:
                best_score = current_score
                best_individual = nm_result.x.tolist()
        except Exception as e:
            pass

    return best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using enhanced hybrid optimization approach."""
    # Use a larger number of steps for better resolution
    n_steps = 2000

    # Use adaptive evolutionary search
    try:
        final_values = adaptive_evolutionary_search(n_steps, max_time=85.0)
    except Exception:
        # Fallback to simple approach if anything fails
        final_values = [1.0] * n_steps

    # Post-processing: ensure non-negative values and normalize
    final_values = np.clip(final_values, 0, None)
    total = np.sum(final_values)
    if total > 0:
        final_values = final_values / total * 2.0

    # Convert to list of floats
    return [float(x) for x in final_values]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")