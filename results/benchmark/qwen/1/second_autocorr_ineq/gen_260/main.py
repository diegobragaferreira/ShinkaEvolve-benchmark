# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import differential_evolution
from typing import List, Tuple
import random

# Optimized autoconvolution and norm computation using Numba
@jit(nopython=True)
def compute_autoconvolution_numba(f_vals: np.ndarray) -> np.ndarray:
    """Efficiently compute autoconvolution using Numba JIT compilation"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)
    
    # Manual convolution loop for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_norms_numba(g_vals: np.ndarray) -> Tuple[float, float, float]:
    """Compute L1, L2^2, and L-infinity norms efficiently"""
    # L1 norm (sum of absolute values)
    l1_norm = 0.0
    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    
    for i in range(len(g_vals)):
        abs_val = abs(g_vals[i])
        l1_norm += abs_val
        l2_sq_norm += g_vals[i] * g_vals[i]
        if abs_val > linf_norm:
            linf_norm = abs_val
    
    return l1_norm, l2_sq_norm, linf_norm

@jit(nopython=True)
def compute_c2_numba(f_vals: np.ndarray) -> float:
    """Compute C2 value using optimized Numba functions"""
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals)
    
    # Compute norms
    l1, l2_sq, linf = compute_norms_numba(g_vals)
    
    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0
    
    # Return C2 value
    return l2_sq / (l1 * linf)

def compute_autoconvolution_norms_manual(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂²
    """
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = np.convolve(f, f, mode='full')
    
    # Compute the norms
    # ||g||₂² using proper piecewise integration
    # For consecutive pairs (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    norm_g_2_squared = 0.0
    for i in range(len(g) - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    
    # ||g||₁ = sum(|g[i]|)
    norm_g_1 = np.sum(np.abs(g))
    
    # ||g||∞ = max(|g[i]|)
    norm_g_inf = np.max(np.abs(g))
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_manual(f_values)

        # Avoid division by zero with stricter thresholds
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def create_bell_shaped_pattern(n_steps: int) -> np.ndarray:
    """Create a bell-shaped pattern emphasizing edges"""
    x = np.linspace(0, 1, n_steps)
    # Gaussian-like shape with emphasis on edges
    pattern = (1.0 + 0.8 * np.exp(-15 * (x - 0.5)**2) -
              0.3 * np.exp(-5 * x**2) - 0.3 * np.exp(-5 * (1-x)**2))
    return np.clip(pattern, 0, np.inf)

def create_alternating_pattern(n_steps: int) -> np.ndarray:
    """Create alternating high/low pattern"""
    pattern = []
    for i in range(n_steps):
        if i % 2 == 0:
            pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
        else:
            pattern.append(max(0.0, 0.1 + np.random.normal(0, 0.05)))
    return np.array(pattern)

def create_peak_centered_pattern(n_steps: int) -> np.ndarray:
    """Create peak-centered pattern with tapering edges"""
    pattern = np.zeros(n_steps)
    center = n_steps // 2
    width = max(1, n_steps // 6 + np.random.randint(-1, 2))

    # Create a central peak
    pattern[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0

    # Add tapering to edges
    for i in range(center - width//2):
        pattern[i] *= (i / (center - width//2))
    for i in range(center + width//2, n_steps):
        pattern[i] *= ((n_steps - i) / (width//2 + 1))

    # Add some noise
    noise = np.random.normal(0, 0.05, n_steps)
    pattern = pattern + noise
    return np.clip(pattern, 0, np.inf)

def create_smooth_transition_pattern(n_steps: int) -> np.ndarray:
    """Create smooth transition pattern"""
    pattern = np.zeros(n_steps)
    # Create smooth ramp with some random variation
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        pattern[i] = 0.5 + 0.5 * np.sin(np.pi * x) + np.random.normal(0, 0.1)
    return np.clip(pattern, 0, np.inf)

def create_multi_scale_initialization(n_steps: int) -> np.ndarray:
    """Create diverse initial solution using multiple strategies"""
    strategies = [
        create_bell_shaped_pattern,
        create_alternating_pattern,
        create_peak_centered_pattern,
        create_smooth_transition_pattern
    ]

    # Choose a strategy randomly
    strategy = np.random.choice(strategies)
    pattern = strategy(n_steps)
    return pattern / np.sum(pattern) * n_steps

def adaptive_evolutionary_optimization(initial_solution: List[float], max_time_seconds: float = 80.0) -> List[float]:
    """Enhanced evolutionary optimization with adaptive population sizing and early stopping"""
    # Track convergence
    best_scores = []
    patience_counter = 0
    max_patience = 10
    population_size = 15

    # Start with initial solution
    current_solution = initial_solution.copy()
    current_c2 = evaluate_c2(current_solution)
    best_scores.append(current_c2)

    # Time tracking
    start_time = time.time()
    time_limit = max_time_seconds  # Leave some buffer

    # Adaptive parameters based on convergence behavior
    for generation in range(100):  # Limited to prevent timeout
        # Early termination check
        if time.time() - start_time > time_limit:
            break

        # Check for convergence
        if len(best_scores) >= 5:
            recent_improvement = best_scores[-1] - best_scores[-5]
            if recent_improvement < 1e-6:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= max_patience:
                # Increase population size to escape local minimum
                population_size = min(population_size * 2, 50)
                patience_counter = 0

        # Define bounds for differential evolution
        bounds = [(0.0, 10.0)] * len(current_solution)

        # Run differential evolution with adaptive parameters
        try:
            result = differential_evolution(
                lambda x: -evaluate_c2(x),  # Negative because we want to maximize
                bounds,
                maxiter=5,  # Fewer iterations per generation for speed
                popsize=population_size,
                seed=42 + generation,
                strategy='best1bin',
                tol=1e-6,
                recombination=0.7,
                disp=False
            )

            if result.success:
                new_solution = result.x.tolist()
                new_c2 = evaluate_c2(new_solution)

                if new_c2 > current_c2:
                    current_solution = new_solution
                    current_c2 = new_c2
                    best_scores.append(current_c2)

        except Exception:
            pass  # Continue with current solution if optimization fails

    return current_solution

def hybrid_local_search(initial_solution: List[float]) -> List[float]:
    """Combine global and local search strategies"""
    # First, run the adaptive evolutionary optimization
    evolved_solution = adaptive_evolutionary_optimization(initial_solution, max_time_seconds=60.0)

    # Then apply a simple gradient-like local refinement
    refined_solution = evolved_solution.copy()
    
    # Simple gradient ascent approach
    for i in range(len(refined_solution)):
        # Try small positive perturbations
        original_value = refined_solution[i]
        perturbations = [0.01, 0.05, 0.1]

        best_value = original_value
        best_c2 = evaluate_c2(refined_solution)

        for delta in perturbations:
            # Try increasing the value
            test_solution = refined_solution.copy()
            test_solution[i] = max(0, original_value + delta)

            c2_test = evaluate_c2(test_solution)
            if c2_test > best_c2:
                best_c2 = c2_test
                best_value = original_value + delta

            # Try decreasing the value
            test_solution = refined_solution.copy()
            test_solution[i] = max(0, original_value - delta)

            c2_test = evaluate_c2(test_solution)
            if c2_test > best_c2:
                best_c2 = c2_test
                best_value = original_value - delta

        refined_solution[i] = best_value

    return refined_solution

def multi_scale_optimization() -> List[float]:
    """Perform enhanced multi-scale optimization with adaptive strategies"""
    # Initialize with multiple random samples
    best_solution = None
    best_c2 = -float('inf')

    # Time tracking
    start_time = time.time()

    # Try several different initializations with different strategies
    for attempt in range(15):  # Increased attempts
        # Early termination check
        if time.time() - start_time > 80.0:  # 80 seconds for computation
            break

        # Create diverse initial solution
        n_steps = np.random.randint(100, 1000)

        # Use different initialization strategies
        if attempt % 3 == 0:
            # Use multi-scale initialization
            initial_solution = create_multi_scale_initialization(n_steps)
        elif attempt % 3 == 1:
            # Use random initialization
            initial_solution = np.random.random(n_steps) + 0.1
            initial_solution = initial_solution / np.sum(initial_solution) * n_steps
        else:
            # Use structured initialization
            x = np.linspace(-1, 1, n_steps)
            pattern = 0.7 + 0.3 * np.sin(3 * np.pi * x)
            initial_solution = pattern / np.sum(pattern) * n_steps

        # Optimize this initialization with hybrid approach
        optimized_solution = hybrid_local_search(initial_solution.tolist())

        # Evaluate result
        c2 = evaluate_c2(optimized_solution)

        if c2 > best_c2:
            best_c2 = c2
            best_solution = optimized_solution

    # Final check of time limit
    if time.time() - start_time > 85.0:
        # Last resort: return the best solution found so far
        pass

    return best_solution if best_solution is not None else [1.0] * 100

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses enhanced modular optimization approach with multiple strategies.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Set start time
    start_time = time.time()

    try:
        # Use enhanced multi-scale optimization approach
        best_solution = multi_scale_optimization()

        # Final evaluation
        final_c2 = evaluate_c2(best_solution)

        end_time = time.time()
        eval_time = end_time - start_time

        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {final_c2:.6f}")

        return best_solution

    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * 100
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")