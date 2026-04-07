# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import time
from joblib import Parallel, delayed
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# JIT compile the core computation functions for speed
@numba.jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2 * n - 1)

    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@numba.jit(nopython=True)
def compute_norms_piecewise(g_vals):
    """Compute norms using piecewise linear integration matching evaluator's method"""
    n = len(g_vals)

    if n <= 1:
        return 0.0, 0.0, 0.0

    # Compute L2 norm squared using trapezoidal-like integration
    # Formula: (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    norm_2_sq = 0.0
    dx = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.5

    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        norm_2_sq += (dx / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

    # Compute L1 norm (sum of absolute values)
    norm_1 = 0.0
    for i in range(n):
        norm_1 += abs(g_vals[i])

    # Compute L-infinity norm (maximum absolute value)
    norm_inf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > norm_inf:
            norm_inf = abs_val

    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation using efficient piecewise integration.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f_arr = np.array(f, dtype=np.float64)

    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_arr)

    # Compute norms using piecewise integration
    norm_2_sq, norm_1, norm_inf = compute_norms_piecewise(g)

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_structured_step_function(n_steps: int) -> List[float]:
    """Create a structured step function with optimized Gaussian peaks and step patterns"""
    # Create base function with multiple Gaussian peaks
    f_vals = np.zeros(n_steps)

    # Use more sophisticated peak spacing strategy for optimal distribution
    # Generate peaks using a combination of geometric and quasi-random distribution
    n_peaks = max(3, min(12, n_steps // 50))

    # Use golden ratio distribution for better spread
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    peak_positions = []

    # Generate positions using golden ratio method to avoid clustering
    for i in range(n_peaks):
        # Use golden ratio to distribute peaks pseudo-randomly but evenly
        pos = 0.05 * n_steps + (i * phi) % 1.0 * 0.9 * n_steps
        peak_positions.append(pos)

    # Ensure minimum gap between peaks (0.05 * domain width)
    min_gap = 0.05 * n_steps
    filtered_positions = []
    for pos in peak_positions:
        # Only add if sufficiently far from existing peaks
        if not filtered_positions or min(abs(pos - existing) for existing in filtered_positions) >= min_gap:
            filtered_positions.append(pos)

    # Add Gaussian peaks with controlled amplitude adjustments based on intermediate feedback
    for i, center in enumerate(filtered_positions):
        # Width inversely proportional to peak importance (smaller for sharper peaks)
        width = max(5, min(100, n_steps / (2 * np.sqrt(2 * np.log(2)))))

        # Dynamic height adjustment to avoid overly sharp autoconvolution
        # Start with base height and reduce for later peaks to maintain balance
        base_height = random.uniform(1.0, 3.0)
        height_factor = 1.0 - 0.1 * i / max(1, len(filtered_positions) - 1)
        height = base_height * height_factor

        # Generate Gaussian curve
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Add some step-like patterns for additional structure
    if n_steps > 100:
        n_steps_regions = min(6, max(2, n_steps // 100))
        for i in range(n_steps_regions):
            start_idx = int(i * n_steps / n_steps_regions)
            end_idx = int((i + 1) * n_steps / n_steps_regions)
            if i % 2 == 0:
                f_vals[start_idx:end_idx] += random.uniform(0.3, 1.0)

    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)

    # Apply mild smoothing to avoid extreme variations
    if n_steps > 20:
        kernel = np.ones(5) / 5
        f_vals = np.convolve(f_vals, kernel, mode='same')

    # Normalize to reasonable scale - but maintain flexibility for optimization
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 2.0

    return f_vals.tolist()

def create_simple_step_function(n_steps: int) -> List[float]:
    """Create a simple step function with random heights"""
    # Create step function with varying heights
    heights = []
    n_steps_per_region = max(1, n_steps // 20)

    for i in range(min(20, n_steps // n_steps_per_region)):
        region_height = random.uniform(0.5, 2.0)
        for _ in range(n_steps_per_region):
            if len(heights) < n_steps:
                heights.append(region_height)

    # Pad or truncate to exact length
    if len(heights) < n_steps:
        heights.extend([random.uniform(0.5, 2.0)] * (n_steps - len(heights)))
    elif len(heights) > n_steps:
        heights = heights[:n_steps]

    return heights

def adaptive_step_function_initialization(n_steps: int) -> List[float]:
    """
    Create initial step function with adaptive construction using multiple strategies
    """
    # Use different initialization strategies based on problem size
    if n_steps < 200:
        # For small functions, use simple approach
        return create_simple_step_function(n_steps)
    else:
        # For larger functions, use structured approach
        return create_structured_step_function(n_steps)

def local_search_refinement(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Apply enhanced local search to improve the function by focusing on critical features
    """
    f_current = np.array(initial_f, dtype=np.float64)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()

    # Enhanced local search with feature-aware perturbations
    for iteration in range(max_iter):
        # Create neighbor by making small changes
        f_new = f_current.copy()

        # Identify important regions: detect peaks and valleys in the current function
        # For a more targeted approach, we focus on modifying key characteristics
        # such as amplitude of peak regions

        # Sample approach: modify 10% of the largest values to avoid over-perturbing
        # the entire function space
        sorted_indices = np.argsort(f_new)[::-1]  # indices sorted by value (descending)
        indices_to_modify = sorted_indices[:max(5, len(f_new) // 10)]

        # More focused perturbations around significant features
        for idx in indices_to_modify:
            # Use larger perturbations near significant peaks to explore effectively
            if f_new[idx] > np.percentile(f_new, 75):
                # Large perturbation for significant peaks
                perturbation = np.random.normal(0, 0.1 * f_new[idx])
            elif f_new[idx] > np.percentile(f_new, 50):
                # Medium perturbation for mid-range values
                perturbation = np.random.normal(0, 0.05 * f_new[idx])
            else:
                # Small perturbation for background levels
                perturbation = np.random.normal(0, 0.02 * f_new[idx]) if f_new[idx] > 0 else np.random.normal(0, 0.05)

            f_new[idx] = max(0, f_new[idx] + perturbation)

        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())

        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()

        f_current = f_new

    return best_f.tolist()

def differential_evolution_refinement(initial_f: List[float], max_evals: int = 300) -> List[float]:
    """
    Use differential evolution for global refinement
    """
    try:
        # Convert individual to array for optimization
        x0 = np.array(initial_f, dtype=np.float64)

        # Define bounds for each parameter (clamped between 0 and 5)
        bounds = [(0, 5) for _ in range(len(x0))]

        # Objective function for differential evolution
        def obj_func(x):
            # Ensure non-negative values
            x = np.maximum(x, 0)
            # Evaluate it
            score = compute_c2(x.tolist())
            # Minimize negative of score (since we want to maximize)
            return -score if score > 0 else 1e10

        # Run differential evolution with fewer evaluations to save time
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=max_evals,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )

        if result.success:
            refined = np.maximum(result.x, 0).tolist()
            # Verify the result
            score = compute_c2(refined)
            if score > compute_c2(initial_f):
                return refined

    except Exception as e:
        pass

    return initial_f

def evaluate_candidate(individual: List[float]) -> float:
    """Evaluate a single candidate function"""
    return compute_c2(individual)

def construct_function() -> List[float]:
    """
    Construct step function with high C2 value using adaptive optimization approach
    """
    start_time = time.time()

    # Set up parameters
    max_time_seconds = 85

    # Try multiple random initializations with different strategies
    best_c2 = 0.0
    best_function = []

    # Multi-start approach with different population sizes
    population_sizes = [30, 50, 70]

    # Evaluate multiple candidate functions in parallel
    all_candidates = []

    for pop_size in population_sizes:
        for i in range(pop_size):
            # Create function with adaptive initialization
            n_steps = max(100, min(5000, 800 + i * 50))  # Vary number of steps

            # Create initial function
            f_init = adaptive_step_function_initialization(n_steps)

            # Add slight randomization to break symmetry
            f_init = [val * (0.9 + random.random() * 0.2) for val in f_init]

            all_candidates.append(f_init)

            # Early exit if time is running out
            if time.time() - start_time > max_time_seconds - 5:
                break

        if time.time() - start_time > max_time_seconds - 5:
            break

    # Parallel evaluation of candidates
    if all_candidates:
        try:
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_candidate)(candidate) for candidate in all_candidates
            )

            # Find best candidate
            best_idx = np.argmax(fitness_scores)
            best_c2 = fitness_scores[best_idx]
            best_function = all_candidates[best_idx].copy()

        except Exception:
            # Fallback to sequential evaluation if parallel fails
            best_c2 = 0.0
            best_function = []
            for i, candidate in enumerate(all_candidates):
                if time.time() - start_time > max_time_seconds - 5:
                    break
                score = evaluate_candidate(candidate)
                if score > best_c2:
                    best_c2 = score
                    best_function = candidate.copy()

    # Final refinement using local search and differential evolution if we have a candidate
    if best_function and time.time() - start_time < max_time_seconds - 5:
        # Apply local search refinement
        refined_local = local_search_refinement(best_function, max_iter=30)
        local_c2 = compute_c2(refined_local)

        if local_c2 > best_c2:
            best_c2 = local_c2
            best_function = refined_local

        # Apply differential evolution refinement (more intensive)
        if time.time() - start_time < max_time_seconds - 5:
            refined_de = differential_evolution_refinement(best_function, max_evals=200)
            de_c2 = compute_c2(refined_de)

            if de_c2 > best_c2:
                best_c2 = de_c2
                best_function = refined_de

    # Ensure we return at least some function
    if not best_function:
        # Fallback to simple construction
        best_function = [1.0] * 100

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")