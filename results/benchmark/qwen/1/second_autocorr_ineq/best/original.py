# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from typing import List
from numba import njit
import warnings
warnings.filterwarnings('ignore')

@njit
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

    # Compute the norms
    # ||g||₂² = sum(g[i]²) using proper piecewise integration
    norm_g_2_squared = 0.0

    # For piecewise linear integration, we use trapezoidal-like approach:
    # for consecutive pairs of points (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # ||g||₁ = sum(|g[i]|)
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

        # Avoid division by zero with stricter thresholds
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def sophisticated_initialization() -> List[float]:
    """
    Generate a sophisticated initial configuration based on mathematical intuition
    """
    n_steps = 500

    # Create a step function that tries to balance flatness with sufficient mass
    # We want to create a function that when convolved produces a relatively flat profile
    # but with enough energy to achieve high C2

    # Create a base function with alternating high/low regions using more varied amplitudes
    f = np.zeros(n_steps)

    # Divide into segments with varying sizes for more complexity
    segment_sizes = [max(1, n_steps // 12), max(1, n_steps // 8), max(1, n_steps // 10)]
    segment_size = segment_sizes[np.random.choice(len(segment_sizes))]

    # Alternate between high and low values to create interesting convolution behavior
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region with slight variation
            amplitude = 0.7 + np.random.random() * 0.25
            f[i:end_idx] = amplitude + np.random.random(end_idx - i) * 0.1
        else:
            # Low region with variation
            amplitude = 0.1 + np.random.random() * 0.15
            f[i:end_idx] = amplitude + np.random.random(end_idx - i) * 0.1

    # Add Gaussian-like structure for smoothness and structure preservation
    x = np.linspace(-1, 1, n_steps)
    # Use more varied gaussian parameters
    gaussian_width = 0.2 + np.random.random() * 0.2
    gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4

    # Add some noise for diversity
    noise_level = 0.01 + np.random.random() * 0.02
    noise = np.random.normal(0, noise_level, n_steps)
    f = f + noise

    # Ensure non-negativity
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def generate_diverse_initial_population(n_individuals: int, n_steps: int) -> List[List[float]]:
    """
    Generate diverse initial population for evolutionary algorithm with enhanced variety
    """
    population = []
    
    # Create various types of initial configurations
    for i in range(n_individuals):
        # Type 1: Alternating segments with smooth transitions
        if i % 5 == 0:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 10)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region
                    f[j:end_idx] = 0.8 + np.random.random(end_idx - j) * 0.15
                else:
                    # Low region
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15
            
            # Smooth with Gaussian
            x = np.linspace(-1, 1, n_steps)
            gaussian_width = 0.2 + np.random.random() * 0.15
            gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
            f = f * gaussian * 0.6 + gaussian * 0.4
            
            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 2: Multi-peak distribution
        elif i % 5 == 1:
            f = np.ones(n_steps) * 0.1  # Base low values
            # Add multiple peaks at different positions
            n_peaks = 3 + np.random.randint(0, 3)
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, n_steps)
                peak_width = max(1, n_steps // 15 + np.random.randint(-2, 3))
                start = max(0, peak_pos - peak_width // 2)
                end = min(n_steps, peak_pos + peak_width // 2)
                f[start:end] = np.maximum(f[start:end], 0.7 + np.random.random(end - start) * 0.2)
            
            # Add smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.25)**2)
            f = f * gaussian * 0.5 + gaussian * 0.5
            
            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 3: Gaussian-like distribution
        elif i % 5 == 2:
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.15 + np.random.random() * 0.2
            mu = np.random.random() * 0.3 - 0.15  # Centered around -0.15 to 0.15
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 4: Uniform distribution with some structure  
        elif i % 5 == 3:
            f = np.random.random(n_steps)
            # Add some structure with clustering
            clusters = 3 + np.random.randint(0, 3)
            for _ in range(clusters):
                center = np.random.randint(0, n_steps)
                width = max(1, n_steps // 10 + np.random.randint(-2, 3))
                start = max(0, center - width // 2)
                end = min(n_steps, center + width // 2)
                f[start:end] = np.maximum(f[start:end], 0.5 + np.random.random(end - start) * 0.3)
            f = np.clip(f, 0, 1)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 5: High-low alternating with enhanced transitions
        else:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 15)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region
                    f[j:end_idx] = 0.75 + np.random.random(end_idx - j) * 0.2
                else:
                    # Low region
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15
            
            # Apply more aggressive smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.3)**2)
            f = f * gaussian * 0.4 + gaussian * 0.6
            
            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

    return population

def adaptive_evolutionary_optimization() -> List[float]:
    """
    Use adaptive evolutionary algorithm to optimize step function
    """
    n_steps = 500  # Reasonable size for exploration

    # Define bounds for each parameter (step height)
    bounds = [(0, 1.0) for _ in range(n_steps)]

    def objective(x):
        # Return negative because we want to maximize C2
        return -evaluate_c2(x.tolist())

    # Adaptive optimization parameters with improved logic
    initial_popsize = 8  # Smaller initial population for faster exploration
    initial_maxiter = 20  # Fewer initial iterations
    max_popsize = 20     # Maximum population size
    max_iter_limit = 50  # Maximum iterations

    # Use differential evolution for global optimization with adaptive parameters
    try:
        # Start with smaller population for faster initial exploration
        popsize = initial_popsize
        maxiter = initial_maxiter

        # Run differential evolution with adaptive parameters
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            disp=False
        )

        if result.success:
            optimized_f = np.maximum(result.x, 0)
            # Normalize to ensure good scaling
            if np.sum(optimized_f) > 0:
                optimized_f = optimized_f / np.sum(optimized_f)
            return optimized_f.tolist()
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Return default if optimization fails
    return [1.0/n_steps] * n_steps

def multi_start_evolutionary_optimization(n_starts: int = 4) -> List[float]:
    """
    Run multiple evolutionary optimizations with different strategies to find the best solution
    """
    n_steps = 500
    best_solution = None
    best_c2 = -np.inf
    
    for start in range(n_starts):
        try:
            # Generate different initial populations based on various patterns
            if start == 0:
                # Use sophisticated initialization for first start
                initial_f = sophisticated_initialization()
            elif start == 1:
                # Use diverse population
                population = generate_diverse_initial_population(1, n_steps)
                initial_f = population[0]
            else:
                # Use random initialization for exploration
                initial_f = np.random.random(n_steps)
                initial_f = initial_f / np.sum(initial_f) if np.sum(initial_f) > 0 else initial_f
                initial_f = initial_f.tolist()
            
            # Define bounds for each parameter (step height)
            bounds = [(0, 1.0) for _ in range(n_steps)]

            def objective(x):
                # Return negative because we want to maximize C2
                return -evaluate_c2(x.tolist())

            # Use differential evolution with adaptive parameters based on start number
            popsize = 10 + start * 2  # Increase population size with start number
            maxiter = 20 + start * 5  # Increase iterations with start number
            
            # Cap the parameters to reasonable values
            popsize = min(popsize, 25)
            maxiter = min(maxiter, 40)

            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=42 + start,  # Different seeds for diversity
                    disp=False
                )
                
                if result.success:
                    optimized_f = np.maximum(result.x, 0)
                    # Normalize to ensure good scaling
                    if np.sum(optimized_f) > 0:
                        optimized_f = optimized_f / np.sum(optimized_f)
                    
                    current_c2 = evaluate_c2(optimized_f.tolist())
                    
                    if current_c2 > best_c2:
                        best_c2 = current_c2
                        best_solution = optimized_f.tolist()
                        
            except Exception as e:
                # If differential evolution fails, try with the initial solution
                current_c2 = evaluate_c2(initial_f)
                if current_c2 > best_c2:
                    best_c2 = current_c2
                    best_solution = initial_f
                    
        except Exception as e:
            continue  # Skip this start if it fails
    
    # If no valid solutions were found, return a default
    if best_solution is None:
        return [1.0/n_steps] * n_steps
    
    return best_solution

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using adaptive methods
    """
    # Try multiple approaches and select the best
    try:
        # Try sophisticated initialization first to get a good baseline
        initial_f = sophisticated_initialization()
        c2_initial = evaluate_c2(initial_f)
        
        # Run multi-start evolutionary optimization
        optimized_f = multi_start_evolutionary_optimization(4)
        c2_optimized = evaluate_c2(optimized_f)
        
        # Also try one more round of adaptive optimization
        adaptive_f = adaptive_evolutionary_optimization()
        c2_adaptive = evaluate_c2(adaptive_f)
        
        # Return the best among all approaches
        candidates = [
            (initial_f, c2_initial),
            (optimized_f, c2_optimized),
            (adaptive_f, c2_adaptive)
        ]
        
        best_candidate = max(candidates, key=lambda x: x[1])
        return best_candidate[0]
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")