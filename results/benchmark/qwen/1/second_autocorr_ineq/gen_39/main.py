# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
import numba
from numba import jit
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
    
    # Manual convolution loop for speed - optimized for numba
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute the norms using proper piecewise integration for ||g||₂²
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
    except Exception as e:
        # Fallback in case of any numerical issues
        return 0.0

def sophisticated_initialization(n_steps: int = 500) -> List[float]:
    """
    Generate a sophisticated initial configuration based on mathematical intuition
    """
    # Create a step function that tries to balance flatness with sufficient mass
    # Based on mathematical insights: create a function that when convolved 
    # produces a relatively flat profile but with enough energy to achieve high C2
    
    # Start with alternating high/low regions with smooth transitions
    f = np.zeros(n_steps)
    
    # Create a base alternating pattern with specific structure
    segment_size = max(1, n_steps // 12)  # Smaller segments for more variation
    
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region - slightly varying to avoid symmetry
            base_val = 0.75 + np.random.random() * 0.2
            f[i:end_idx] = base_val + np.random.random(end_idx - i) * 0.1
        else:
            # Low region
            f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.15
    
    # Add Gaussian-based smoothing for more natural transitions
    x = np.linspace(-1, 1, n_steps)
    gaussian = np.exp(-0.5 * (x / 0.22)**2)  # Slightly narrower for sharper transitions
    f = f * gaussian * 0.5 + gaussian * 0.5
    
    # Add some noise to break symmetry and encourage diversity
    noise = np.random.normal(0, 0.015, n_steps)
    f = f + noise
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def generate_diverse_initial_population(n_individuals: int, n_steps: int) -> List[List[float]]:
    """
    Generate diverse initial population for evolutionary algorithm with better structure
    """
    population = []
    
    # Create various types of initial configurations
    for i in range(n_individuals):
        # Type 1: Alternating segments with smooth transitions (most important)
        if i % 5 == 0:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 10)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region
                    f[j:end_idx] = 0.8 + np.random.random(end_idx - j) * 0.1
                else:
                    # Low region
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15
            
            # Smooth with Gaussian
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.25)**2)
            f = f * gaussian * 0.6 + gaussian * 0.4
            
            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 2: Gaussian-like distribution
        elif i % 5 == 1:
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.15 + np.random.random() * 0.2
            mu = np.random.random() * 0.3 - 0.15  # Centered around -0.15 to 0.15
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 3: Uniform distribution with some structure  
        elif i % 5 == 2:
            f = np.random.random(n_steps)
            f = np.clip(f, 0, 1)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 4: Peak centered distribution
        elif i % 5 == 3:
            f = np.zeros(n_steps)
            center = n_steps // 2
            width = max(1, n_steps // 12 + np.random.randint(-3, 4))
            f[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0
            f += np.random.normal(0, 0.02, n_steps)
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())
            
        # Type 5: Mixed pattern with more complexity
        else:
            # Create a multi-peak distribution
            f = np.zeros(n_steps)
            peaks = [n_steps // 4, n_steps // 2, 3*n_steps // 4]
            for peak in peaks:
                width = max(1, n_steps // 20)
                start = max(0, peak - width // 2)
                end = min(n_steps, peak + width // 2)
                f[start:end] = 0.6 + np.random.random(end - start) * 0.3
            
            # Add Gaussian smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.3)**2)
            f = f * gaussian * 0.5 + gaussian * 0.5
            
            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

    return population

def multi_start_evolutionary_optimization(n_starts: int = 3) -> List[float]:
    """
    Run multiple evolutionary optimizations to find better solutions
    """
    n_steps = 500
    best_solution = None
    best_c2 = -np.inf
    
    for start in range(n_starts):
        try:
            # Generate different initial populations for each start
            if start == 0:
                # Use sophisticated initialization for first start
                initial_f = sophisticated_initialization(n_steps)
            else:
                # Use diverse population for subsequent starts
                population = generate_diverse_initial_population(1, n_steps)
                initial_f = population[0]
            
            # Define bounds for each parameter (step height)
            bounds = [(0, 1.0) for _ in range(n_steps)]

            def objective(x):
                # Return negative because we want to maximize C2
                return -evaluate_c2(x.tolist())

            # Use differential evolution for global optimization
            # Adapt parameters for each start to balance exploration vs exploitation
            popsize = 12 if start == 0 else 15
            maxiter = 30 if start == 0 else 20
            
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
                    
                    # Apply local refinement for the best solution so far
                    try:
                        # Local refinement with L-BFGS-B
                        def local_objective(f_vals):
                            return -evaluate_c2(f_vals.tolist())
                        
                        bounds_local = [(0, 1.0) for _ in range(n_steps)]
                        local_result = minimize(
                            local_objective,
                            optimized_f,
                            method='L-BFGS-B',
                            bounds=bounds_local,
                            options={'maxiter': 15}
                        )
                        
                        if local_result.success:
                            refined_f = np.maximum(local_result.x, 0)
                            if np.sum(refined_f) > 0:
                                refined_f = refined_f / np.sum(refined_f)
                            current_c2 = evaluate_c2(refined_f.tolist())
                        else:
                            current_c2 = evaluate_c2(optimized_f.tolist())
                    except:
                        current_c2 = evaluate_c2(optimized_f.tolist())
                    
                    if current_c2 > best_c2:
                        best_c2 = current_c2
                        best_solution = optimized_f.tolist() if 'refined_f' not in locals() else refined_f.tolist()
                        
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
    Function to construct step-function with high C2 value using improved methods
    """
    try:
        # Try sophisticated initialization first to get a good baseline
        initial_f = sophisticated_initialization(500)
        c2_initial = evaluate_c2(initial_f)
        
        # Run multi-start evolutionary optimization
        optimized_f = multi_start_evolutionary_optimization(3)
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