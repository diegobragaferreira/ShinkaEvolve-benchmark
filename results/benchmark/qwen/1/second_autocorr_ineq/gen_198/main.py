# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit
import time
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
import random
from typing import Tuple, List, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Enable JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')

# JIT compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using Numba for speed with optimized memory access"""
    n = len(f_vals)
    # Create convolution result array
    g = np.zeros(2*n - 1)

    # Compute autoconvolution: g[k] = sum(f[i]*f[k-i])
    # Optimized nested loop with better cache locality
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            k = i + j
            g[k] += f_i * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with Numba and proper piecewise integration"""
    n = len(g_vals)

    # L2 norm squared using trapezoidal approximation for piecewise linear functions
    # For each pair of adjacent points with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    l2_sq = 0.0
    for i in range(n-1):
        y1 = g_vals[i]
        y2 = g_vals[i+1]
        l2_sq += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val

    return l2_sq, l1, linf

def compute_c2_score(f_vals):
    """Compute C2 score for given step function values with robust error handling"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)

        # Avoid division by zero with small epsilon
        epsilon = 1e-15
        if l1 < epsilon or linf < epsilon:
            return 0.0

        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return c2

    except Exception as e:
        warnings.warn(f"C2 computation failed: {e}")
        return 0.0

# JAX-based computation for gradients and optimization
@jax_jit
def compute_c2_jax(f_vals):
    """JAX version for automatic differentiation"""
    # Convert to JAX array
    f = jnp.array(f_vals, dtype=jnp.float32)

    # Compute autoconvolution using JAX operations  
    g = jnp.convolve(f, f, mode='full')
    n = len(f)
    offset = (n - 1) // 2
    g = g[offset:-offset]

    # Compute norms
    g_abs = jnp.abs(g)
    norm_l2_sq = jnp.sum(g_abs**2)
    norm_l1 = jnp.sum(g_abs)
    norm_inf = jnp.max(g_abs)

    # Avoid division by zero
    eps = 1e-12
    norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
    norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 with respect to f_vals using JAX"""
    try:
        f = jnp.array(f_vals, dtype=jnp.float32)
        grad_fn = grad(compute_c2_jax)
        grad_val = grad_fn(f)
        return np.array(grad_val)
    except Exception as e:
        warnings.warn(f"Gradient computation failed: {e}")
        return np.zeros_like(f_vals, dtype=np.float32)

def generate_adaptive_initialization(n: int) -> np.ndarray:
    """
    Create intelligent initial candidates using mathematical insights:
    - Mix of high and low values to encourage diverse convolution properties
    - Smooth transitions to avoid numerical issues
    - Concentrated mass in center to promote good C2 behavior
    """
    # Create base pattern with alternating regions
    f_vals = np.zeros(n)
    
    # Divide into segments with dynamic sizing
    segment_size = max(1, n // 12)
    
    # Create alternating high/medium/low pattern with mathematical structure
    for i in range(0, n, segment_size):
        end_idx = min(i + segment_size, n)
        segment_idx = i // segment_size
        
        if segment_idx % 3 == 0:
            # High region with some variation
            base_val = 1.2 + np.random.random() * 0.5
            f_vals[i:end_idx] = base_val + np.random.random(end_idx - i) * 0.3
        elif segment_idx % 3 == 1:
            # Medium region
            base_val = 0.6 + np.random.random() * 0.3
            f_vals[i:end_idx] = base_val + np.random.random(end_idx - i) * 0.2
        else:
            # Low region
            base_val = 0.2 + np.random.random() * 0.2
            f_vals[i:end_idx] = base_val + np.random.random(end_idx - i) * 0.1
    
    # Add central concentration to promote better convolution
    center = n // 2
    width = max(1, n // 8)
    for i in range(max(0, center - width // 2), min(n, center + width // 2)):
        # Gradually increase values towards center for better autoconvolution
        dist_from_center = abs(i - center)
        decay_factor = 1.0 - 0.7 * (dist_from_center / (width // 2))
        f_vals[i] *= decay_factor
    
    # Apply Gaussian smoothing for better transitions
    if n > 10:
        kernel_size = max(3, min(7, n // 15))
        if kernel_size > 1:
            kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
            kernel = kernel / np.sum(kernel)
            # Use convolution with careful boundary handling
            f_vals = np.convolve(f_vals, kernel, mode='same')
    
    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)
    if np.sum(f_vals) > 0:
        f_vals = f_vals / np.sum(f_vals) * 1.5
    
    return f_vals

def adaptive_gradient_optimization(initial_f: np.ndarray, 
                                 max_iterations: int = 100,
                                 learning_rate: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Apply adaptive gradient-based refinement on an initial solution with improved convergence
    """
    f_current = np.array(initial_f, dtype=np.float32)
    
    # Adaptive learning rate schedule with momentum-like behavior
    initial_lr = learning_rate
    patience = 0
    best_score = compute_c2_score(f_current)
    
    for iteration in range(max_iterations):
        try:
            # Compute gradient
            grad_val = compute_gradient_jax(f_current)
            
            # Adaptive learning rate based on gradient magnitude
            grad_mag = np.linalg.norm(grad_val)
            adaptive_lr = initial_lr / (1.0 + grad_mag * 0.05)
            
            # Update with gradient ascent
            f_new = f_current + adaptive_lr * grad_val
            
            # Ensure non-negativity
            f_new = np.maximum(f_new, 0)
            
            # Check improvement
            new_c2 = compute_c2_score(f_new)
            
            if new_c2 > best_score:
                f_current = f_new
                best_score = new_c2
                patience = 0  # Reset patience
            else:
                patience += 1
                # Reduce learning rate if no improvement for a few iterations
                if patience > 5:
                    adaptive_lr *= 0.7
                    if adaptive_lr < 1e-6:
                        break
                if patience > 10:
                    break  # Stop if no improvement for too long
                    
        except Exception as e:
            warnings.warn(f"Gradient optimization failed at iteration {iteration}: {e}")
            break
    
    return f_current, best_score

def multi_scale_adaptive_optimization(n: int, max_time: float) -> Tuple[List[float], float]:
    """
    Perform multi-scale adaptive optimization with dynamic parameter adjustment
    """
    start_time = time.time()
    
    # Scale factors for resolution hierarchy with optimized progression
    scales = [0.25, 0.5, 1.0]
    best_solution = None
    best_score = 0.0
    
    # Start with coarser resolution for faster exploration
    for scale_idx, scale in enumerate(scales):
        if time.time() - start_time > max_time * 0.9:
            break
            
        current_n = max(50, int(n * scale))
        
        # Dynamic population size based on scale level
        base_popsize = 10
        if scale_idx == 0:  # Coarse resolution
            popsize = max(5, min(15, base_popsize // 2))
        elif scale_idx == 1:  # Medium resolution
            popsize = max(10, min(20, base_popsize))
        else:  # Fine resolution
            popsize = max(15, min(30, base_popsize * 2))
        
        # Dynamic maxiter based on scale and available time
        maxiter = max(20, min(80, int(60 * scale)))
        
        # Generate initial population with adaptive initialization
        population = []
        for i in range(popsize):
            # Create diversified initial solution with more structure
            initial_f = generate_adaptive_initialization(current_n)
            # Add some randomization with scale-dependent variance
            noise_level = 0.02 * (1 - scale) + 0.01
            noise = np.random.normal(0, noise_level, current_n)
            initial_f = np.maximum(initial_f + noise, 0)
            if np.sum(initial_f) > 0:
                initial_f = initial_f / np.sum(initial_f)
            population.append(initial_f)
        
        # Evolve population with differential evolution
        bounds = [(0.0, 3.0) for _ in range(current_n)]
        
        def objective(x):
            score = compute_c2_score(x)
            return -score  # Minimize negative to maximize original score
            
        # Run differential evolution with adaptive parameters
        try:
            # Use different strategies based on scale level
            strategy = 'best1bin' if scale_idx < 2 else 'rand1bin'
            
            result = differential_evolution(
                objective,
                bounds,
                seed=random.randint(0, 1000),
                maxiter=maxiter,
                popsize=popsize,
                mutation=(0.5, 1.0) if scale_idx < 2 else (0.7, 1.0),
                recombination=0.7 if scale_idx < 2 else 0.8,
                strategy=strategy,
                disp=False
            )
            
            if result.success and -result.fun > best_score:
                best_score = -result.fun
                best_solution = result.x.copy()
                
        except Exception as e:
            warnings.warn(f"Differential evolution failed at scale {scale_idx}: {e}")
            continue  # Skip this scale if optimization fails
    
    # If we found a solution, refine it with gradient-based optimization
    if best_solution is not None:
        try:
            # Use more aggressive refinement for the best solution
            refined_solution, refined_score = adaptive_gradient_optimization(
                best_solution, 
                max_iterations=min(100, max(20, n // 5))
            )
            
            if refined_score > best_score:
                best_score = refined_score
                best_solution = refined_solution
        except Exception as e:
            warnings.warn(f"Gradient refinement failed: {e}")
            pass  # Continue with previous solution if refinement fails
    
    return best_solution.tolist() if best_solution is not None else [], best_score

def parallel_multi_start_optimization(configurations: List[int], max_time: float) -> Tuple[List[float], float]:
    """
    Run parallel optimizations on multiple configurations to better explore the solution space
    """
    def run_single_optimization(n):
        try:
            solution, score = multi_scale_adaptive_optimization(n, max_time)
            return solution, score
        except Exception as e:
            warnings.warn(f"Single optimization failed for n={n}: {e}")
            return [], 0.0
    
    best_solution = None
    best_score = 0.0
    
    start_time = time.time()
    num_processes = min(mp.cpu_count(), len(configurations))
    
    if num_processes > 1:
        # Use parallel processing for better efficiency
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(run_single_optimization, n) for n in configurations]
            
            for future in futures:
                try:
                    solution, score = future.result(timeout=max_time)
                    if score > best_score:
                        best_score = score
                        best_solution = solution
                except Exception as e:
                    warnings.warn(f"Parallel optimization failed: {e}")
                    continue
    else:
        # Sequential fallback
        for n in configurations:
            if time.time() - start_time > max_time * 0.9:
                break
            solution, score = run_single_optimization(n)
            if score > best_score:
                best_score = score
                best_solution = solution
    
    return best_solution if best_solution is not None else [], best_score

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive multi-scale approach"""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Time limit enforcement
    start_time = time.time()
    
    # Try different configurations with multi-scale refinement
    best_c2 = 0.0
    best_f = []
    
    # Try different sizes with multi-scale optimization
    configurations = [200, 400, 600, 800, 1000, 1200, 1500, 2000] 
    
    # Use parallel optimization for better resource utilization
    try:
        solution, score = parallel_multi_start_optimization(configurations, 85)
        if score > best_c2:
            best_c2 = score
            best_f = solution
    except Exception as e:
        warnings.warn(f"Parallel optimization failed: {e}")
        # Fall back to sequential approach
        for n in configurations:
            if time.time() - start_time > 85:  # Leave 5 seconds buffer
                break
                
            try:
                # Use multi-scale adaptive optimization approach
                solution, score = multi_scale_adaptive_optimization(n, 85 - (time.time() - start_time))
                
                if score > best_c2:
                    best_c2 = score
                    best_f = solution
            except Exception as e2:
                warnings.warn(f"Sequential optimization failed for n={n}: {e2}")
                continue
    
    # If no good solution found, fallback to sophisticated initialization
    if len(best_f) == 0:
        try:
            n = 1000
            best_f = generate_adaptive_initialization(n).tolist()
            best_c2 = compute_c2_score(best_f)
        except Exception as e:
            warnings.warn(f"Fallback initialization failed: {e}")
            # Last resort - uniform distribution
            n = 500
            best_f = [1.0/n] * n
            best_c2 = compute_c2_score(best_f)
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")