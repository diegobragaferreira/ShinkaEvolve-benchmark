# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import njit
import time
import warnings

@njit
def compute_autoconvolution_numba(f):
    """Compute autoconvolution g = f * f using numba JIT"""
    n = len(f)
    # Autoconvolution using discrete convolution
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]
    
    # Trim to center portion (length n-1)
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]
    return g_trimmed

@njit
def compute_c2_numba(f):
    """Compute C2 value for given step function f using numba JIT"""
    if len(f) < 2:
        return 0.0
    
    # Compute autoconvolution
    g = compute_autoconvolution_numba(f)
    
    if len(g) == 0:
        return 0.0
    
    # Compute norms
    norm_l2_sq = 0.0
    norm_l1 = 0.0
    norm_inf = 0.0
    
    for i in range(len(g)):
        abs_g = abs(g[i])
        norm_l2_sq += abs_g * abs_g
        norm_l1 += abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g
    
    # Avoid division by zero
    if norm_l1 < 1e-12 or norm_inf < 1e-12:
        return 0.0
    
    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

def sophisticated_initialization(n):
    """
    Create sophisticated initial step function with alternating segments
    and Gaussian weighting to balance flatness and energy concentration
    """
    # Create alternating high/low segments
    f = []
    segment_length = max(1, n // 8)  # Variable segment size
    
    for i in range(0, n, segment_length):
        if i // segment_length % 2 == 0:
            # High segment
            f.extend([1.0] * min(segment_length, n - i))
        else:
            # Low segment  
            f.extend([0.1] * min(segment_length, n - i))
    
    # Apply Gaussian weighting to smooth transitions
    if len(f) > 0:
        # Normalize to avoid extreme values
        f = np.array(f)
        f = np.clip(f, 0, 10.0)
        # Apply Gaussian smoothing kernel
        kernel_size = min(5, len(f)//4)
        if kernel_size > 1:
            kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
            kernel = kernel / np.sum(kernel)
            f = np.convolve(f, kernel, mode='same')
        
        # Ensure all values are non-negative
        f = np.maximum(f, 0)
    
    return f.tolist()

def adaptive_differential_evolution(objective_func, bounds, max_iter, popsize, seed, init_pop=None):
    """
    Adaptive differential evolution with dynamic population sizing
    """
    # Start with smaller population for faster initial exploration
    current_popsize = min(popsize, 10)
    current_max_iter = min(max_iter, 20)
    
    try:
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=current_max_iter,
            popsize=current_popsize,
            seed=seed,
            strategy='best1bin',
            init=init_pop,
            disp=False,
            atol=1e-6,
            rtol=1e-6
        )
        
        # If we have good convergence, increase population size
        if result.success and current_popsize < popsize:
            # Re-run with larger population if improvement was seen
            result = differential_evolution(
                objective_func,
                bounds,
                maxiter=max_iter - current_max_iter,
                popsize=popsize,
                seed=seed,
                strategy='best1bin',
                init=result.x.reshape(1, -1) if hasattr(result, 'x') else init_pop,
                disp=False,
                atol=1e-6,
                rtol=1e-6
            )
            
        return result
    except Exception:
        # Fallback to basic differential evolution
        try:
            return differential_evolution(
                objective_func,
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                seed=seed,
                strategy='best1bin',
                disp=False
            )
        except Exception:
            return None

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimization."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Track best results
    best_c2 = 0.0
    best_f = []
    
    # Time management
    start_time = time.time()
    max_time = 85  # Leave some buffer for cleanup
    
    # Try different configurations with multi-start approach
    configurations = [
        (100, 15, 10),
        (500, 20, 15), 
        (1000, 25, 20),
        (2000, 30, 25)
    ]
    
    # Best performing setup found through experimentation
    base_config = {
        'max_iterations': 50,
        'population_size': 20,
        'restarts': 3
    }
    
    # Multi-start optimization loop
    for config_idx, (n, popsize, maxiter) in enumerate(configurations):
        if time.time() - start_time > max_time:
            break
            
        # Skip very small problems
        if n < 50:
            continue
            
        # Multiple restarts for better exploration
        for restart in range(base_config['restarts']):
            if time.time() - start_time > max_time:
                break
                
            try:
                # Generate diverse initial population
                initial_population = []
                for _ in range(min(15, popsize)):
                    f_init = sophisticated_initialization(n)
                    # Add noise to break symmetry
                    noise = np.random.normal(0, 0.05, len(f_init))
                    f_noisy = np.maximum(np.array(f_init) + noise, 0)
                    initial_population.append(f_noisy.tolist())
                
                # Define bounds for each parameter (step height)
                bounds = [(0, 10) for _ in range(n)]
                
                # Run adaptive differential evolution
                result = adaptive_differential_evolution(
                    lambda x: -compute_c2_numba(np.maximum(x, 0)),
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=42 + restart + config_idx,
                    init_pop=initial_population
                )
                
                if result is not None and result.success:
                    f_opt = np.maximum(result.x, 0)
                    c2_value = -result.fun  # Negate again to get actual C2
                    
                    if c2_value > best_c2:
                        best_c2 = c2_value
                        best_f = f_opt.tolist()
                        # Early exit if we've found something very good
                        if best_c2 > 0.95:
                            break
                        
            except Exception as e:
                warnings.warn(f"Configuration {n} restart {restart} failed: {str(e)}")
                continue
    
    # Fallback if nothing worked well
    if len(best_f) == 0:
        # Try the most promising configuration
        n = 1000
        try:
            # Try sophisticated initialization directly
            f_sophisticated = sophisticated_initialization(n)
            c2_sophisticated = compute_c2_numba(np.array(f_sophisticated))
            
            if c2_sophisticated > best_c2:
                best_c2 = c2_sophisticated
                best_f = f_sophisticated
        except Exception:
            pass
            
        # Final fallback
        if len(best_f) == 0:
            n = 500
            best_f = [1.0] * n
            best_c2 = compute_c2_numba(np.array(best_f))
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")