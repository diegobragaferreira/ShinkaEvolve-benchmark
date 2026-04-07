# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit
import optuna
from scipy.optimize import minimize

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # For L2^2 norm using proper trapezoidal integration
    # For convolution on domain [-1/2, 1/2], with len(g_vals) points
    # Step width h = 1.0 / (len(g_vals) - 1) if len > 1, else 0.001
    if len(g_vals) >= 2:
        # Trapezoidal rule: h * (y0^2 + 2*y1^2 + ... + yn-1^2)/2  
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_c2_for_params(params):
    """Wrapper function for optimization"""
    try:
        # Ensure non-negative values
        f_vals = np.clip(params, 0, None)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute C2
        c2 = compute_c2_numba(g_vals)
        
        return c2
    except Exception:
        return 0.0

def sophisticated_initialization(dim):
    """Create sophisticated initial step function using multiple strategies"""
    # Strategy 1: Alternating high/low pattern
    pattern1 = []
    for i in range(dim):
        pattern1.append(1.0 if i % 2 == 0 else 0.1)
    
    # Strategy 2: Sinusoidal modulation with variation
    x = np.linspace(-1, 1, dim)
    peak1 = np.exp(-((x - 0.3)**2) / 0.1)
    peak2 = np.exp(-((x + 0.3)**2) / 0.1)
    pattern2 = np.maximum(peak1, peak2)
    
    # Strategy 3: Structured pattern with segments
    segments = max(1, dim // 20)
    pattern3 = []
    for i in range(segments):
        segment_size = dim // segments
        base_height = 0.5 + 0.3 * np.sin(i * 0.7)
        if i % 3 == 0:
            height = 1.0  # Peaks
        elif i % 3 == 1:
            height = 0.3  # Valleys
        else:
            height = base_height  # Middle values
        segment_vals = [height] * segment_size
        pattern3.extend(segment_vals)
    
    # Trim or extend to exact size
    if len(pattern3) > dim:
        pattern3 = pattern3[:dim]
    elif len(pattern3) < dim:
        padding = [0.5] * (dim - len(pattern3))
        pattern3.extend(padding)
    
    # Strategy 4: Sobol sequence based initialization
    try:
        sampler = qmc.Sobol(d=dim, seed=42)
        points = sampler.random(n=100)
    except:
        points = np.random.random((100, dim))
    
    pattern4 = []
    for i in range(dim):
        pattern_val = 0.5 + 0.3 * np.sin(i * 0.7)
        variation = points[i % 100][0] * 0.2 if i < 100 else np.random.random() * 0.2
        pattern4.append(max(0, pattern_val + variation - 0.1))
    
    # Evaluate and return best pattern
    patterns = [pattern1, pattern2.tolist(), pattern3, pattern4]
    best_pattern = pattern1
    best_score = -1.0
    
    for p in patterns:
        score = compute_c2_for_params(p)
        if score > best_score:
            best_score = score
            best_pattern = p
    
    return best_pattern

def evolutionary_optimization_single(size):
    """Optimize step function using differential evolution with enhanced parameters"""
    # Set up bounds (0 to 5 for robustness)
    bounds = [(0, 5.0)] * size

    # Use sophisticated initialization
    x0 = sophisticated_initialization(size)

    # Run differential evolution with improved parameters
    result = differential_evolution(
        lambda x: -compute_c2_for_params(x),
        bounds,
        maxiter=100,
        popsize=20,
        seed=42,
        disp=False,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7
    )

    return result.x

def convolution_aware_local_search(initial_params, max_iter=100):
    """Local search method that exploits the structure of convolution"""
    def objective(x):
        return -compute_c2_for_params(x)
    
    # Use L-BFGS-B for local refinement
    try:
        result = minimize(
            objective, 
            initial_params, 
            method='L-BFGS-B', 
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        return result.x
    except:
        return initial_params

def adaptive_optimization_with_refinement():
    """Use multi-config approach with local refinement"""
    start_time = time.time()
    best_c2 = -np.inf
    best_params = None
    
    # Try different configurations to find best
    configurations = [
        (200, 50),
        (400, 50),
        (600, 50),
        (800, 50),
        (1000, 50)
    ]
    
    # Add some randomness to better explore
    for _ in range(10):
        dim = np.random.randint(200, 1000)
        iterations = np.random.randint(30, 60)
        configurations.append((dim, iterations))
    
    # Run optimization for each configuration
    for n_steps, n_trials in configurations:
        if time.time() - start_time > 85:
            break
            
        try:
            # Create temporary study for this configuration
            temp_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=np.random.randint(1000)),
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Run trials
            for trial_num in range(n_trials):
                if time.time() - start_time > 85:
                    break
                    
                trial = temp_study.ask()
                
                # Create parameters with structured pattern
                params = []
                for i in range(n_steps):
                    # Dynamic pattern based on position
                    if i % 5 == 0:
                        param = 1.0 + np.random.normal(0, 0.2)  # Peaks
                    elif i % 5 == 2:
                        param = 0.2 + np.random.normal(0, 0.1)  # Valleys
                    else:
                        param = 0.5 + 0.3 * np.sin(i * 0.3) + np.random.normal(0, 0.1)  # Middle
                    params.append(max(0, param))
                
                # Evaluate
                c2 = compute_c2_for_params(params)
                
                # Report result
                temp_study.tell(trial, c2)
                
                # Check if this is the best so far
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()
                    
        except Exception as e:
            continue
    
    # Perform local refinement on the best found solution
    if best_params is not None and len(best_params) > 0:
        try:
            refined_params = convolution_aware_local_search(best_params, 100)
            refined_c2 = compute_c2_for_params(refined_params)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_params = refined_params
        except:
            pass
    
    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    try:
        # First try the hybrid optuna approach
        best_params, best_c2 = adaptive_optimization_with_refinement()
        
        # If we found a good solution, use it
        if best_params is not None and best_c2 > 0:
            return best_params
        
        # Otherwise fall back to deterministic evolutionary approach
        size = np.random.randint(500, 1000)
        best_f_vals = evolutionary_optimization_single(size)
        
        # Check if it's a good enough solution
        c2_val = compute_c2_for_params(best_f_vals)
        print(f"Best C2 found: {c2_val}")

        # Return the optimized values
        return best_f_vals.tolist()

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to structured initialization
        return sophisticated_initialization(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")