# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time
from typing import Tuple, List
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from jax.config import config

# Enable JAX CPU usage
config.update('jax_platform_name', 'cpu')

# Use JAX for automatic differentiation
@jax_jit
def compute_autoconvolution_jax(f_vals):
    """Compute autoconvolution using JAX for speed and automatic differentiation"""
    f = jnp.array(f_vals, dtype=jnp.float32)
    
    # Compute autoconvolution using JAX's convolve
    g = jnp.convolve(f, f, mode='full')
    
    # Center the convolution result properly
    n = len(f)
    offset = n - 1
    g_centered = g[offset:-offset]
    
    return g_centered

@jax_jit  
def compute_norms_jax(g_vals):
    """Compute norms using JAX operations"""
    # Compute all norms efficiently using JAX
    g_abs = jnp.abs(g_vals)
    
    # L2 norm squared
    norm_l2_sq = jnp.sum(g_abs**2)
    
    # L1 norm
    norm_l1 = jnp.sum(g_abs)
    
    # L-infinity norm
    norm_inf = jnp.max(g_abs)
    
    return norm_l2_sq, norm_l1, norm_inf

@jax_jit
def compute_c2_jax(f_vals):
    """Compute C2 using JAX for automatic differentiation"""
    # Compute autoconvolution
    g = compute_autoconvolution_jax(f_vals)
    
    # Compute norms
    norm_l2_sq, norm_l1, norm_inf = compute_norms_jax(g)
    
    # Avoid division by zero
    eps = 1e-12
    norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
    norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)
    
    # Compute C2
    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

# Gradient computation for optimization
def compute_gradient_jax(f_vals):
    """Compute gradient of C2 with respect to f_vals using JAX"""
    try:
        f = jnp.array(f_vals, dtype=jnp.float32)
        grad_fn = grad(compute_c2_jax)
        grad_val = grad_fn(f)
        return np.array(grad_val)
    except Exception:
        return np.zeros_like(f_vals)

def compute_autoconvolution_fast(f_vals):
    """Fast autoconvolution using optimized NumPy operations"""
    # Ensure non-negative
    f = np.maximum(f_vals, 0.0)
    
    # Compute autoconvolution using NumPy's convolve
    g = convolve(f, f, mode='full')
    
    # Take only the central part representing the convolution over [-1/4, 1/4]
    n = len(f)
    offset = n - 1
    g_centered = g[offset:-offset]
    
    return g_centered

def compute_norms_fast(g_vals):
    """Fast computation of norms with piecewise integration"""
    # L2 norm squared using trapezoidal rule approximation
    # For piecewise linear segments, use trapezoidal rule: (y1+y2)/2 * h
    # But since we're working with discrete values, we use the integral of g^2
    # We assume unit spacing and use weighted average for better integration
    
    n = len(g_vals)
    if n < 2:
        norm_l2_sq = g_vals[0]**2 if n > 0 else 0.0
    else:
        # Trapezoidal integration for g^2
        g_squared = g_vals**2
        # Weighted average approach: (h/3)*(y1^2 + y1*y2 + y2^2) for each segment
        norm_l2_sq = 0.0
        for i in range(n-1):
            y1 = g_vals[i]
            y2 = g_vals[i+1]
            norm_l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # L1 norm
    norm_l1 = np.sum(np.abs(g_vals))
    
    # L-infinity norm
    norm_inf = np.max(np.abs(g_vals))
    
    return norm_l2_sq, norm_l1, norm_inf

def compute_c2_fast(f_vals):
    """Fast computation of C2 score"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0.0)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_fast(f_vals)
        
        # Compute norms
        l2_sq, l1, linf = compute_norms_fast(g_vals)
        
        # Avoid division by zero
        if l1 < 1e-12 or linf < 1e-12:
            return 0.0
            
        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return c2
        
    except Exception:
        return 0.0

def sophisticated_initialization(n: int) -> np.ndarray:
    """Create good initial candidate using multiple patterns and noise"""
    # Create base pattern with multiple regions
    f_vals = []
    
    # Segment the function into multiple regions
    segments = 6
    segment_size = n // segments
    
    for i in range(n):
        segment_idx = i // segment_size if segment_size > 0 else 0
        # Alternate between high and low regions with variation
        if segment_idx % 2 == 0:
            # High region with some noise
            val = 1.0 + np.random.random() * 0.5
        else:
            # Low region with some noise
            val = 0.1 + np.random.random() * 0.3
            
        # Add Gaussian-like smoothing effect
        if len(f_vals) > 0:
            smooth_factor = 0.6
            val = smooth_factor * val + (1-smooth_factor) * f_vals[-1]
            
        f_vals.append(max(0, val))
    
    # Add some structured variation
    noise_amplitude = 0.05
    noise = np.random.normal(0, noise_amplitude, n)
    f_vals = np.array(f_vals) + noise
    
    # Ensure non-negative and normalize to reasonable range
    f_vals = np.maximum(f_vals, 0)
    max_val = np.max(f_vals)
    if max_val > 0:
        f_vals = f_vals / max_val * 2.0
        
    return f_vals

def adaptive_gradient_optimization(initial_f: np.ndarray, 
                                 max_iter: int = 100,
                                 step_size: float = 0.01) -> np.ndarray:
    """Refine solution using adaptive gradient-based optimization"""
    f_current = initial_f.astype(np.float32)
    
    try:
        for iter_num in range(max_iter):
            # Compute gradient using JAX
            if iter_num % 5 == 0:  # Recompute gradient every few iterations
                grad_val = compute_gradient_jax(f_current)
            else:
                # Use previous gradient for stability
                grad_val = grad_val
                
            # Update with gradient ascent
            f_new = f_current + step_size * grad_val
            
            # Ensure non-negativity
            f_new = np.maximum(f_new, 0)
            
            # Check improvement with fast computation
            old_c2 = compute_c2_fast(f_current)
            new_c2 = compute_c2_fast(f_new)
            
            if new_c2 > old_c2:
                f_current = f_new
                # Gradually reduce step size if improvement happens
                step_size = max(step_size * 0.995, 1e-6)
            else:
                # Reduce step size if no improvement
                step_size *= 0.5
                if step_size < 1e-6:
                    break
                    
    except Exception:
        pass  # Return current solution if optimization fails
        
    return f_current

def multi_stage_optimization(n: int = 1000) -> Tuple[np.ndarray, float]:
    """Multi-stage optimization with progressive refinement"""
    # Stage 1: Coarse evolutionary search
    bounds = [(0.0, 3.0) for _ in range(n)]
    
    def objective(x):
        return -compute_c2_fast(x)  # Minimize negative C2
    
    # Run differential evolution with multiple seeds for better exploration
    best_score = -np.inf
    best_solution = None
    
    # Multi-start with three different seeds
    for seed in [42, 123, 456]:
        try:
            # Generate different initializations
            x0 = sophisticated_initialization(n) + np.random.normal(0, 0.05, n)
            
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                seed=seed,
                maxiter=80,
                popsize=max(10, n//50),
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                tol=1e-6
            )
            
            if -result.fun > best_score:
                best_score = -result.fun
                best_solution = result.x.copy()
                
        except Exception:
            continue
    
    # Stage 2: Fine-grained refinement using gradient-based optimization
    if best_solution is not None:
        refined_solution = adaptive_gradient_optimization(
            best_solution, 
            max_iter=80, 
            step_size=0.01
        )
        refined_score = compute_c2_fast(refined_solution)
        
        # Keep the better solution
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined_solution
    
    # Stage 3: Final polish with additional refinement
    if best_solution is not None:
        # Apply another round of gradient descent with reduced learning rate
        final_solution = adaptive_gradient_optimization(
            best_solution,
            max_iter=40,
            step_size=0.005
        )
        final_score = compute_c2_fast(final_solution)
        
        if final_score > best_score:
            best_score = final_score
            best_solution = final_solution
    
    return best_solution, best_score

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value"""
    start_time = time.time()
    
    try:
        # Multi-stage optimization
        f_values, score = multi_stage_optimization(1000)
        
        # Ensure we don't exceed time limit
        elapsed = time.time() - start_time
        if elapsed > 85:  # Leave 5 seconds buffer
            print(f"Warning: Time limit approached. Elapsed: {elapsed:.2f}s")
            
        # Return the best solution
        if f_values is not None:
            return f_values.tolist()
        else:
            # Fallback to simple pattern
            return sophisticated_initialization(1000).tolist()
            
    except Exception as e:
        # Fallback to simple initialization if optimization fails
        print(f"Fallback due to error: {e}")
        return sophisticated_initialization(1000).tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")