# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
from numba import jit
import time

# JIT compiled autoconvolution computation
@jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """
    Fast numba-based autoconvolution computation for step functions.
    """
    n = len(f_vals)
    # Result has length 2*n-1
    g = np.zeros(2*n - 1)

    # Manual computation of the convolution sum for step functions
    for i in range(n):
        for j in range(n):
            # In convolution, the value at index i+j comes from f[i] * f[j]
            g[i + j] += f_vals[i] * f_vals[j]

    return g

# JIT compiled computation of norms
@jit(nopython=True)
def compute_autoconvolution_norms_jit(f_array, dx):
    """JIT compiled computation of autoconvolution norms"""
    n = len(f_array)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_array)
    
    # Scale by step width for proper normalization
    g = g * dx

    # Compute norms using trapezoidal-like integration
    g_abs = np.abs(g)
    
    # L2 norm squared using trapezoidal-like formula
    g2_squared = 0.0
    for i in range(len(g)-1):
        g2_squared += (dx/3) * (g_abs[i]**2 + g_abs[i]*g_abs[i+1] + g_abs[i+1]**2)
    
    # L1 norm
    g1 = 0.0
    for i in range(len(g)):
        g1 += g_abs[i] * dx
    
    # L-infinity norm
    g_inf = 0.0
    for i in range(len(g)):
        val = g_abs[i]
        if val > g_inf:
            g_inf = val

    return g2_squared, g1, g_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    n = len(f_values)
    if n == 0:
        return 0.0

    # Generate step spacing
    dx = 0.5 / n

    # Convert to numpy array and ensure non-negative
    f_array = np.array(f_values, dtype=np.float64)
    f_array = np.maximum(f_array, 0.0)  # Clip negative values

    # Compute norms using JIT compiled version
    g2_squared, g1, g_inf = compute_autoconvolution_norms_jit(f_array, dx)

    # Avoid division by zero
    if g1 <= 1e-15 or g_inf <= 1e-15:
        return 0.0

    return g2_squared / (g1 * g_inf)

def construct_multiscale_gaussian_pattern(n: int) -> List[float]:
    """
    Construct a step function using multi-scale Gaussian patterns.
    Creates a hierarchy of Gaussian bumps with varying scales and amplitudes.
    """
    # Create base pattern with multiple Gaussian components
    pattern = np.zeros(n)

    # Define multiple scales for the Gaussian bumps
    scales = [n//8, n//16, n//32, n//64]  # Different scale levels
    scales = [s for s in scales if s >= 2]  # Filter out too small scales

    # Create Gaussian bumps at different locations and scales
    for i, scale in enumerate(scales):
        # Position the bump in the middle of the function
        center = n // 2 + (i - len(scales)//2) * n // 8
        center = max(scale, min(n - scale, center))  # Bound to valid range

        # Create Gaussian with decreasing amplitude for smaller scales
        amplitude = 1.0 / (i + 1)  # Smaller scales have lower amplitude

        # Generate Gaussian curve
        x = np.arange(n)
        gaussian = amplitude * np.exp(-0.5 * ((x - center) / scale)**2)

        # Add to pattern
        pattern += gaussian

    # Ensure non-negativity and normalize
    pattern = np.maximum(pattern, 0)

    # Add some random variation to avoid being too deterministic
    noise_factor = 0.1
    noise = np.random.normal(0, noise_factor * np.std(pattern), n)
    pattern = np.maximum(pattern + noise, 0)

    # Convert to list and return
    return pattern.tolist()

def adaptive_coordinate_descent(f_values: List[float], max_iter: int = 200, 
                              tolerance: float = 1e-6) -> List[float]:
    """
    Perform adaptive coordinate-wise descent optimization
    """
    current_f = np.array(f_values, dtype=np.float64)
    n = len(current_f)
    current_c2 = calculate_c2(current_f.tolist())
    
    # Track improvement history for adaptive step size
    improvement_history = []
    max_history = 10
    
    # Base learning rate
    base_lr = 0.05
    
    for iteration in range(max_iter):
        # Adaptive step sizing based on recent progress
        current_lr = base_lr
        
        if len(improvement_history) > 5:
            recent_improvements = improvement_history[-5:]
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement > 1e-8:
                current_lr = min(0.2, current_lr * 1.05)  # Increase LR
            else:
                current_lr = max(1e-5, current_lr * 0.9)  # Decrease LR
        
        # Track whether any improvement was made
        improved = False
        
        # Shuffle indices for stochastic updates
        indices = list(range(n))
        random.shuffle(indices)
        
        # Iterate through coordinates
        for idx in indices:
            # Compute directional derivative using finite difference
            eps = current_lr * max(1e-6, current_f[idx])
            
            # Test positive perturbation
            test_f_pos = current_f.copy()
            test_f_pos[idx] = max(0.0, current_f[idx] + eps)
            pos_c2 = calculate_c2(test_f_pos.tolist())
            
            # Test negative perturbation  
            test_f_neg = current_f.copy()
            test_f_neg[idx] = max(0.0, current_f[idx] - eps)
            neg_c2 = calculate_c2(test_f_neg.tolist())
            
            # Estimate gradient using central difference
            grad_est = (pos_c2 - neg_c2) / (2 * eps)
            
            # Update using gradient descent
            new_val = current_f[idx] - current_lr * grad_est
            new_val = max(0.0, new_val)
            
            # Check if update improves C2
            test_f = current_f.copy()
            test_f[idx] = new_val
            new_c2 = calculate_c2(test_f.tolist())
            
            if new_c2 > current_c2:
                current_f = test_f
                current_c2 = new_c2
                improved = True
                
                # Record improvement for adaptive learning rate
                improvement_history.append(new_c2 - current_c2)
            else:
                improvement_history.append(0.0)
        
        # Keep history bounded
        if len(improvement_history) > max_history:
            improvement_history.pop(0)
        
        # Early stopping if no improvement
        if not improved:
            break
            
        # Check for convergence
        if len(improvement_history) >= 2 and np.mean(improvement_history[-2:]) < tolerance:
            break
    
    return current_f.tolist()

def gradient_free_optimization_strategy() -> List[float]:
    """
    Main optimization routine using gradient-free coordinate descent
    """
    # Set up parameters
    n_trials = 5
    best_c2 = -float('inf')
    best_solution = None
    
    # Try multiple diverse initializations
    for trial in range(n_trials):
        # Generate diverse initial patterns using different strategies
        if trial == 0:
            # Multi-scale Gaussian pattern
            n_steps = random.randint(300, 700)
            initial_func = construct_multiscale_gaussian_pattern(n_steps)
        elif trial == 1:
            # Simple ramp pattern
            n_steps = random.randint(300, 700)
            initial_func = []
            half = n_steps // 2
            for i in range(n_steps):
                if i < half:
                    initial_func.append(i / half)
                else:
                    initial_func.append((n_steps - i) / half)
        elif trial == 2:
            # Uniform distribution with some noise
            n_steps = random.randint(300, 700)
            initial_func = [random.uniform(0.5, 1.5) for _ in range(n_steps)]
        elif trial == 3:
            # Alternating pattern
            n_steps = random.randint(300, 700)
            initial_func = [1.0 if i % 2 == 0 else 0.3 for i in range(n_steps)]
        else:
            # Pure random pattern
            n_steps = random.randint(300, 700)
            initial_func = [random.uniform(0, 1) for _ in range(n_steps)]
        
        # Ensure non-negativity
        initial_func = [max(0.0, x) for x in initial_func]
        
        # Apply coordinate descent optimization
        optimized_func = adaptive_coordinate_descent(initial_func, max_iter=150)
        
        # Evaluate final result
        final_c2 = calculate_c2(optimized_func)
        
        if final_c2 > best_c2:
            best_c2 = final_c2
            best_solution = optimized_func.copy()
        
        # Print progress
        if trial % 2 == 0:
            print(f"Trial {trial}: C2 = {final_c2:.6f}")
    
    if best_solution is None:
        # Fallback to simple initialization
        n_steps = 500
        best_solution = [1.0] * n_steps
    
    # Final local refinement with differential evolution
    try:
        solution_array = np.array(best_solution)
        bounds = [(0.0, 3.0) for _ in range(len(solution_array))]
        
        def objective(x):
            return -calculate_c2(x.tolist())
        
        result = differential_evolution(objective, bounds, maxiter=20, 
                                      popsize=10, seed=42, disp=False)
        
        refined = np.maximum(result.x, 0)
        final_c2 = calculate_c2(refined.tolist())
        
        if final_c2 > best_c2:
            best_solution = refined.tolist()
            
    except Exception:
        pass
    
    return best_solution

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    Uses gradient-free optimization approach for improved performance.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run gradient-free optimization
    try:
        start_time = time.time()
        best_solution = gradient_free_optimization_strategy()
        end_time = time.time()
        
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        return best_solution
        
    except Exception as e:
        # Fallback to basic approach if optimization fails
        print(f"Optimization failed with error: {e}, using fallback")
        return [1.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")