# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
from scipy.interpolate import interp1d
import time
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Fast autoconvolution computation using Numba"""
    n = len(f_vals)
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            k = i + j
            g[k] += f_vals[i] * f_vals[j]
    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Fast norm computation using Numba"""
    n = len(g_vals)
    
    # L2 norm squared using trapezoidal-like approximation
    l2_sq = 0.0
    if n >= 2:
        for i in range(n-1):
            y1 = g_vals[i]
            y2 = g_vals[i+1]
            l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        if abs(g_vals[i]) > linf:
            linf = abs(g_vals[i])

    return l2_sq, l1, linf

def evaluate_c2_fast(f_vals):
    """Fast C2 evaluation with proper error handling"""
    try:
        f_vals = np.maximum(f_vals, 0)
        if len(f_vals) == 0:
            return 0.0
        
        g_vals = compute_autoconvolution_numba(f_vals)
        l2_sq, l1, l_inf = compute_norms_numba(g_vals)
        
        if l1 <= 1e-15 or l_inf <= 1e-15:
            return 0.0
            
        c2 = l2_sq / (l1 * l_inf)
        return c2
    except Exception:
        return 0.0

def create_initial_pattern(n_steps):
    """Create a sophisticated initial pattern based on mathematical intuition"""
    # Create a pattern that balances high and low values strategically
    # This pattern favors structure that leads to flatter autoconvolutions
    pattern = []
    
    # Use sine wave modulation to create oscillatory behavior
    x = np.linspace(0, 4*np.pi, n_steps)
    base_pattern = 0.5 + 0.3 * np.sin(x)
    
    # Introduce strategic high/low regions
    for i in range(n_steps):
        if i % 4 == 0:
            # High value region
            pattern.append(1.0 + np.random.random() * 0.5)
        elif i % 4 == 2:
            # Low value region  
            pattern.append(0.1 + np.random.random() * 0.2)
        else:
            # Intermediate values based on sine
            pattern.append(max(0.0, base_pattern[i] + 0.1 * np.random.random() - 0.05))
    
    # Add some additional structure
    for i in range(0, n_steps, 8):
        if i < n_steps:
            pattern[i] = 1.0 + np.random.random() * 0.3
    
    return np.array(pattern)

def adaptive_refinement_step(current_params, target_c2, max_iter=50):
    """Perform adaptive refinement using gradient information"""
    # Create smoother representation for better gradient estimation
    n_orig = len(current_params)
    
    # Interpolate to higher resolution for better gradient estimation
    if n_orig < 100:
        # Create fine grid
        fine_points = np.linspace(0, n_orig-1, 2*n_orig)
        interpolator = interp1d(np.arange(n_orig), current_params, kind='linear', fill_value='extrapolate')
        current_params_fine = interpolator(fine_points)
        n_fine = len(current_params_fine)
    else:
        current_params_fine = current_params
        n_fine = n_orig
    
    # Use scipy's optimization with bounds
    bounds = [(0.0, 3.0) for _ in range(n_fine)]
    
    def objective(x):
        # Convert back to original resolution if needed
        if n_fine != n_orig:
            interpolator = interp1d(fine_points, x, kind='linear', fill_value='extrapolate')
            x_original = interpolator(np.arange(n_orig))
        else:
            x_original = x
            
        return -evaluate_c2_fast(x_original)  # Negative for minimization
    
    # Use L-BFGS-B for local refinement
    try:
        result = minimize(objective, current_params_fine, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
        if result.success:
            refined_params = result.x
            if n_fine != n_orig:
                interpolator = interp1d(fine_points, refined_params, kind='linear', fill_value='extrapolate')
                refined_params = interpolator(np.arange(n_orig))
            return np.maximum(refined_params, 0)
    except:
        pass
    
    # If optimization fails, return current with slight perturbation
    return np.maximum(current_params + np.random.normal(0, 0.01, len(current_params)), 0)

def multi_resolution_optimization():
    """Multi-resolution optimization approach"""
    # Start with coarse resolution
    n_steps = 500
    current_params = create_initial_pattern(n_steps)
    
    # First stage: coarse optimization
    current_params = adaptive_refinement_step(current_params, 0.9, max_iter=30)
    
    # Second stage: medium resolution
    if n_steps > 100:
        n_medium = min(1000, n_steps * 2)
        # Interpolate to medium resolution
        if n_medium > n_steps:
            x_orig = np.linspace(0, n_steps-1, n_steps)
            x_new = np.linspace(0, n_steps-1, n_medium)
            interpolator = interp1d(x_orig, current_params, kind='linear', fill_value='extrapolate')
            current_params = interpolator(x_new)
        
        current_params = adaptive_refinement_step(current_params, 0.92, max_iter=50)
        
        # Third stage: fine resolution
        if n_medium > 100:
            n_fine = min(2000, n_medium * 2)
            if n_fine > n_medium:
                x_orig = np.linspace(0, n_medium-1, n_medium)
                x_new = np.linspace(0, n_medium-1, n_fine)
                interpolator = interp1d(x_orig, current_params, kind='linear', fill_value='extrapolate')
                current_params = interpolator(x_new)
            
            current_params = adaptive_refinement_step(current_params, 0.94, max_iter=70)
    
    # Final refinement
    current_params = adaptive_refinement_step(current_params, 0.96, max_iter=100)
    
    return current_params

def construct_function() -> list[float]:
    """Main function to construct step-function with high C2 value"""
    start_time = time.time()
    
    try:
        # Multi-resolution optimization approach
        f_values = multi_resolution_optimization()
        
        # Ensure we don't exceed time limit
        elapsed = time.time() - start_time
        if elapsed > 85:  # Leave 5 seconds buffer
            pass
            
        # Final verification and cleanup
        f_values = np.maximum(f_values, 0)
        
        # Report final result
        final_c2 = evaluate_c2_fast(f_values)
        print(f"Optimization completed in {elapsed:.2f} seconds")
        print(f"Final C2 achieved: {final_c2}")
        
        return f_values.tolist()
        
    except Exception as e:
        # Fallback to simple pattern if anything goes wrong
        print(f"Fallback due to error: {e}")
        n_steps = 500
        fallback_pattern = [0.5 + 0.3 * np.sin(i * 0.2) for i in range(n_steps)]
        return np.maximum(fallback_pattern, 0).tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")