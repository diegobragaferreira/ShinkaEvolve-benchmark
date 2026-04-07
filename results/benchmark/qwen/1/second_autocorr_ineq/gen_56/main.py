# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from numba import jit
import random
import time
from scipy.special import erf
import warnings

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values):
    """Optimized computation of autoconvolution norms using numba"""
    n = len(f_values)
    
    # Initialize autoconvolution array
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Keep only center portion
    half_len = n - 1
    g_center = g[half_len:-half_len]
    
    # Compute norms
    norm_2_squared = np.sum(g_center**2)
    norm_1 = np.sum(np.abs(g_center))
    norm_inf = np.max(np.abs(g_center))
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation with proper handling"""
    try:
        if not f_values:
            return 0.0, 0.0, 0.0, 0.0

        # Convert to numpy array for easier manipulation
        f = np.array(f_values, dtype=np.float64)

        # Ensure non-negative values
        f = np.maximum(f, 0.0)

        # Compute autoconvolution g = f * f
        g = np.convolve(f, f, mode='full')

        # Keep only the valid convolution part (middle)
        half_len = len(f) - 1
        g_valid = g[half_len:-half_len]

        # Compute norms
        norm_2_squared = np.sum(g_valid**2)
        norm_1 = np.sum(np.abs(g_valid))
        norm_inf = np.max(np.abs(g_valid))

        # Avoid division by zero
        if norm_1 == 0 or norm_inf == 0:
            return 0.0, 0.0, 0.0, 0.0

        # C2 = ||g||₂² / (||g||₁ · ||g||∞)
        c2 = norm_2_squared / (norm_1 * norm_inf)

        return c2, norm_2_squared, norm_1, norm_inf
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_c2(individual):
    """Evaluate fitness of individual (step function) for maximizing C2"""
    try:
        # Ensure non-negative values
        individual = np.maximum(individual, 0.0)

        # Compute C2 value
        c2, _, _, _ = compute_autoconvolution_norms(individual)

        # Return negative because we want to maximize
        return -c2
    except Exception:
        # Return very poor fitness if error occurs
        return 1e10

def smooth_step_function(x, center, width, height):
    """Generate a smooth approximation to a step function using error functions"""
    # Convert to smooth sigmoid-like shape
    return height * (erf((x - center + width/2) / (width/3)) - 
                     erf((x - center - width/2) / (width/3))) / 2

def multi_scale_initialization(n_steps):
    """Create multi-scale initialization for better exploration"""
    # Create coarse grid first
    coarse_points = np.linspace(-0.25, 0.25, min(21, n_steps//2 + 1))
    coarse_values = np.random.rand(len(coarse_points)) * 0.5 + 0.5
    
    # Interpolate to fine grid
    fine_grid = np.linspace(-0.25, 0.25, n_steps)
    
    # Use piecewise linear interpolation
    coarse_fine = np.interp(fine_grid, coarse_points, coarse_values)
    
    # Add some noise for diversity
    noise = np.random.normal(0, 0.02, n_steps)
    adjusted = coarse_fine + noise
    
    # Ensure non-negative
    adjusted = np.maximum(adjusted, 0.0)
    
    # Normalize to reasonable scale
    if np.sum(adjusted) > 0:
        adjusted = adjusted * n_steps / np.sum(adjusted)
    
    return adjusted.tolist()

def kernel_smoothed_initialization(n_steps):
    """Initialize with kernel-smoothed random pattern"""
    # Generate base random pattern
    base_pattern = np.random.rand(n_steps) * 0.8 + 0.2
    
    # Apply gaussian smoothing
    kernel_size = max(1, n_steps // 50)
    kernel = np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * (kernel_size/3)**2))
    kernel = kernel / np.sum(kernel)
    
    smoothed = np.convolve(base_pattern, kernel, mode='same')
    
    # Apply soft thresholding to encourage sparsity
    threshold = np.mean(smoothed) * 0.3
    smoothed = np.maximum(smoothed - threshold, 0.0)
    
    # Normalize
    if np.sum(smoothed) > 0:
        smoothed = smoothed * n_steps / np.sum(smoothed)
    
    return smoothed.tolist()

def adaptive_gradient_descent(initial_solution, max_iter=1000):
    """Perform adaptive gradient descent with momentum and learning rate scheduling"""
    x = np.array(initial_solution, dtype=float)
    n = len(x)
    
    # Optimizer parameters
    lr = 0.1
    momentum = 0.9
    decay_rate = 0.99
    
    # Initialize momentum terms
    velocity = np.zeros_like(x)
    
    best_x = x.copy()
    best_score = evaluate_c2(x)
    patience_counter = 0
    
    for iteration in range(max_iter):
        # Compute gradient numerically
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            grad_i = (evaluate_c2(x_plus) - evaluate_c2(x_minus)) / (2 * eps)
            grad[i] = grad_i
            
        # Update with momentum
        velocity = momentum * velocity - lr * grad
        x = x + velocity
        
        # Project onto feasible region
        x = np.maximum(x, 0.0)
        
        # Check improvement
        current_score = evaluate_c2(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Adaptive learning rate
        if patience_counter > 10:
            lr *= decay_rate
            patience_counter = 0
        
        # Early stopping
        if lr < 1e-6:
            break
    
    return best_x.tolist()

def convex_relaxed_optimization(n_steps):
    """Use convex relaxation approach to initialize and optimize"""
    # Start with multi-scale initialization
    init_solution = multi_scale_initialization(n_steps)
    
    # Apply kernel smoothing for better structure
    init_solution = kernel_smoothed_initialization(n_steps)
    
    # Refine with adaptive gradient descent
    refined_solution = adaptive_gradient_descent(init_solution, max_iter=500)
    
    # Apply local refinement with scipy.optimize
    try:
        bounds = [(0.0, 10.0) for _ in range(n_steps)]
        result = minimize(
            evaluate_c2,
            refined_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8},
            tol=1e-8
        )
        if result.success:
            final_solution = result.x.tolist()
        else:
            final_solution = refined_solution
    except:
        final_solution = refined_solution
    
    return final_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using convex optimization"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Use optimized number of steps
        n_steps = 1500
        
        # Multi-scale convex optimization approach
        best_solution = convex_relaxed_optimization(n_steps)
        
        # Ensure non-negative values
        best_solution = [max(0, x) for x in best_solution]
        
        # Normalize to avoid extreme values that might cause numerical issues
        total = sum(best_solution)
        if total > 0:
            best_solution = [x / total * len(best_solution) for x in best_solution]
        
        # Apply final smoothing to ensure good behavior
        if len(best_solution) > 10:
            # Simple moving average smoothing
            window_size = max(1, len(best_solution) // 100)
            smoothed = []
            for i in range(len(best_solution)):
                start_idx = max(0, i - window_size)
                end_idx = min(len(best_solution), i + window_size)
                avg = np.mean(best_solution[start_idx:end_idx])
                smoothed.append(avg)
            best_solution = smoothed
        
        return best_solution

    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Fallback due to error: {e}")
        return multi_scale_initialization(500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")