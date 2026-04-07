# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit
import random
import time

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

def gaussian_weighted_alternating_initialization(n_steps):
    """Create sophisticated initial step function with Gaussian-weighted alternating segments"""
    # Create alternating high/low segments with gaussian weighting for smooth transitions
    f_values = []
    high_val = 1.0
    low_val = 0.1
    
    # Create base alternating pattern
    for i in range(n_steps):
        if i % 2 == 0:
            f_values.append(high_val)
        else:
            f_values.append(low_val)
    
    # Apply Gaussian kernel for smoother transitions
    smoothed = []
    sigma = max(1, n_steps // 20)  # Adaptive sigma based on problem size
    
    for i in range(n_steps):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Apply Gaussian weighting from neighboring points
        for j in range(n_steps):
            distance = abs(i - j)
            weight = np.exp(-0.5 * (distance / sigma)**2)
            weighted_sum += weight * f_values[j]
            weight_sum += weight
            
        smoothed.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)
    
    # Add some randomness for diversity while maintaining structure
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, n_steps)
    smoothed = np.array(smoothed) + noise
    smoothed = np.maximum(smoothed, 0.0)  # Ensure non-negative
    
    # Normalize to maintain reasonable magnitude
    total = np.sum(smoothed)
    if total > 0:
        smoothed = smoothed * n_steps / total
    
    return smoothed.tolist()

def multi_start_differential_evolution(n_steps, max_evaluations=2000):
    """Run multiple differential evolution runs with different initializations"""
    best_c2 = -float('inf')
    best_solution = None
    
    # Run multiple independent optimizations
    num_starts = 5
    
    for start_idx in range(num_starts):
        # Use different seed for each start
        np.random.seed(start_idx * 100 + 42)
        
        # Different initialization strategies for diversity
        if start_idx == 0:
            # Gaussian-weighted alternating pattern
            initial_guess = gaussian_weighted_alternating_initialization(n_steps)
        elif start_idx == 1:
            # Random initialization with some structure
            initial_guess = [np.random.random() * 0.5 + 0.25 for _ in range(n_steps)]
        else:
            # Simple alternating pattern
            initial_guess = [1.0 if i % 2 == 0 else 0.1 for i in range(n_steps)]
            # Add noise
            np.random.seed(start_idx)
            noise = np.random.normal(0, 0.03, n_steps)
            initial_guess = np.array(initial_guess) + noise
            initial_guess = np.maximum(initial_guess, 0.0)
            
        # Adjust population size based on iteration count
        popsize = min(20, max(5, n_steps // 100 + 5))
        
        # Set bounds for each parameter
        bounds = [(0.0, 10.0) for _ in range(n_steps)]
        
        try:
            # Use differential evolution with adapted parameters
            result = differential_evolution(
                evaluate_c2,
                bounds,
                maxiter=max_evaluations // num_starts,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=start_idx * 42,
                disp=False,
                polish=False  # Skip polishing to save time
            )
            
            # Check if this solution is better
            current_c2 = -result.fun
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = result.x.tolist()
                
        except Exception:
            continue
    
    return best_solution if best_solution is not None else gaussian_weighted_alternating_initialization(n_steps)

def local_refinement(initial_solution, max_iter=20):
    """Apply local optimization to refine the solution"""
    try:
        bounds = [(0.0, 10.0) for _ in range(len(initial_solution))]
        
        # Apply bounds constraint and local minimization
        result = minimize(
            evaluate_c2,
            initial_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter},
            tol=1e-6
        )
        
        if result.success:
            return result.x.tolist()
    except:
        pass
    return initial_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid optimization"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()
    
    try:
        # Dynamically choose problem size based on time budget
        n_steps = 1000  # Start with larger size for better solution quality
        max_evaluations = 2000  # Total evaluations across all runs
        
        # Multi-start differential evolution
        intermediate_solution = multi_start_differential_evolution(n_steps, max_evaluations)
        
        # Local refinement
        refined_solution = local_refinement(intermediate_solution, max_iter=20)
        
        # Ensure non-negative values
        refined_solution = [max(0, x) for x in refined_solution]
        
        # Normalize to avoid extreme values that might cause numerical issues
        total = sum(refined_solution)
        if total > 0:
            refined_solution = [x / total * len(refined_solution) for x in refined_solution]
        
        # Final check to ensure reasonable bounds
        refined_solution = [min(10.0, max(0.0, x)) for x in refined_solution]
        
        return refined_solution

    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Fallback due to error: {e}")
        return gaussian_weighted_alternating_initialization(500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
