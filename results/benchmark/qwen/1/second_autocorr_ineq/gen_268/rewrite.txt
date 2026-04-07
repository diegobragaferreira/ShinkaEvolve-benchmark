# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.special import sinc
from typing import List
import warnings
from numba import jit
import time

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
    """Fast norm computation using Numba with proper piecewise integration"""
    n = len(g_vals)
    l2_sq = 0.0
    for i in range(n-1):
        y1 = g_vals[i]
        y2 = g_vals[i+1]
        l2_sq += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    l1 = 0.0
    linf = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val
    return l2_sq, l1, linf

def compute_c2_score(f_vals):
    """Compute C2 score with error handling"""
    try:
        f_vals = np.maximum(f_vals, 0)
        g_vals = compute_autoconvolution_numba(f_vals)
        l2_sq, l1, linf = compute_norms_numba(g_vals)
        if l1 < 1e-15 or linf < 1e-15:
            return 0.0
        return l2_sq / (l1 * linf)
    except Exception:
        return 0.0

def create_basis_functions(n_steps, n_bases=10):
    """Create a set of basis functions that can represent step functions effectively"""
    # Create Gaussian basis functions
    bases = []
    centers = np.linspace(-0.5, 0.5, n_bases)
    widths = np.logspace(-2, 0, n_bases)  # Logarithmic spacing of widths
    
    for i in range(n_bases):
        # Create Gaussian basis with different centers and widths
        x = np.linspace(-0.5, 0.5, n_steps)
        basis = np.exp(-0.5 * ((x - centers[i]) / widths[i])**2)
        bases.append(basis)
    
    # Add sinc basis for oscillatory behavior
    x = np.linspace(-0.5, 0.5, n_steps)
    for i in range(n_bases // 2):
        # Sinc functions with different frequencies
        freq = 2 + i * 1.5
        basis = sinc(freq * x)
        # Normalize to reasonable range
        basis = (basis - np.min(basis)) / (np.max(basis) - np.min(basis) + 1e-10)
        bases.append(basis)
    
    return np.array(bases)

def generate_convex_representation(n_steps, n_bases=10):
    """Generate a convex combination of basis functions that form valid step function"""
    # Create basis functions
    bases = create_basis_functions(n_steps, n_bases)
    
    # Generate random weights that sum to 1 (convex combination)
    weights = np.random.random(n_bases)
    weights = weights / np.sum(weights)
    
    # Create step function as convex combination
    step_function = np.zeros(n_steps)
    for i in range(n_bases):
        step_function += weights[i] * bases[i]
    
    # Apply soft threshold to ensure minimal positivity
    step_function = np.maximum(step_function, 1e-10)
    
    # Normalize to sum to 1
    step_function = step_function / np.sum(step_function)
    
    return step_function.tolist()

def adaptive_convex_optimization():
    """Main optimization routine using convex optimization approach"""
    n_steps = 500
    max_iter = 200
    
    # Initialize with convex combination
    best_f = generate_convex_representation(n_steps)
    best_c2 = compute_c2_score(best_f)
    
    # Create basis functions once
    bases = create_basis_functions(n_steps)
    
    # Parameters for adaptive optimization
    learning_rates = [0.1, 0.05, 0.01, 0.005]
    current_lr = learning_rates[0]
    
    # Adaptive optimization loop
    for iteration in range(max_iter):
        # Sample several candidate solutions via perturbed convex combinations
        candidates = []
        candidate_scores = []
        
        # Generate multiple candidates with small variations
        for _ in range(20):
            # Create variation of current best
            weights = np.random.random(len(bases))
            weights = weights / np.sum(weights)
            
            # Add small random noise
            noise = np.random.normal(0, 0.05, len(bases))
            weights = weights + noise
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            # Create candidate function
            candidate = np.zeros(n_steps)
            for i in range(len(bases)):
                candidate += weights[i] * bases[i]
            
            # Ensure non-negativity and normalization
            candidate = np.maximum(candidate, 1e-10)
            candidate = candidate / np.sum(candidate)
            
            candidates.append(candidate.tolist())
            
        # Evaluate candidates
        for candidate in candidates:
            score = compute_c2_score(candidate)
            candidate_scores.append(score)
            
        # Find best candidate
        if len(candidate_scores) > 0:
            best_candidate_idx = np.argmax(candidate_scores)
            if candidate_scores[best_candidate_idx] > best_c2:
                best_c2 = candidate_scores[best_candidate_idx]
                best_f = candidates[best_candidate_idx]
                
                # Adapt learning rate based on success
                if iteration > 0 and iteration % 10 == 0:
                    current_lr = max(0.001, current_lr * 0.95)
    
    # Final local refinement with constrained optimization
    try:
        # Use scipy optimization with bounds
        def objective(x):
            return -compute_c2_score(x)
        
        bounds = [(1e-10, 1.0) for _ in range(n_steps)]
        # Use L-BFGS-B with multiple restarts
        result = minimize(
            objective,
            best_f,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}
        )
        
        if result.success:
            refined_f = np.maximum(result.x, 0)
            refined_f = refined_f / np.sum(refined_f)
            refined_c2 = compute_c2_score(refined_f)
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_f = refined_f.tolist()
    except Exception:
        pass
    
    return best_f

def heuristic_step_function_generation():
    """Generate step function using heuristic patterns derived from mathematical insights"""
    n_steps = 500
    
    # Create pattern with specific characteristics for good autoconvolution
    f = np.zeros(n_steps)
    
    # Create multi-scale structure that encourages flat convolution
    scales = [20, 40, 80, 160]
    scale_weights = [0.3, 0.25, 0.2, 0.25]  # Weighted contributions
    
    for s, w in zip(scales, scale_weights):
        if s < n_steps:
            # Create block structures at different scales
            num_blocks = n_steps // s
            for i in range(num_blocks):
                start = i * s
                end = min(start + s, n_steps)
                # Alternate between high and low values with some randomness
                if i % 2 == 0:
                    f[start:end] = 0.8 + np.random.random(end - start) * 0.2
                else:
                    f[start:end] = 0.1 + np.random.random(end - start) * 0.1
    
    # Add some smoothness with Gaussian envelope
    x = np.linspace(-1, 1, n_steps)
    envelope = np.exp(-0.5 * (x / 0.3)**2)
    f = f * envelope + envelope * 0.3
    
    # Ensure non-negativity and normalize
    f = np.maximum(f, 0)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def construct_function() -> list[float]:
    """Main function that constructs high C2 step function using adaptive convex optimization"""
    try:
        # Start with heuristic initialization
        heuristic_f = heuristic_step_function_generation()
        heuristic_c2 = compute_c2_score(heuristic_f)
        
        # Run adaptive convex optimization
        convex_f = adaptive_convex_optimization()
        convex_c2 = compute_c2_score(convex_f)
        
        # Return the better of the two approaches
        if convex_c2 > heuristic_c2:
            return convex_f
        else:
            return heuristic_f
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to uniform distribution
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")