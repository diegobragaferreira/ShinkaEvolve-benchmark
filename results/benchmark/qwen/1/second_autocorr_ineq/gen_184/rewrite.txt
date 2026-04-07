# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import minimize
from scipy.linalg import block_diag
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')

# Global constants
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 1000
MIN_STEPS = 100
MAX_STEPS = 5000
INITIAL_REFINEMENT_ITERATIONS = 50
FINAL_REFINEMENT_ITERATIONS = 30

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Autoconvolution using discrete convolution formula
    g = np.zeros(2*n - 1)

    # Manual convolution loop for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms efficiently
    """
    n = len(g_vals)

    # L1 norm approximation (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])

    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    for i in range(n):
        l2_sq_norm += g_vals[i] * g_vals[i]

    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val

    return l1_norm, l2_sq_norm, linf_norm

@jit(nopython=True)
def compute_c2_numba(f_vals, dx):
    """
    Compute C2 value using optimized numba functions
    """
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals, dx)

    # Compute norms
    l1, l2_sq, linf = compute_norms_numba(g_vals)

    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0

    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function(f_vals):
    """
    Evaluate a step function and return C2 value
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute C2 value
        c2 = compute_c2_numba(f_vals, dx)
        return c2
    except Exception as e:
        return 0.0

def compute_gradient_descent_step(f_vals, learning_rate, previous_grad=None, momentum=0.9):
    """
    Compute a gradient descent step with adaptive momentum and learning rate adjustment
    """
    # This is a simplified version - in practice would compute true gradients
    # For adaptive step, we'll use a proxy based on local search patterns
    
    n = len(f_vals)
    step_direction = np.zeros(n)
    
    # Simple gradient approximation using finite differences
    epsilon = 1e-6
    for i in range(n):
        # Forward difference approximation
        f_new = f_vals.copy()
        f_new[i] = max(0.0, f_vals[i] + epsilon)
        c2_plus = evaluate_step_function(f_new)
        
        # Backward difference
        f_new = f_vals.copy()
        f_new[i] = max(0.0, f_vals[i] - epsilon)
        c2_minus = evaluate_step_function(f_new)
        
        # Gradient estimate
        grad_i = (c2_plus - c2_minus) / (2 * epsilon)
        step_direction[i] = grad_i
    
    # Apply momentum if previous gradient available
    if previous_grad is not None:
        step_direction = momentum * previous_grad + (1 - momentum) * step_direction
    
    # Adaptive learning rate based on gradient magnitude
    grad_norm = np.linalg.norm(step_direction)
    if grad_norm > 1e-10:
        adaptive_lr = min(learning_rate / grad_norm, 1.0)
    else:
        adaptive_lr = learning_rate
        
    # Compute new values
    new_vals = f_vals - adaptive_lr * step_direction
    
    # Ensure non-negativity
    new_vals = np.clip(new_vals, 0, None)
    
    # Normalize to maintain scale
    if np.sum(new_vals) > 0:
        new_vals = new_vals / np.sum(new_vals) * len(new_vals)
    
    return new_vals

def adaptive_convex_optimization(initial_solution):
    """
    Adaptive convex optimization approach that iteratively refines the solution
    """
    # Initialize tracking variables
    current_solution = np.array(initial_solution)
    previous_solution = None
    previous_c2 = evaluate_step_function(current_solution)
    learning_rate = 0.01
    momentum = 0.9
    patience_counter = 0
    max_patience = 10
    
    # Track convergence
    convergence_history = []
    
    # Maximum iterations based on time budget
    max_iter = 1000  # Adjusted for time constraints
    
    for iteration in range(max_iter):
        # Check time limit
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break
            
        # Store history for convergence tracking
        convergence_history.append(previous_c2)
        if len(convergence_history) > 20:
            convergence_history.pop(0)
        
        # Compute adaptive step using gradient descent
        new_solution = compute_gradient_descent_step(
            current_solution, 
            learning_rate, 
            previous_solution,
            momentum
        )
        
        # Evaluate new solution
        new_c2 = evaluate_step_function(new_solution)
        
        # Check improvement
        if new_c2 > previous_c2:
            # Accept new solution
            current_solution = new_solution
            previous_c2 = new_c2
            patience_counter = 0
        else:
            patience_counter += 1
            # Reduce learning rate if no improvement
            if patience_counter > 5:
                learning_rate *= 0.9
                patience_counter = 0
        
        # Early stopping based on convergence
        if len(convergence_history) >= 10:
            recent_improvement = np.mean(convergence_history[-5:]) - np.mean(convergence_history[-10:-5])
            if recent_improvement < 1e-8:
                break
                
        # Update previous solution for momentum
        previous_solution = new_solution
        
        # Occasionally adjust the solution structure
        if iteration % 100 == 0 and iteration > 0:
            # Periodic refinement with local search
            refined_solution = local_refinement(current_solution)
            refined_c2 = evaluate_step_function(refined_solution)
            if refined_c2 > previous_c2:
                current_solution = refined_solution
                previous_c2 = refined_c2
    
    return current_solution.tolist()

def local_refinement(solution):
    """
    Perform local refinement on the solution with improved optimization strategies
    """
    # Try multiple optimization strategies
    strategies = [
        lambda x: simple_local_search(x),
        lambda x: gradient_based_refinement(x),
    ]
    
    best_solution = solution.copy()
    best_c2 = evaluate_step_function(best_solution)
    
    for strategy in strategies:
        try:
            refined = strategy(best_solution)
            refined_c2 = evaluate_step_function(refined)
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined
        except:
            continue
            
    return best_solution

def simple_local_search(solution):
    """
    Simple local search with neighborhood exploration
    """
    current = np.array(solution)
    n = len(current)
    best_c2 = evaluate_step_function(current)
    
    # Try small perturbations
    for i in range(n):
        for delta in [0.01, 0.05, 0.1]:
            # Test increasing value
            test = current.copy()
            test[i] = max(0, current[i] + delta)
            test_c2 = evaluate_step_function(test)
            if test_c2 > best_c2:
                best_c2 = test_c2
                current = test.copy()
                
            # Test decreasing value
            test = current.copy()
            test[i] = max(0, current[i] - delta)
            test_c2 = evaluate_step_function(test)
            if test_c2 > best_c2:
                best_c2 = test_c2
                current = test.copy()
    
    return current.tolist()

def gradient_based_refinement(solution):
    """
    Gradient-based refinement using approximate gradients
    """
    n = len(solution)
    current = np.array(solution)
    
    # Use a simple gradient approximation with random directions
    for _ in range(10):  # Limited iterations for speed
        direction = np.random.randn(n)
        direction = direction / np.linalg.norm(direction)
        
        # Evaluate along the direction
        epsilon = 1e-4
        c2_current = evaluate_step_function(current)
        
        # Forward step
        forward = current + epsilon * direction
        c2_forward = evaluate_step_function(forward)
        
        # Backward step
        backward = current - epsilon * direction
        c2_backward = evaluate_step_function(backward)
        
        # Estimate gradient
        estimated_grad = (c2_forward - c2_backward) / (2 * epsilon)
        
        # Step towards improvement
        if estimated_grad > 0:
            current = current + 0.01 * estimated_grad * direction
            
            # Clip to non-negative
            current = np.clip(current, 0, None)
            
            # Normalize
            if np.sum(current) > 0:
                current = current / np.sum(current) * len(current)
    
    return current.tolist()

def create_convex_initialization(n_steps):
    """
    Create convex-initialization with mathematically informed structure
    """
    # This creates a structured pattern designed to promote good convolution properties
    pattern = np.zeros(n_steps)
    
    # Create a base pattern that is inherently well-suited for convolution
    # This is inspired by the structure that produces flat convolution results
    
    # Phase 1: Create a series of smooth regions
    segments = max(3, n_steps // 50)
    segment_width = n_steps // segments
    
    for i in range(segments):
        start_idx = i * segment_width
        end_idx = min((i + 1) * segment_width, n_steps)
        
        # Alternate between high and low values to create structure
        if i % 2 == 0:
            # High values
            pattern[start_idx:end_idx] = 1.0 + np.random.random(end_idx - start_idx) * 0.5
        else:
            # Low values
            pattern[start_idx:end_idx] = 0.2 + np.random.random(end_idx - start_idx) * 0.3
    
    # Add a smooth envelope to avoid sharp discontinuities
    smoothed = np.zeros(n_steps)
    for i in range(n_steps):
        # Weighted average of nearby points
        window_size = min(10, n_steps // 20)
        start_w = max(0, i - window_size)
        end_w = min(n_steps, i + window_size + 1)
        smoothed[i] = np.mean(pattern[start_w:end_w])
    
    # Add Gaussian smoothing
    x = np.linspace(-1, 1, n_steps)
    gaussian = np.exp(-0.5 * (x / 0.3)**2)
    smoothed = smoothed * 0.7 + gaussian * 0.3
    
    # Ensure non-negativity and normalize
    smoothed = np.clip(smoothed, 0, np.inf)
    if np.sum(smoothed) > 0:
        smoothed = smoothed / np.sum(smoothed) * n_steps
        
    return smoothed

def multi_resolution_optimization():
    """
    Multi-resolution optimization that starts coarse and refines
    """
    # Initialize at coarse resolution
    coarse_resolution = max(100, DEFAULT_STEPS // 10)
    coarse_solution = create_convex_initialization(coarse_resolution)
    
    # Optimize at coarse level
    coarse_optimized = adaptive_convex_optimization(coarse_solution)
    
    # Refine to medium resolution
    medium_resolution = max(200, DEFAULT_STEPS // 5)
    medium_solution = create_convex_initialization(medium_resolution)
    
    # Interpolate coarse solution to medium
    coarse_array = np.array(coarse_optimized)
    medium_array = np.interp(np.linspace(0, len(coarse_array)-1, medium_resolution), 
                            np.arange(len(coarse_array)), coarse_array)
    medium_solution = medium_array.tolist()
    
    # Optimize medium solution
    medium_optimized = adaptive_convex_optimization(medium_solution)
    
    # Finally refine to target resolution
    fine_solution = create_convex_initialization(DEFAULT_STEPS)
    
    # Interpolate medium solution to fine
    medium_array = np.array(medium_optimized)
    fine_array = np.interp(np.linspace(0, len(medium_array)-1, DEFAULT_STEPS), 
                          np.arange(len(medium_array)), medium_array)
    fine_solution = fine_array.tolist()
    
    # Final optimization
    final_solution = adaptive_convex_optimization(fine_solution)
    
    return final_solution

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value.
    Uses adaptive convex optimization with multi-resolution refinement.
    """
    global start_time
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Use multi-resolution convex optimization
        result = multi_resolution_optimization()
        
        # Final local refinement
        result = local_refinement(result)
        
        # Evaluate final result
        final_c2 = evaluate_step_function(result)
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Print debug info
        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {final_c2:.6f}")
        
        return result
        
    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * DEFAULT_STEPS
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")