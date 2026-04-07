# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def compute_autoconvolution_norms(f: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f_arr = np.array(f)
    
    # Compute autoconvolution g = f * f
    g = signal.convolve(f_arr, f_arr, mode='full')
    
    # Adjust indexing for correct convolution
    g = g[len(f_arr)-1:]
    
    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)
    
    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))
    
    return norm_2_sq, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)
    
    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def adaptive_step_function_initialization(n_steps: int) -> List[float]:
    """
    Create initial step function with adaptive construction
    """
    # Use multiple initialization strategies
    strategies = [
        lambda: np.random.exponential(0.5, n_steps),
        lambda: np.abs(np.random.normal(0, 0.3, n_steps)),
        lambda: np.random.gamma(2, 0.3, n_steps)
    ]
    
    # Choose one strategy randomly
    strategy = random.choice(strategies)
    base_func = strategy()
    
    # Normalize to reasonable scale
    base_func = base_func / np.max(base_func) * 1.5
    
    # Add some smoothing
    kernel = np.ones(5) / 5
    smooth_func = np.convolve(base_func, kernel, mode='same')
    
    # Ensure non-negativity
    smooth_func = np.clip(smooth_func, 0, None)
    
    # Return as list with proper scaling
    return smooth_func.tolist()

def refine_with_local_search(initial_f: List[float], max_iter: int = 20) -> List[float]:
    """
    Apply local search to improve the function
    """
    f_current = np.array(initial_f)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()
    
    # Simple local search with small perturbations
    for _ in range(max_iter):
        # Create neighbor by making small changes
        f_new = f_current.copy()
        
        # Choose random indices to modify
        indices_to_modify = np.random.choice(len(f_new), size=max(1, len(f_new) // 10), replace=False)
        
        for idx in indices_to_modify:
            # Small random perturbation
            perturbation = np.random.normal(0, 0.05 * np.mean(f_new))
            f_new[idx] = max(0, f_new[idx] + perturbation)
        
        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())
        
        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()
            
        f_current = f_new
    
    return best_f.tolist()

def construct_function() -> List[float]:
    """
    Construct step function with high C2 value using adaptive optimization approach
    """
    # Set up parameters
    np.random.seed(42)
    random.seed(42)
    
    # Try multiple random initializations with different strategies
    best_c2 = 0.0
    best_function = []
    
    # Multi-start approach with different population sizes
    population_sizes = [50, 100]
    
    for pop_size in population_sizes:
        # Generate multiple candidate functions
        candidates = []
        
        for i in range(pop_size):
            # Create function with adaptive initialization
            n_steps = max(100, min(5000, 1000 + i * 50))  # Vary number of steps
            
            # Create initial function
            f_init = adaptive_step_function_initialization(n_steps)
            
            # Refine with local search
            f_refined = refine_with_local_search(f_init, max_iter=10 + i // 10)
            
            # Evaluate
            c2_val = compute_c2(f_refined)
            
            candidates.append((c2_val, f_refined))
        
        # Select best from this population
        if candidates:
            best_in_pop = max(candidates, key=lambda x: x[0])
            if best_in_pop[0] > best_c2:
                best_c2 = best_in_pop[0]
                best_function = best_in_pop[1]
    
    # Final refinement using a more sophisticated approach
    if best_function:
        # Apply additional local search with larger neighborhood
        refined_final = refine_with_local_search(best_function, max_iter=50)
        final_c2 = compute_c2(refined_final)
        
        if final_c2 > best_c2:
            best_c2 = final_c2
            best_function = refined_final
    
    # Ensure we return at least some function
    if not best_function:
        # Fallback to simple construction
        best_function = [1.0] * 100
        
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
