# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
from typing import List
import warnings

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

def gaussian_peak_function(x: np.ndarray, peak_positions: np.ndarray, peak_heights: np.ndarray) -> np.ndarray:
    """
    Create a function as sum of Gaussian peaks
    """
    result = np.zeros_like(x)
    sigma = 0.02  # Fixed standard deviation for Gaussian peaks
    
    for pos, height in zip(peak_positions, peak_heights):
        result += height * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
    
    return result

def build_gaussian_step_function(peak_positions: np.ndarray, peak_heights: np.ndarray, 
                               n_steps: int = 1000) -> List[float]:
    """
    Build a step function approximation from Gaussian peaks on [-1/4, 1/4]
    """
    # Define domain
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Generate Gaussian function
    func_values = gaussian_peak_function(x, peak_positions, peak_heights)
    
    # Ensure non-negativity
    func_values = np.maximum(func_values, 0)
    
    # Smooth the function to reduce numerical artifacts
    func_values = gaussian_filter1d(func_values, sigma=0.5)
    
    return func_values.tolist()

def objective_function(params, n_steps: int = 1000) -> float:
    """
    Objective function for optimization - negative C2 to minimize
    """
    # Parse parameters: alternating peak positions and heights
    n_peaks = len(params) // 2
    peak_positions = params[:n_peaks]
    peak_heights = params[n_peaks:]
    
    # Scale peak positions to [-0.25, 0.25]
    peak_positions = np.array(peak_positions) * 0.5 - 0.25
    
    # Ensure peak heights are non-negative
    peak_heights = np.maximum(peak_heights, 0)
    
    # Build function
    try:
        func_values = build_gaussian_step_function(peak_positions, peak_heights, n_steps)
        
        # Compute C2
        c2 = compute_c2(func_values)
        
        # Minimize negative C2
        return -c2
    except Exception:
        # If there's an error, return very negative value
        return -1e10

def optimize_gaussian_peaks(n_peaks: int, n_steps: int = 1000) -> List[float]:
    """
    Optimize Gaussian peaks for maximum C2
    """
    # Initialize random peak positions and heights
    initial_positions = np.random.uniform(0, 1, n_peaks)
    initial_heights = np.random.exponential(1.0, n_peaks)
    
    # Combine into single parameter vector
    initial_params = np.concatenate([initial_positions, initial_heights])
    
    # Set bounds: positions [-0.25, 0.25], heights [0, 10]
    bounds = []
    for _ in range(n_peaks):
        bounds.extend([(-0.25, 0.25)])  # Position bounds
    for _ in range(n_peaks):
        bounds.extend([(0, 10)])        # Height bounds
    
    # Use differential evolution for global optimization
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            args=(n_steps,),
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42
        )
        
        if result.success:
            # Extract optimal parameters
            n_peaks = len(result.x) // 2
            opt_positions = result.x[:n_peaks]
            opt_heights = result.x[n_peaks:]
            
            # Scale positions back to [-0.25, 0.25]
            opt_positions = opt_positions * 0.5 - 0.25
            
            # Build final function
            final_func = build_gaussian_step_function(opt_positions, opt_heights, n_steps)
            return final_func
    except Exception:
        pass
    
    # Fallback to simpler optimization
    try:
        # Use L-BFGS-B for local optimization
        result = minimize(
            objective_function,
            initial_params,
            args=(n_steps,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            n_peaks = len(result.x) // 2
            opt_positions = result.x[:n_peaks]
            opt_heights = result.x[n_peaks:]
            
            # Scale positions back to [-0.25, 0.25]
            opt_positions = opt_positions * 0.5 - 0.25
            
            # Build final function
            final_func = build_gaussian_step_function(opt_positions, opt_heights, n_steps)
            return final_func
    except Exception:
        pass
    
    # Return simple fallback
    return [1.0] * n_steps

def construct_function() -> List[float]:
    """
    Construct step function with high C2 value using Gaussian peak optimization
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Try different numbers of peaks to find good configuration
    best_c2 = 0.0
    best_function = []
    
    # Test different peak configurations
    peak_configs = [3, 5, 7, 10, 15, 20]
    
    for n_peaks in peak_configs:
        # Try with different numbers of steps to balance quality vs speed
        n_steps = min(2000, max(500, 1000 + n_peaks * 100))
        
        try:
            # Optimize for this configuration
            func = optimize_gaussian_peaks(n_peaks, n_steps)
            
            # Evaluate C2
            c2 = compute_c2(func)
            
            if c2 > best_c2:
                best_c2 = c2
                best_function = func
                
        except Exception as e:
            continue
    
    # If we didn't find anything, fall back to a simple approach
    if not best_function:
        # Create a simple symmetric step function
        n_steps = 1000
        func_values = np.ones(n_steps)
        # Apply a bit of smoothing and make it non-negative
        func_values = gaussian_filter1d(func_values, sigma=2.0)
        func_values = np.maximum(func_values, 0)
        best_function = func_values.tolist()
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
