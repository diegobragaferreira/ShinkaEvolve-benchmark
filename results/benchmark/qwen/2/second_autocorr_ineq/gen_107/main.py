# EVOLVE-BLOCK-START

import numpy as np
from typing import List, Tuple
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
    """Compute the autoconvolution g = f * f of step function f."""
    n = len(f_values)
    if n == 0:
        return np.array([])
    
    f_array = np.array(f_values)
    g = np.convolve(f_array, f_array, mode='full')
    g = g[n-1:-(n-1)] if n > 1 else g
    return g

def compute_norms(g_values: np.ndarray) -> tuple:
    """Compute the three required norms for C2 calculation."""
    if len(g_values) == 0:
        return 0.0, 0.0, 0.0
    
    # ||g||₂² using trapezoidal-like piecewise linear integration
    if len(g_values) <= 1:
        norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
    else:
        norm_2_sq = 0.0
        for i in range(len(g_values)-1):
            h = 1.0
            norm_2_sq += (h/3.0) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)
    
    # ||g||₁: L1 norm, approximated as sum(|g|) / (len(g) + 1) 
    if len(g_values) > 0:
        norm_1 = np.sum(np.abs(g_values)) / (len(g_values) + 1)
    else:
        norm_1 = 0.0
        
    # ||g||∞: Infinity-norm
    norm_inf = np.max(np.abs(g_values)) if len(g_values) > 0 else 0.0
    
    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 for given step function values."""
    g = compute_autoconvolution(f_values)
    norm_2_sq, norm_1, norm_inf = compute_norms(g)
    
    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def gaussian_peak_function(x: np.ndarray, peak_params: List[float]) -> np.ndarray:
    """Generate a function composed of multiple Gaussian peaks."""
    result = np.zeros_like(x)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        width = max(width, 1e-6)
        result += amp * np.exp(-0.5 * ((x - center) / width)**2)
    return result

def quadratic_peak_optimization(n_steps: int = 1000) -> List[float]:
    """
    Quadratic peak optimization approach that models the C2 maximization
    as a constrained quadratic optimization problem.
    """
    # Domain setup
    domain_width = 0.5
    domain_center = 0.0
    step_width = domain_width / n_steps
    
    # Use analytical approach: model peaks as quadratic contributions
    # Start with optimized peak distribution based on mathematical insights
    
    # Number of peaks chosen based on empirical analysis for good performance
    n_peaks = 7
    
    # Optimal peak positions using golden ratio spacing to avoid interference
    # This provides good distribution across the domain
    positions = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n_peaks):
        # Distribute peaks using golden ratio to avoid regular patterns
        ratio = (i * phi) % 1.0
        pos = domain_center + (ratio - 0.5) * domain_width
        positions.append(pos)
    
    # Peak amplitudes - start with equal values for simplicity, 
    # then optimize using first-order analysis
    amplitudes = [50.0] * n_peaks
    widths = [0.05] * n_peaks  # Fixed widths for simplicity
    
    # Combine into parameter vector for optimization
    peak_params = []
    for i in range(n_peaks):
        peak_params.extend([amplitudes[i], positions[i], widths[i]])
    
    # Domain points for evaluation
    domain_points = np.linspace(-domain_width/2, domain_width/2, n_steps)
    
    # Optimization using quadratic approximation of C2 behavior
    # This is a direct analytical approach rather than iterative search
    
    # Use the fact that for Gaussian peaks, the autoconvolution forms a predictable pattern
    # We can estimate optimal parameters analytically
    
    # Analytical initialization based on peak interaction theory
    # Peaks should be spaced to minimize destructive interference
    # and maximize constructive interference in the autoconvolution
    
    # Improve peak positions based on theoretical predictions
    # Better spacing pattern
    improved_positions = []
    
    # Create more sophisticated distribution
    for i in range(n_peaks):
        # Use Chebyshev nodes for better distribution
        if n_peaks == 1:
            improved_positions.append(0.0)
        else:
            # Use Chebyshev distribution which places points closer to edges
            # but distributes more evenly
            theta = np.pi * (2*i + 1) / (2 * n_peaks)
            # Transform to [-0.25, 0.25] range
            pos = -0.25 + 0.25 * (1 + np.cos(theta))  
            improved_positions.append(pos)
    
    # Reconstruct peak parameters with improved positions
    peak_params = []
    for i in range(n_peaks):
        peak_params.extend([50.0, improved_positions[i], 0.05])
    
    # Now refine using gradient-based approach on the quadratic approximation
    # Create function that computes C2 for given parameters
    def objective(params):
        """Compute negative C2 (since we want to maximize C2)"""
        # Extract parameters
        amps = params[::3]
        centers = params[1::3] 
        widths = params[2::3]
        
        # Reconstruct parameter list
        peak_list = []
        for i in range(len(amps)):
            peak_list.extend([amps[i], centers[i], widths[i]])
        
        # Generate function
        func_values = gaussian_peak_function(domain_points, peak_list)
        step_values = func_values.tolist()
        
        # Compute C2
        c2_val = compute_c2(step_values)
        
        # Return negative because we minimize
        return -c2_val
    
    # Simple optimization using scipy minimize with bounds
    bounds = []
    for i in range(n_peaks):
        # Amplitude bounds: 10 to 100
        bounds.append((10.0, 100.0))
        # Center bounds: [-0.25, 0.25]
        bounds.append((-0.25, 0.25))
        # Width bounds: 0.01 to 0.2
        bounds.append((0.01, 0.2))
    
    # Try optimization with different methods
    try:
        result = minimize(objective, peak_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 100})
        if result.success:
            # Use optimized parameters
            optimized_params = result.x
            peak_params = optimized_params.tolist()
    except:
        # Fall back to simple refinement if optimization fails
        pass
    
    # Final function generation
    func_values = gaussian_peak_function(domain_points, peak_params)
    step_values = func_values.tolist()
    
    # Apply final post-processing to ensure good quality
    # Ensure non-negative, smooth the function slightly
    step_values = [max(0, x) for x in step_values]
    
    # Smooth a bit by averaging with neighbors
    smoothed = []
    window = 5
    for i in range(len(step_values)):
        start_idx = max(0, i - window//2)
        end_idx = min(len(step_values), i + window//2 + 1)
        avg_val = np.mean(step_values[start_idx:end_idx])
        smoothed.append(avg_val)
    
    step_values = smoothed
    
    return step_values

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses quadratic peak optimization approach.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Use our specialized quadratic peak optimization
    try:
        # Try different resolutions to find best one
        resolutions = [500, 750, 1000, 1250]
        best_result = []
        best_c2 = 0.0
        
        for res in resolutions:
            result = quadratic_peak_optimization(res)
            c2_val = compute_c2(result)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_result = result
        
        if len(best_result) > 0:
            return best_result
        else:
            # Fallback to simple construction
            return [10.0] * 500
            
    except Exception as e:
        # If something goes wrong, fallback to simple uniform distribution
        return [10.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
