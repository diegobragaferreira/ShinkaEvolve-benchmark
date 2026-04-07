# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import numba
from typing import List

@numba.jit(nopython=True)
def compute_autoconvolution_norms(f_values: np.ndarray) -> tuple:
    """
    Compute the three norms needed for C2 calculation using fast numba-compiled code
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Convolve f with itself to get autoconvolution g
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Trim to proper size (should be 2*n-1)
    g = g[n-1:2*n-1]
    
    # Compute norms
    g_abs = np.abs(g)
    
    # ||g||_2^2 using trapezoidal-like integration formula
    # For piecewise linear segments with heights y1, y2 and width h = 1/n
    # contribution is (h/3)(y1^2 + y1*y2 + y2^2)
    g2_squared = 0.0
    if len(g) >= 2:
        # Width between points
        h = 1.0 / (n - 1) if n > 1 else 1.0
        
        # First and last points contribute differently
        g2_squared += (h/3) * (g[0]**2 + g[0]*g[1] + g[1]**2)
        
        # Middle points
        for i in range(1, len(g)-1):
            g2_squared += (h/3) * (g[i-1]**2 + g[i-1]*g[i] + g[i]**2)
            
        # Last point (we already counted first part so just add last segment)
        g2_squared += (h/3) * (g[-2]**2 + g[-2]*g[-1] + g[-1]**2)
    
    # ||g||_1
    g1 = np.sum(g_abs)
    
    # ||g||_infty
    g_inf = np.max(g_abs)
    
    return g2_squared, g1, g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate the C2 value for given function values"""
    if len(f_values) < 2:
        return 0.0
    
    # Convert to numpy array
    f_array = np.array(f_values)
    
    # Compute norms
    g2_squared, g1, g_inf = compute_autoconvolution_norms(f_array)
    
    # Avoid division by zero
    if g1 <= 1e-12 or g_inf <= 1e-12:
        return 0.0
    
    # Compute C2
    c2 = g2_squared / (g1 * g_inf)
    return c2

def construct_initial_function(size: int) -> List[float]:
    """
    Construct an initial function using exponential decay to create balanced peaks
    """
    # Create a base function with exponential decay characteristics
    # This tends to produce more favorable autoconvolution profiles
    x = np.linspace(-0.5, 0.5, size)
    
    # Create exponentially decaying peaks around center
    # Using multiple exponential components for complexity
    f = np.zeros(size)
    
    # Add several peaks with different widths and amplitudes
    centers = np.linspace(-0.25, 0.25, min(5, size//4))
    widths = np.linspace(0.05, 0.15, len(centers))
    amplitudes = np.random.uniform(0.5, 1.5, len(centers))
    
    for center, width, amp in zip(centers, widths, amplitudes):
        # Gaussian-like decay
        f += amp * np.exp(-0.5 * ((x - center) / width)**2)
    
    # Normalize to have reasonable magnitude
    f = f / np.max(f) * 2.0
    
    # Apply soft thresholding to avoid very sharp transitions
    f = np.clip(f, 0, None)
    
    # Add some randomization to break symmetry
    noise = np.random.normal(0, 0.02, size)
    f = np.clip(f + noise, 0, None)
    
    return f.tolist()

def adaptive_evolution_step(current_f: List[float], iteration: int) -> List[float]:
    """
    Perform an adaptive evolution step
    """
    f_array = np.array(current_f)
    n = len(f_array)
    
    # Make a copy for modification
    new_f = f_array.copy()
    
    # Adaptive mutation based on iteration number
    if iteration % 5 == 0:
        # Global mutation - change several random elements
        indices = np.random.choice(n, size=min(5, n//4), replace=False)
        for idx in indices:
            # Add normally distributed noise
            new_f[idx] += np.random.normal(0, 0.1)
    else:
        # Local mutation - small changes to nearby elements
        start_idx = np.random.randint(0, n-2)
        for i in range(start_idx, min(start_idx+3, n)):
            new_f[i] += np.random.normal(0, 0.05)
    
    # Ensure non-negativity
    new_f = np.clip(new_f, 0, None)
    
    # Apply constraint-aware normalization to keep norms balanced
    # Check current state before normalization
    current_c2 = evaluate_c2(new_f.tolist())
    
    # If the function has high peak, reduce the peak height
    if current_c2 > 0.5:  # Threshold for when we detect problematic behavior
        # Apply gentle normalization to reduce peaks
        max_val = np.max(new_f)
        if max_val > 0:
            new_f = new_f * 0.95  # Reduce all by 5%
    
    return new_f.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Try different sizes to find a good starting point
    best_size = np.random.randint(100, 1000)
    
    # Generate initial function
    f_values = construct_initial_function(best_size)
    
    # Evaluate initial quality
    current_c2 = evaluate_c2(f_values)
    
    # Perform adaptive evolution for 50 iterations
    best_f = f_values[:]
    best_c2 = current_c2
    
    for iteration in range(50):
        # Apply evolution step
        new_f = adaptive_evolution_step(f_values, iteration)
        
        # Evaluate new function
        new_c2 = evaluate_c2(new_f)
        
        # Acceptance criteria
        if new_c2 > best_c2:
            best_f = new_f[:]
            best_c2 = new_c2
            f_values = new_f[:]
        elif np.random.random() < 0.1:  # Allow some bad moves occasionally
            f_values = new_f[:]
    
    # Final validation and cleanup
    final_c2 = evaluate_c2(best_f)
    
    # If we got a valid result, return the best one, otherwise return a fallback
    if final_c2 > 0.0:
        return best_f
    else:
        # Fallback to simpler construction
        return [np.random.random()] * np.random.randint(100, 1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
