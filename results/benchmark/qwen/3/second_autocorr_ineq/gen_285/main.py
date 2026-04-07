# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.fft import fft, ifft
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_sparse_convolution_norms(f_vals: np.ndarray, domain_length: float = 0.5) -> Tuple[float, float, float]:
    """
    Fast computation using sparse representation and analytical integration
    for the autoconvolution norms, with specialized handling for step functions.
    """
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    # Step width
    dx = domain_length / n
    
    # For step functions, we can compute exact convolutions with proper scaling
    # The autoconvolution g = f * f involves convolution sums with dx factors
    # Using the fact that convolution of piecewise constants gives piecewise linear functions
    
    # Create sparse representation of input (use only non-zero elements)
    # But since all values are used in convolution, we'll use dense approach but optimized
    
    # Compute autoconvolution using efficient nested loops
    # Result size is 2*n-1 
    g_size = 2 * n - 1
    g = np.zeros(g_size)
    
    # Optimized convolution - this is the key performance bottleneck
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_vals[i] * f_vals[j] * dx  # dx factor for proper integration
    
    # Compute norms using more accurate piecewise integration for g²
    g2_sq = 0.0
    for i in range(g_size - 1):
        # Trapezoidal-like integration for g² 
        y1, y2 = g[i], g[i+1]
        g2_sq += (dx/3) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))
    
    return g2_sq, g1, ginf

@jit(nopython=True)
def compute_c2_sparse(f_vals: np.ndarray) -> float:
    """Fast C2 computation"""
    g2_sq, g1, ginf = compute_sparse_convolution_norms(f_vals)
    
    if g1 <= 1e-15 or ginf <= 1e-15:
        return 0.0
    
    return g2_sq / (g1 * ginf)

def construct_sparse_multiscale_pattern(n_steps: int) -> List[float]:
    """
    Efficiently construct a multiscale pattern using mathematical properties
    for faster computation and better convergence.
    """
    # Use mathematical insight: combine sinusoidal components with proper normalization
    pattern = np.zeros(n_steps)
    
    # Generate components with different frequencies and amplitudes
    # This creates a pattern that naturally produces flat autoconvolutions
    frequencies = [1, 2, 3, 5, 7, 11]  # Prime frequencies for diversity
    amplitudes = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]  # Decreasing amplitudes
    
    x = np.linspace(0, 4*np.pi, n_steps)  # Extended domain for better sampling
    
    for freq, amp in zip(frequencies, amplitudes):
        component = amp * np.sin(freq * x / 4)
        # Ensure component fits in domain
        if len(component) > n_steps:
            component = component[:n_steps]
        pattern += component
    
    # Apply sigmoid to constrain to positive values and create smooth transitions
    pattern = 1.0 + np.tanh(pattern)  # Maps to [0, 2] range, then clip appropriately
    
    # Ensure non-negativity and normalize  
    pattern = np.maximum(pattern, 0.0)
    
    # Normalize to reasonable values  
    total_area = np.sum(pattern) * (0.5 / n_steps)
    if total_area > 0:
        pattern = pattern / total_area * 2.0
    
    # Add small random noise for diversity
    noise = np.random.normal(0, 0.05, n_steps)
    pattern = np.maximum(pattern + noise, 0.0)
    
    return pattern.tolist()

def adaptive_mathematical_optimization(f_init: List[float], max_iter: int = 100) -> List[float]:
    """
    Direct mathematical optimization approach instead of evolutionary methods.
    Uses gradient information and convex optimization techniques.
    """
    n = len(f_init)
    f_current = np.array(f_init, dtype=np.float64)
    
    def objective(x):
        # Return negative C2 for minimization (since we want to maximize C2)
        return -compute_c2_sparse(x)
    
    def objective_grad(x):
        """Compute gradient numerically with better finite difference"""
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            # Central difference
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Start with simple gradient descent
    learning_rate = 0.01
    momentum = 0.9
    velocity = np.zeros_like(f_current)
    
    best_f = f_current.copy()
    best_c2 = -objective(f_current)
    
    # Adaptive learning rate
    for i in range(max_iter):
        current_c2 = -objective(f_current)
        
        # Compute gradient
        try:
            grad = objective_grad(f_current)
        except:
            # If gradient computation fails, fall back to simple modification
            grad = np.random.randn(n) * 0.001
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        f_new = f_current + velocity
        
        # Clip to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # Evaluate new solution
        new_c2 = -objective(f_new)
        
        if new_c2 > current_c2:
            f_current = f_new
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_f = f_current.copy()
        else:
            # Reduce learning rate if no improvement
            learning_rate *= 0.99
            
        # Early stopping if improvement is minimal
        if i > 10 and abs(new_c2 - current_c2) < 1e-8:
            break
    
    return best_f.tolist()

def domain_decomposition_optimization(f_init: List[float], max_depth: int = 3) -> List[float]:
    """
    Apply domain decomposition where different regions are optimized differently
    based on curvature and behavior analysis.
    """
    f_current = np.array(f_init, dtype=np.float64)
    n = len(f_current)
    
    # Analyze the current function to determine interesting regions
    if n > 50:  # Only do decomposition for larger functions
        # Compute first differences to detect regions of interest
        diff = np.diff(f_current)
        # Find regions with high variation or flat areas
        var_regions = np.where(np.abs(diff) > np.std(diff) * 0.5)[0]
        
        if len(var_regions) > 0:
            # Optimize in chunks around variable regions
            chunk_size = max(10, n // 8)
            for chunk_start in range(0, n, chunk_size):
                # Process chunk in place
                chunk_end = min(chunk_start + chunk_size, n)
                chunk = f_current[chunk_start:chunk_end]
                
                # Apply mathematical optimization to this sub-region
                if len(chunk) > 5:  # Only optimize if enough points
                    # Temporarily optimize just this chunk
                    chunk_opt = adaptive_mathematical_optimization(chunk.tolist(), max_iter=20)
                    f_current[chunk_start:chunk_end] = chunk_opt
    
    return f_current.tolist()

def multi_resolution_refinement(f_init: List[float], coarse_to_fine: List[int] = None) -> List[float]:
    """
    Multi-resolution refinement starting from coarse resolution to capture global structure
    before refining to detail.
    """
    if coarse_to_fine is None:
        coarse_to_fine = [50, 100, 200, 500, 1000]
    
    f_current = np.array(f_init, dtype=np.float64)
    n = len(f_current)
    
    # Start with coarse resolution if needed
    if n < coarse_to_fine[0]:
        # Upsample
        upsampled = []
        for i in range(coarse_to_fine[0]):
            idx = int(i * (n - 1) / (coarse_to_fine[0] - 1)) if coarse_to_fine[0] > 1 else 0
            upsampled.append(f_current[idx])
        f_current = np.array(upsampled)
        n = len(f_current)
    
    # Process different resolutions
    for resolution in coarse_to_fine:
        if resolution > n:
            # Upsample to target resolution
            target = []
            for i in range(resolution):
                idx = int(i * (n - 1) / (resolution - 1)) if resolution > 1 else 0
                target.append(f_current[idx])
            f_current = np.array(target)
            n = len(f_current)
            # Refine slightly at each level
            f_current = adaptive_mathematical_optimization(f_current.tolist(), max_iter=50)
        else:
            # Downsample and refine if needed
            if resolution < n:
                downsampled = []
                for i in range(resolution):
                    idx = int(i * (n - 1) / (resolution - 1)) if resolution > 1 else 0
                    downsampled.append(f_current[idx])
                f_current = np.array(downsampled)
                n = len(f_current)
            
            # Refine at this resolution
            f_current = adaptive_mathematical_optimization(f_current.tolist(), max_iter=50)
    
    return f_current.tolist()

def construct_function() -> List[float]:
    """
    Main function that constructs a high-C2 step function using mathematical optimization approach.
    """
    # Fixed seeds for reproducibility
    np.random.seed(42)
    random_seed = 42
    
    # Start with mathematical pattern construction
    n_steps = 1000  # Increase resolution for better optimization
    
    # Create initial pattern using mathematical insights
    initial_f = construct_sparse_multiscale_pattern(n_steps)
    
    # Apply mathematical optimization directly
    optimized_f = adaptive_mathematical_optimization(initial_f, max_iter=150)
    
    # Apply domain decomposition for fine-grained optimization
    decomposed_f = domain_decomposition_optimization(optimized_f)
    
    # Apply multi-resolution refinement
    refined_f = multi_resolution_refinement(decomposed_f)
    
    # Final numerical refinement with local search
    final_f = adaptive_mathematical_optimization(refined_f, max_iter=50)
    
    # Ensure non-negativity one more time
    final_f = [max(0.0, x) for x in final_f]
    
    return final_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")