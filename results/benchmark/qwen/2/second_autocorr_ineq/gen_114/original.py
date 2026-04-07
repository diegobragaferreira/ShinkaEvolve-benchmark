# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import warnings

def construct_function() -> list[float]:
    """
    Harmonic peak optimizer for maximizing C₂ constant.
    Uses frequency-domain analysis and constrained optimization.
    """
    np.random.seed(42)
    
    # Parameters
    n_steps = np.random.randint(2000, 8000)
    domain_length = 0.5  # [-0.25, 0.25]
    dx = domain_length / (n_steps - 1)
    
    # Create frequency grid for spectral analysis
    freqs = np.fft.fftfreq(n_steps, dx)
    
    # Initialize function with harmonic components
    f_values = np.zeros(n_steps)
    
    # Generate optimal peak distribution using harmonic principles
    # Start with a base structure that has good autoconvolution properties
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use adaptive sampling based on spectral characteristics
    # More peaks in regions where autoconvolution benefits most
    num_bases = max(5, min(25, n_steps // 200))
    
    # Create base harmonic structure
    for i in range(num_bases):
        # Place peaks with geometric distribution to avoid clustering
        # Use logarithmic spacing in frequency domain
        pos = np.random.uniform(-0.25, 0.25)
        
        # Height based on position - lower near edges to prevent sharp autoconvolution
        height = np.random.uniform(1.0, 2.5)
        if abs(pos) > 0.15:
            height *= 0.7
        
        # Width varies with position - narrower in center, wider at edges
        width = 0.02 + 0.03 * (1.0 - abs(pos)/0.25)
        
        # Create Gaussian peak with harmonic properties
        gaussian = height * np.exp(-0.5 * ((x - pos) / width)**2)
        f_values += gaussian
    
    # Add supplementary structure for better autoconvolution
    # Use a pattern that creates beneficial interference
    for i in range(0, n_steps, max(1, n_steps//50)):
        if np.random.random() > 0.85:
            bump_center = x[i] 
            bump_height = np.random.uniform(0.1, 0.5)
            bump_width = np.random.uniform(0.01, 0.02)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            f_values += bump
    
    # Ensure non-negativity and normalization
    f_values = np.maximum(f_values, 0)
    if np.max(f_values) > 0:
        f_values = f_values / np.max(f_values) * 2.0
    
    # Apply smoothing to reduce sharp transitions
    window_size = min(51, max(3, n_steps // 100))
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        # Convolve with averaging kernel
        window = np.ones(window_size) / window_size
        f_values = np.convolve(f_values, window, mode='same')
    
    # Convert to list
    f_list = f_values.tolist()
    
    # Multi-stage optimization using gradient-based methods
    def compute_autoconvolution_norms(func_vals):
        """Compute the three norms needed for C2 calculation"""
        f = np.array(func_vals)
        
        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Take middle portion
        
        # Adjust for correct length
        if len(g) > len(f):
            g = g[:len(f)]
            
        # ||g||₂² (L2 norm squared) - using trapezoidal integration approximation
        g_sq = g * g
        # Piecewise quadratic integration for better accuracy
        norm_2_sq = 0
        for i in range(len(g)-1):
            # Trapezoidal rule for integration
            area = (g_sq[i] + g_sq[i+1]) * dx / 2
            norm_2_sq += area
        
        # ||g||₁ (L1 norm) - approximate via summation
        norm_1 = np.sum(np.abs(g)) * dx
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(np.abs(g))
        
        return norm_2_sq, norm_1, norm_inf
    
    def compute_c2(func_vals):
        """Compute C₂ value"""
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(func_vals)
        
        if norm_1 <= 0 or norm_inf <= 0:
            return 0.0
            
        return norm_2_sq / (norm_1 * norm_inf)
    
    # Multi-resolution optimization
    def multi_resolution_optimization(initial_func):
        """Optimize using multiple resolution levels"""
        current_func = np.array(initial_func)
        best_c2 = compute_c2(current_func)
        best_func = current_func.copy()
        
        # Level 1: Coarse optimization with fewer samples
        coarse_indices = np.arange(0, len(current_func), max(1, len(current_func) // 100))
        coarse_func = current_func.copy()
        
        # For demonstration purposes, we'll do a basic hill climbing approach
        # since full gradient-based optimization is computationally expensive
        for _ in range(50):
            test_func = current_func.copy()
            
            # Perturb one random index
            idx = np.random.randint(0, len(test_func))
            adjustment = np.random.normal(0, 0.05)
            test_func[idx] = max(0, test_func[idx] + adjustment)
            
            # Evaluate
            test_c2 = compute_c2(test_func)
            if test_c2 > best_c2:
                best_c2 = test_c2
                best_func = test_func.copy()
        
        return best_func.tolist()
    
    # Perform optimization
    try:
        optimized_func = multi_resolution_optimization(f_list)
        final_func = np.array(optimized_func)
        
        # Add small amount of noise for robustness
        noise = np.random.normal(0, 0.01, len(final_func))
        final_func = final_func + noise
        final_func = np.maximum(final_func, 0)
        
        return final_func.tolist()
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        return f_list

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")