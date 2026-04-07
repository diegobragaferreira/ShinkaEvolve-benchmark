# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import time
import warnings

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation with improved numerical stability.
    """
    if not f_values:
        return 0.0, 1e-12, 1e-12
        
    # Convert to numpy array
    f = np.array(f_values, dtype=np.float64)
    
    # Validate input
    if len(f) < 1:
        return 0.0, 1e-12, 1e-12
        
    # Compute autoconvolution - this is the core operation
    g = convolve(f, f, mode='full')
    g = g[len(f)-1:]  # Keep only the relevant part
    
    # Compute norms with numerical stability checks
    g_abs = np.abs(g)
    
    # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
    g_sq = g_abs ** 2
    if len(g_sq) < 2:
        norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
    else:
        # Trapezoidal integration: sum((y[i] + y[i+1])/2 * delta_x)
        norm_2_sq = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0)
    
    # ||g||₁ (L1 norm) - sum of absolute values
    norm_1 = np.sum(g_abs)
    
    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs)
    
    # Numerical stability checks
    norm_2_sq = max(0.0, norm_2_sq)
    norm_1 = max(1e-12, norm_1)  # Avoid division by zero
    norm_inf = max(1e-12, norm_inf)  # Avoid division by zero
    
    return norm_2_sq, norm_1, norm_inf

def construct_quadratic_peaks(n_points=2000):
    """
    Construct a function with quadratic peaks that are easier to optimize
    """
    # Create domain
    x = np.linspace(-0.25, 0.25, n_points)
    
    # Initialize function
    f_values = np.zeros(n_points)
    
    # Number of quadratic peaks
    n_peaks = min(20, max(5, n_points // 100))
    
    # Generate peak parameters
    peak_positions = np.linspace(-0.23, 0.23, n_peaks)
    
    # For each peak, define a quadratic shape (we'll make these bell-shaped)
    for i, pos in enumerate(peak_positions):
        # Create a bell-shaped profile (approximating gaussian with quadratic for easy optimization)
        # Quadratic profile centered at pos with varying width
        width = 0.03 + 0.02 * np.sin(i * 0.5)  # Varying widths
        height = 0.8 + 0.4 * np.cos(i * 0.3)   # Varying heights
        
        # Quadratic form: ax^2 + bx + c, but we'll simplify to bell-like shape
        # A bell curve based on quadratic expression: exp(-(x-center)^2/(2*sigma^2))
        # However for optimization purposes we'll make it more amenable to calculus
        # Using a quadratic approximation: -a*(x-center)^2 + b
        # We'll construct this as: height * (1 - ((x-pos)/width)^2)^2 for |x-pos| <= width
        # But let's use a more direct approach with a smooth bell shape
        
        # Simple quadratic approximation for peak
        quad_form = np.exp(-0.5 * ((x - pos) / width)**2) 
        quad_form = height * quad_form / np.max(quad_form)  # normalize 
        f_values += quad_form
        
    # Apply smoothing to ensure smooth transitions
    # Simple moving average smoothing
    if n_points > 100:
        window = min(21, n_points // 50)
        if window % 2 == 0:
            window += 1
        f_values = np.convolve(f_values, np.ones(window)/window, mode='same')
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)
    
    # Normalize for stability 
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / (max_val * 1.2)
        
    return f_values.tolist()

def optimize_quadratic_peaks():
    """
    Direct optimization approach for quadratic peaks to maximize C2.
    """
    # Start with a reasonable base function
    n_points = 1000
    f_values = construct_quadratic_peaks(n_points)
    
    # Define objective function to minimize (negative C2)
    def objective(params):
        try:
            # Reshape params into function values (use piecewise linear for now)
            f = np.array(params)
            # Ensure non-negativity
            f = np.maximum(f, 0)
            
            # Compute autoconvolution and norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f)
            
            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 1e10  # Large penalty for invalid norms
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            
            # Return negative because we want to maximize C2
            return -c2 if not np.isnan(c2) and not np.isinf(c2) else 1e10
            
        except Exception as e:
            warnings.warn(f"Objective evaluation failed: {str(e)}")
            return 1e10
            
    # Use L-BFGS-B optimizer for better convergence properties
    # Start with the base function as initial guess
    x0 = np.array(f_values)
    
    # Minimize negative C2 (i.e., maximize C2)
    try:
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        if result.success:
            optimized_f = np.maximum(result.x, 0)
            return optimized_f.tolist()
        else:
            warnings.warn(f"Optimization failed: {result.message}")
    except Exception as e:
        warnings.warn(f"Optimization error: {str(e)}")
        
    # Fallback to base function if optimization fails
    return f_values

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses direct quadratic optimization approach.
    """
    start_time = time.time()
    
    # Try several strategies and pick best
    best_c2 = 0.0
    best_f = None
    
    # Strategy 1: Direct optimization of quadratic peaks
    try:
        f_values = optimize_quadratic_peaks()
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        if norm_1 > 1e-12 and norm_inf > 1e-12:
            c2 = norm_2_sq / (norm_1 * norm_inf)
            if c2 > best_c2:
                best_c2 = c2
                best_f = f_values
    except Exception as e:
        warnings.warn(f"Strategy 1 failed: {str(e)}")
    
    # Strategy 2: Generate more diverse initial peaks and optimize
    if best_f is None or best_c2 < 0.8:
        try:
            # Create diverse peaks with different shapes
            n_points = 2000
            x = np.linspace(-0.25, 0.25, n_points)
            f_values = np.zeros(n_points)
            
            # Create multiple types of quadratic features
            n_features = 15
            
            for i in range(n_features):
                # Different types of features
                if i % 3 == 0:  # Sharp peaks
                    pos = np.random.uniform(-0.23, 0.23)
                    width = 0.01 + 0.02 * np.random.random()
                    height = 0.5 + 0.5 * np.random.random()
                    peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                elif i % 3 == 1:  # Wider peaks
                    pos = np.random.uniform(-0.23, 0.23)
                    width = 0.03 + 0.04 * np.random.random()
                    height = 0.3 + 0.4 * np.random.random()
                    peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                else:  # Flatter features
                    pos = np.random.uniform(-0.23, 0.23)
                    width = 0.05 + 0.03 * np.random.random()
                    height = 0.2 + 0.3 * np.random.random() 
                    peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
                    
                f_values += peak
                
            # Normalize and smooth
            f_values = np.maximum(f_values, 0)
            max_val = np.max(f_values)
            if max_val > 0:
                f_values = f_values / (max_val * 1.5)
                
            # Apply smoothing
            if n_points > 100:
                window = max(3, min(21, n_points // 100))
                if window % 2 == 0:
                    window += 1
                f_values = np.convolve(f_values, np.ones(window)/window, mode='same')
                
            f_values = f_values.tolist()
            
            # Optimize
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
            if norm_1 > 1e-12 and norm_inf > 1e-12:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = f_values
                    
        except Exception as e:
            warnings.warn(f"Strategy 2 failed: {str(e)}")
    
    # Final fallback if no valid solution
    if best_f is None:
        # Simple uniform distribution
        n_points = 500
        best_f = [np.random.random() * 0.5 for _ in range(n_points)]
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")