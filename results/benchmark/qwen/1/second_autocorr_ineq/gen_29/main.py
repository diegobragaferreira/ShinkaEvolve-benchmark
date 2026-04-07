# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import math

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive convex optimization."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Determine number of steps (within reasonable bounds)
    n_steps = np.random.randint(1000, 5000)
    
    # Initialize with a sophisticated pattern that promotes good C₂ values
    # Start with a combination of high and low regions to encourage convolution energy concentration
    initial_f = []
    
    # Create a pattern that balances uniformity and variation 
    # to promote favorable autoconvolution properties
    segment_length = max(1, n_steps // 20)  # Create about 20 segments
    
    for i in range(n_steps):
        # Create segments with alternating high/low values
        segment_idx = i // segment_length
        if segment_idx % 2 == 0:
            # High amplitude segments
            amplitude = np.random.uniform(0.7, 1.0)
        else:
            # Low amplitude segments  
            amplitude = np.random.uniform(0.0, 0.3)
        initial_f.append(amplitude)
    
    # Convert to numpy array for easier manipulation
    f_values = np.array(initial_f)
    
    # Normalize to ensure values are reasonable
    max_val = np.max(f_values)
    if max_val > 0:
        f_values = f_values / max_val
    
    # Apply a more sophisticated optimization approach
    # Use gradient-based optimization with constraint handling
    def compute_c2(f_vec):
        # Compute autoconvolution using discrete convolution
        g = convolve(f_vec, f_vec, mode='full')
        g = g[len(f_vec)-1:]  # Take the valid part
        
        # Compute norms
        g_squared = g ** 2
        norm_2_sq = np.sum(g_squared)
        norm_1 = np.sum(np.abs(g))
        norm_inf = np.max(np.abs(g))
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0
            
        return norm_2_sq / (norm_1 * norm_inf)
    
    # Create a smooth approximation that better reflects the discrete nature
    def smooth_function(x, alpha=10.0):
        """Smooth approximation to step function with tunable sharpness"""
        return 1.0 / (1.0 + np.exp(-alpha * (x - 0.5)))
    
    # Create an objective function for optimization
    def objective(params):
        # Reshape and normalize parameters to [0,1] range
        f_normalized = np.clip(params, 0, 1)
        
        # Apply softmax-like transformation for smooth transitions  
        # while maintaining the overall shape characteristics
        if len(f_normalized) > 1:
            # Apply exponential transform to create sharper transitions
            exp_params = np.exp(3 * f_normalized)
            f_transformed = exp_params / np.sum(exp_params) * len(f_normalized)
        else:
            f_transformed = f_normalized
            
        # Ensure non-negativity and normalization
        f_final = np.maximum(f_transformed, 0)
        
        # Compute C2 value
        try:
            c2 = compute_c2(f_final)
        except:
            c2 = 0.0
            
        # Return negative because we want to maximize C2
        return -c2
    
    # Initial guess
    x0 = f_values.copy()
    
    # Use a combination of different optimization strategies 
    try:
        # First attempt: L-BFGS-B with bounds
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=[(0, 1)] * len(x0),
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )
        
        if result.success:
            optimized_f = np.clip(result.x, 0, 1)
        else:
            optimized_f = x0
            
    except:
        # Fallback to simpler approach if complex optimization fails
        optimized_f = x0
    
    # Final refinement through iterative adjustments
    best_c2 = 0.0
    best_f = optimized_f.copy()
    
    # Try several refinements with different approaches
    for _ in range(3):
        # Small random perturbations
        perturbed = np.clip(optimized_f + np.random.normal(0, 0.01, len(optimized_f)), 0, 1)
        
        try:
            c2_val = compute_c2(perturbed)
            if c2_val > best_c2:
                best_c2 = c2_val
                best_f = perturbed.copy()
        except:
            pass
    
    # Ensure final function has the right shape
    final_f = best_f.copy()
    
    # Normalize once more to maintain consistent scale
    max_val = np.max(final_f)
    if max_val > 0:
        final_f = final_f / max_val
    
    # Convert to list and return
    return final_f.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")