# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal
import random

def construct_function() -> list[float]:
    """
    Adaptive step function optimizer for maximizing C₂ constant.
    Improves upon previous versions through better peak distribution,
    efficient optimization, and enhanced numerical integration.
    """
    np.random.seed(42)
    
    # Parameters
    n_steps = 5000  # Fixed for consistency with AlphaEvolve
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Base function initialization
    base_function = np.zeros_like(x)
    
    # Adaptive peak generation with improved distribution
    num_peaks = np.random.randint(12, 25)  # More controlled peak count
    
    # Logarithmic spacing with strategic positioning
    log_positions = np.logspace(np.log10(0.01), np.log10(0.24), num_peaks)
    peak_positions = np.concatenate([log_positions, -log_positions[::-1]])
    peak_positions = peak_positions[peak_positions <= 0.25]
    peak_positions = peak_positions[peak_positions >= -0.25]
    
    # Implement min gap constraint (15% of domain width)
    min_gap = 0.15 * 0.5  
    safe_positions = []
    for pos in sorted(peak_positions):
        if not safe_positions or abs(pos - safe_positions[-1]) >= min_gap:
            safe_positions.append(pos)
    
    num_peaks = len(safe_positions)
    
    # Construct optimized peaks with position-dependent parameters
    for i in range(num_peaks):
        peak_center = safe_positions[i]
        
        # Position-based height adjustment
        if abs(peak_center) > 0.15:
            peak_height = np.random.uniform(1.0, 1.5)
        else:
            peak_height = np.random.uniform(1.2, 2.0)
            
        # Position-based width adjustment (narrower near center)
        if abs(peak_center) < 0.05:
            peak_width = np.random.uniform(0.015, 0.025)  # Narrower
        elif abs(peak_center) < 0.15:
            peak_width = np.random.uniform(0.025, 0.04)   # Medium
        else:
            peak_width = np.random.uniform(0.035, 0.06)   # Wider
            
        # Create Gaussian peak with optimized parameters
        gaussian_peak = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width)**2)
        base_function += gaussian_peak
    
    # Add supplementary structure for enhanced autoconvolution properties
    for i in range(0, len(x), max(1, len(x)//30)):  # More frequent structure
        if np.random.random() > 0.85:  # Increased probability (15%)
            bump_center = x[i]
            bump_height = np.random.uniform(0.1, 0.4)
            bump_width = np.random.uniform(0.008, 0.018)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            base_function += bump
    
    # Ensure non-negativity and normalize
    base_function = np.maximum(base_function, 0)
    if np.max(base_function) > 0:
        base_function = base_function / np.max(base_function) * 1.8  # Slightly increased
    
    # Apply smoothing with fixed window size for consistency
    window_size = 21  # Fixed window size for reproducibility
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        smoothed_function = signal.savgol_filter(base_function, window_size, 1)
        smoothed_function = np.maximum(smoothed_function, 0)
        base_function = smoothed_function
    
    # Convert to step values
    step_values = base_function.tolist()
    
    # Efficient local optimization focusing on key parameters
    def compute_autoconvolution_norms(func_vals):
        """Improved numerical integration for autoconvolution norms"""
        f = np.array(func_vals)
        
        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Take middle portion
        
        # Adjust for correct length
        if len(g) > len(f):
            g = g[:len(f)]
            
        # ||g||₂² using more accurate piecewise quadratic integration
        dx = 0.5 / (len(f) - 1) if len(f) > 1 else 0.5
        norm_2_sq = 0
        for i in range(len(g)-1):
            # Use trapezoidal rule for better accuracy: (y1^2 + y1*y2 + y2^2) * dx / 3
            # But we use trapezoidal for first-order integration
            area = dx * (g[i]**2 + g[i+1]**2) / 2
            norm_2_sq += area
            
        # ||g||₁ (L1 norm) - via summation with proper normalization
        norm_1 = np.sum(np.abs(g)) * dx

        # ||g||∞ (infinity norm)
        norm_inf = np.max(np.abs(g))

        return norm_2_sq, norm_1, norm_inf
    
    def compute_c2(func_vals):
        """Compute C₂ value with numerical stability"""
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(func_vals)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-15
        if norm_1 <= epsilon or norm_inf <= epsilon:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)
    
    # Simplified but effective local optimization
    def optimize_function(initial_func):
        """Faster optimization focusing on critical adjustments"""
        current_func = np.array(initial_func)
        best_c2 = compute_c2(current_func)
        best_func = current_func.copy()
        
        # Multi-pass optimization with decreasing step sizes
        for pass_num in range(2):
            step_size = 0.05 if pass_num == 0 else 0.02
            
            # Limited number of iterations for efficiency
            for _ in range(100 if pass_num == 0 else 50):
                test_func = current_func.copy()
                
                # Randomly select indices to modify
                indices_to_modify = random.sample(range(len(test_func)), 
                                                min(10, len(test_func) // 20))
                
                for idx in indices_to_modify:
                    adjustment = np.random.normal(0, step_size)
                    test_func[idx] = max(0, test_func[idx] + adjustment)
                
                # Evaluate
                test_c2 = compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()
                    
            current_func = best_func.copy()
            
        return best_func.tolist()
    
    # Apply optimization
    try:
        optimized_func = optimize_function(step_values)
        final_func = np.array(optimized_func)
        
        # Add minimal noise for robustness
        noise_level = 0.005  # Much smaller than before
        noisy_func = final_func + np.random.normal(0, noise_level, len(final_func))
        noisy_func = np.maximum(noisy_func, 0)
        
        return noisy_func.tolist()
        
    except Exception as e:
        return step_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")