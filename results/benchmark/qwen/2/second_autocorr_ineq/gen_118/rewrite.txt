# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal
import random
from typing import List, Tuple

def construct_function() -> List[float]:
    """
    Enhanced adaptive peak optimizer for maximizing C₂ constant.
    Combines multi-scale peak placement with sophisticated local optimization.
    """
    # Fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use a reasonable number of steps for good resolution
    n_steps = np.random.randint(2000, 8000)  # Variable range for exploration
    
    # Create x-axis grid
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Initialize function with multi-scale peak strategy
    f_values = np.zeros(n_steps)
    
    # Multi-scale peak placement strategy (inspired by PeakOptimizer)
    peaks = []
    
    # Scale 1: Fine scale peaks (high frequency details)
    fine_count = np.random.randint(10, 25)
    fine_positions = np.random.uniform(-0.05, 0.05, fine_count)
    fine_heights = np.random.uniform(1.5, 2.5, fine_count)
    fine_widths = np.random.uniform(0.005, 0.015, fine_count)
    
    # Scale 2: Medium scale peaks (mid frequency content)
    medium_count = np.random.randint(15, 30)
    medium_positions = np.random.uniform(-0.15, 0.15, medium_count)
    medium_heights = np.random.uniform(1.2, 2.0, medium_count)
    medium_widths = np.random.uniform(0.015, 0.035, medium_count)
    
    # Scale 3: Coarse scale peaks (low frequency structure)
    coarse_count = np.random.randint(8, 15)
    coarse_positions = np.random.choice([-0.2, -0.18, -0.16, -0.14, -0.12, -0.1,
                                       -0.08, -0.06, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
                                     coarse_count)
    coarse_heights = np.random.uniform(1.0, 1.8, coarse_count)
    coarse_widths = np.random.uniform(0.025, 0.055, coarse_count)
    
    # Combine all peak scales
    all_positions = np.concatenate([fine_positions, medium_positions, coarse_positions])
    all_heights = np.concatenate([fine_heights, medium_heights, coarse_heights])
    all_widths = np.concatenate([fine_widths, medium_widths, coarse_widths])
    
    # Filter peaks with minimum separation to avoid interference
    min_gap = 0.01  # Minimum spatial separation
    filtered_positions = []
    filtered_heights = []
    filtered_widths = []
    
    for pos, height, width in zip(all_positions, all_heights, all_widths):
        valid = True
        for existing_pos in filtered_positions:
            if abs(pos - existing_pos) < min_gap:
                valid = False
                break
        if valid:
            filtered_positions.append(pos)
            filtered_heights.append(height)
            filtered_widths.append(width)
    
    # Apply peaks with adaptive widths
    for pos, height, width in zip(filtered_positions, filtered_heights, filtered_widths):
        # Adjust height based on position to avoid very sharp peaks near edges
        if abs(pos) > 0.15:
            height *= 0.8
            
        # Apply adaptive width calculation based on position
        width_factor = 1.0 - abs(pos) / 0.25
        base_width = 0.02 + 0.03 * width_factor
        actual_width = max(0.01, base_width)  # Minimum width constraint
        
        # Create Gaussian-like peak
        gaussian = height * np.exp(-0.5 * ((x - pos) / actual_width)**2)
        f_values += gaussian
    
    # Add supplementary structure for better autoconvolution
    # Create beneficial interference patterns
    for i in range(0, n_steps, max(1, n_steps//40)):
        if np.random.random() > 0.8:
            bump_center = x[i] 
            bump_height = np.random.uniform(0.05, 0.3)
            bump_width = np.random.uniform(0.005, 0.015)
            bump = bump_height * np.exp(-0.5 * ((x - bump_center) / bump_width)**2)
            f_values += bump
    
    # Ensure non-negativity and normalization
    f_values = np.maximum(f_values, 0)
    if np.max(f_values) > 0:
        f_values = f_values / np.max(f_values) * 2.0
    
    # Apply smoothing to reduce sharp transitions
    window_size = min(51, max(3, n_steps // 150))
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        # Convolve with averaging kernel
        window = np.ones(window_size) / window_size
        f_values = np.convolve(f_values, window, mode='same')
    
    # Convert to list for initial function
    step_list = f_values.tolist()
    
    # Enhanced local optimization with multiple refinement passes
    def compute_autoconvolution_norms(func_vals: List[float]) -> Tuple[float, float, float]:
        """Compute the three norms needed for C2 calculation with improved accuracy."""
        f = np.array(func_vals)
        n_steps = len(f)
        
        # Autoconvolution using convolution
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Take middle portion

        # Adjust for correct length
        if len(g) > n_steps:
            g = g[:n_steps]
            
        # ||g||₂² (L2 norm squared) - more accurate piecewise integration
        # Using trapezoidal rule for better accuracy
        dx = 0.5 / (n_steps - 1) if n_steps > 1 else 0.5
        norm_2_sq = 0
        for i in range(len(g)-1):
            # Trapezoidal rule: integral of g^2 from i to i+1
            area = dx * (g[i]**2 + g[i+1]**2) / 2
            norm_2_sq += area

        # ||g||₁ (L1 norm) - via summation with proper normalization
        norm_1 = np.sum(np.abs(g)) * dx

        # ||g||∞ (infinity norm)
        norm_inf = np.max(np.abs(g))

        return norm_2_sq, norm_1, norm_inf

    def compute_c2(func_vals: List[float]) -> float:
        """Compute C₂ value with numerical stability."""
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(func_vals)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-15
        if norm_1 <= epsilon or norm_inf <= epsilon:
            return 0.0

        return norm_2_sq / (norm_1 * norm_inf)
    
    # Advanced peak refinement with multiple strategies
    def refine_peaks(original_func):
        # Simple hill climbing approach for peak refinement
        refined_func = np.array(original_func)
        current_c2 = compute_c2(refined_func)
        
        # Try small adjustments to various points
        best_func = refined_func.copy()
        best_c2 = current_c2
        
        # Multiple refinement passes for better convergence
        for iteration in range(3):
            # Sample several neighborhood adjustments
            for _ in range(150):  # Reduced iterations per pass but multiple passes
                test_func = refined_func.copy()
                
                # Select random point to adjust
                idx = random.randint(0, len(test_func) - 1)
                adjustment = random.uniform(-0.08, 0.08)  # Larger adjustment range
                
                # Apply adjustment
                test_func[idx] = max(0, test_func[idx] + adjustment)
                
                # Evaluate
                test_c2 = compute_c2(test_func)
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_func = test_func.copy()
            
            # Update refined_func for next iteration
            refined_func = best_func.copy()
        
        return best_func, best_c2
    
    # Apply local refinement with retry mechanism
    try:
        refined_func, refined_c2 = refine_peaks(step_list)
        
        # If improvement found, use refined version
        if refined_c2 > compute_c2(step_list):
            step_list = refined_func.tolist()
            
    except Exception:
        # Continue with original if refinement fails
        pass
    
    # Final optimization using a more sophisticated approach with multiple restarts
    def advanced_optimization(func_list, max_restarts=3):
        best_func = np.array(func_list)
        best_c2 = compute_c2(best_func)
        
        for restart in range(max_restarts):
            # Random perturb the function slightly
            perturbed_func = best_func.copy()
            for i in range(len(perturbed_func)):
                if random.random() > 0.8:  # 20% chance to perturb
                    perturbed_func[i] = max(0, perturbed_func[i] + random.uniform(-0.02, 0.02))
            
            # Run local refinement
            try:
                refined_func, refined_c2 = refine_peaks(perturbed_func.tolist())
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_func = np.array(refined_func)
            except Exception:
                continue
                
        return best_func.tolist(), best_c2
    
    # Final optimization pass
    try:
        final_func, final_c2 = advanced_optimization(step_list)
        if final_c2 > compute_c2(step_list):
            step_list = final_func
    except Exception:
        pass
    
    # Add final small noise for robustness
    final_noise = np.random.normal(0, 0.005, n_steps)
    final_func = np.array(step_list) + final_noise
    final_func = np.maximum(final_func, 0)
    
    return final_func.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")