# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List

def construct_function() -> List[float]:
    """
    High-performance step function construction using adaptive peak sampling
    that directly optimizes the discrete function representation.
    """
    # Fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use a reasonable number of steps for good resolution
    n_steps = 5000  # Fixed at 5000 to ensure consistency with AlphaEvolve
    
    # Initialize with a simple symmetric peak structure
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create base step function using adaptive peak sampling
    step_values = np.zeros(n_steps)
    
    # Strategy: sample peak locations and heights adaptively
    # Use a grid-based approach with adaptive density
    peak_count = 25  # Increased peak count for better optimization potential
    
    # Place peaks using strategic spacing to avoid interference
    peak_positions = []
    peak_heights = []
    
    # First, place some central peaks with varying heights
    for i in range(peak_count):
        # Adaptive positioning with more peaks near center
        if i < peak_count // 3:
            # Central region with higher density
            pos = np.random.uniform(-0.08, 0.08)
            height = np.random.uniform(1.8, 3.0)
        elif i < 2 * peak_count // 3:
            # Middle region
            pos = np.random.uniform(-0.15, 0.15)
            height = np.random.uniform(1.5, 2.5)
        else:
            # Outer regions with lower density
            pos = np.random.choice([-0.2, -0.18, -0.16, -0.14, -0.12, 
                                  -0.1, -0.08, -0.06, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
            height = np.random.uniform(1.2, 2.2)
            
        peak_positions.append(pos)
        peak_heights.append(height)
    
    # Apply peaks with optimized widths based on position
    for pos, height in zip(peak_positions, peak_heights):
        # Width varies with distance from center - narrower near center, wider at edges
        width_factor = 1.0 - abs(pos) / 0.25
        # Use a more refined width calculation with minimum constraints
        base_width = 0.02 + 0.03 * width_factor
        width = max(0.01, base_width)  # Minimum width constraint
        
        # Create Gaussian-like peak
        peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
        step_values += peak
    
    # Ensure non-negative values
    step_values = np.maximum(step_values, 0)
    
    # Normalize to prevent extreme values
    if np.max(step_values) > 0:
        step_values = step_values / np.max(step_values) * 2.0
    
    # Add some random noise to ensure variety
    noise_level = 0.015
    step_values += np.random.normal(0, noise_level, n_steps)
    step_values = np.maximum(step_values, 0)
    
    # Convert to list form for output
    step_list = step_values.tolist()
    
    # Enhanced local optimization using a hybrid approach
    def compute_c2(func):
        # Compute autoconvolution g = f * f
        g = np.convolve(func, func, mode='full')
        g = g[len(g)//2:]  # Take positive part
        
        # Truncate if necessary to match original length
        if len(g) > len(func):
            g = g[:len(func)]
            
        # Compute norms using the exact specification
        norm_2_sq = np.sum(g**2) * (0.5 / len(func))  # Approximate integral
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
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