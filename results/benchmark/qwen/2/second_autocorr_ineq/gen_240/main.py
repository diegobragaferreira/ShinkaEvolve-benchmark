# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal
import random
from typing import List
import time

def construct_function() -> List[float]:
    """
    High-performance adaptive peak optimizer using geometric spacing and feedback-driven refinement.
    """
    # Fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use a reasonable number of steps for good resolution
    n_steps = 5000  # Fixed at 5000 to ensure consistency with AlphaEvolve
    
    # Initialize with geometrically distributed peaks using feedback mechanism
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create base step function using adaptive geometric peak sampling
    step_values = np.zeros(n_steps)
    
    # Strategy: geometric sampling with dynamic adaptation based on performance feedback
    peak_count = 18  # Slightly more peaks than previous attempts for richer structure
    
    # Geometric peak positioning with adaptive compression
    # Start with a base compression that gets adjusted based on early results
    compression_factor = 0.85  # Initial guess for geometric progression
    
    # Track early performance to adapt compression factor
    early_performance_history = []
    
    # Generate peak positions using geometric progression with dynamic adjustment
    peak_positions = []
    peak_heights = []
    
    # First phase: generate geometrically spaced peaks
    base_positions = []
    if peak_count > 0:
        # Generate geometric progression from center outward
        if peak_count == 1:
            base_positions = [0.0]
        else:
            # Create geometric progression from 0.05 to 0.25 (with symmetry)
            geometric_progression = np.geomspace(0.05, 0.25, peak_count//2 + 1)[1:]
            # Mirror to get negative side
            base_positions = [0.0] + list(geometric_progression) + [-x for x in reversed(geometric_progression[:-1])]
            # Shuffle to prevent systematic bias
            random.shuffle(base_positions)
    
    # Second phase: refine positions and add heights with performance feedback
    final_positions = []
    final_heights = []
    
    # Use a multi-stage approach for better structure
    for i in range(peak_count):
        # Position based on geometric progression but allow some randomness
        if i < len(base_positions):
            pos = base_positions[i]
        else:
            # Fallback to random if needed
            pos = np.random.uniform(-0.25, 0.25)
        
        # Height determined by strategic placement
        if abs(pos) < 0.1:  # Central region
            height = np.random.uniform(2.0, 3.0)
        elif abs(pos) < 0.15:  # Mid region
            height = np.random.uniform(1.5, 2.5)
        else:  # Outer region
            height = np.random.uniform(1.0, 2.0)
        
        final_positions.append(pos)
        final_heights.append(height)
    
    # Apply peaks with optimized widths based on position and performance expectations
    for pos, height in zip(final_positions, final_heights):
        # Width varies with distance from center and expected interaction
        width_factor = 1.0 - abs(pos) / 0.25
        # Base width with performance-adjusted scaling
        base_width = 0.02 + 0.03 * width_factor
        # Add small variance to prevent perfect symmetry
        width = max(0.01, base_width + np.random.normal(0, 0.005))
        
        # Create Gaussian-like peak
        peak = height * np.exp(-0.5 * ((x - pos) / width)**2)
        step_values += peak
    
    # Ensure non-negative values
    step_values = np.maximum(step_values, 0)
    
    # Normalize to prevent extreme values
    if np.max(step_values) > 0:
        step_values = step_values / np.max(step_values) * 2.0
    
    # Add some random noise to ensure variety and escape local minima
    noise_level = 0.008
    step_values += np.random.normal(0, noise_level, n_steps)
    step_values = np.maximum(step_values, 0)
    
    # Convert to list form for output
    step_list = step_values.tolist()
    
    # Multi-resolution refinement with early termination
    def compute_c2(func):
        """Compute C2 using precise trapezoidal integration matching evaluator"""
        f = np.array(func)
        
        # Autoconvolution g = f * f
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Take middle portion
        
        # Truncate if necessary to match original length
        if len(g) > len(func):
            g = g[:len(func)]
        
        # Compute the precise norms per evaluator specification
        # ||g||₂² using trapezoidal-like integration (h/3)(y1² + y1*y2 + y2²)
        dx = 0.5 / (len(func) - 1) if len(func) > 1 else 0.5
        norm_2_sq = 0
        for i in range(len(g)-1):
            y1 = g[i]
            y2 = g[i+1]
            norm_2_sq += (dx / 3.0) * (y1*y1 + y1*y2 + y2*y2)
            
        # ||g||₁ (normalized sum)
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        return norm_2_sq / (norm_1 * norm_inf)
    
    # Multi-phase refinement with early stopping
    max_iterations = 150
    best_c2 = compute_c2(step_list)
    best_function = step_list.copy()
    
    # Phase 1: Coarse-grained adjustments
    for iter_num in range(max_iterations):
        test_func = np.array(step_list)
        
        # Select a few indices to modify (coarse adjustment)
        indices_to_modify = random.sample(range(len(test_func)), min(100, len(test_func)//100))
        for idx in indices_to_modify:
            # Larger perturbations for coarse tuning
            adjustment = random.uniform(-0.08, 0.08)
            test_func[idx] = max(0, test_func[idx] + adjustment)
        
        test_c2 = compute_c2(test_func.tolist())
        if test_c2 > best_c2:
            best_c2 = test_c2
            best_function = test_func.tolist()
            if test_c2 > 0.95:  # Early stopping if very good
                break
    
    # Phase 2: Fine-grained refinement
    step_list = best_function.copy()
    for iter_num in range(100):  # Fewer iterations for fine-tuning
        test_func = np.array(step_list)
        
        # Select a small subset for fine adjustment
        indices_to_modify = random.sample(range(len(test_func)), min(30, len(test_func)//200))
        for idx in indices_to_modify:
            # Small perturbations for fine-tuning
            adjustment = random.uniform(-0.02, 0.02)
            test_func[idx] = max(0, test_func[idx] + adjustment)
        
        test_c2 = compute_c2(test_func.tolist())
        if test_c2 > best_c2:
            best_c2 = test_c2
            best_function = test_func.tolist()
    
    # Final stage: Add robustness noise
    final_func = np.array(best_function)
    final_noise = np.random.normal(0, 0.003, len(final_func))
    final_func = final_func + final_noise
    final_func = np.maximum(final_func, 0)
    
    return final_func.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")