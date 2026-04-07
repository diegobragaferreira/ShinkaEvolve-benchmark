# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import time
from typing import List, Tuple, Optional

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation using piecewise linear integration.
    f_values: list of step heights
    Returns: ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array for easier manipulation
    f = np.array(f_values, dtype=np.float64)
    
    # Handle edge case of all zeros
    if np.sum(np.abs(f)) == 0:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4] with equal spacing
    n_steps = len(f)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n_steps

    # Compute autoconvolution using discrete convolution
    g = np.convolve(f, f, mode='full')
    # Trim g to the correct size (this accounts for the convolution)
    g = g[len(f)-1:2*len(f)-1]

    # Compute L2 norm squared using piecewise linear integration
    # For each pair of adjacent points, integrate quadratic function
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        # Trapezoidal-like integration for quadratic function
        # Using formula for integral of ax^2 + bx + c over [x0,x1]
        # But here we approximate with piecewise linear segments
        # So we use: (dx/3)(y0^2 + y0*y1 + y1^2)
        y0, y1 = g[i], g[i+1]
        norm_2_squared += (dx/3) * (y0**2 + y0*y1 + y1**2)

    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g))

    # Infinity norm
    norm_inf = np.max(np.abs(g))

    # Handle numerical edge cases
    if norm_1 <= 1e-15:
        norm_1 = 1e-15
    if norm_inf <= 1e-15:
        norm_inf = 1e-15

    return norm_2_squared, norm_1, norm_inf

def evaluate_c2_score(f_values: List[float]) -> float:
    """Compute C2 score for given function values"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

def create_adaptive_peak_function(n_steps: int = 1000) -> List[float]:
    """
    Create a function with strategically placed Gaussian peaks based on mathematical insights
    """
    # Initialize empty function
    f_values = np.zeros(n_steps)
    
    # Create structured peak locations using logarithmic spacing
    # This helps distribute peaks across the domain without clustering
    n_peaks = max(5, min(25, n_steps // 40))
    
    # Generate log-spaced peak positions across domain [-0.25, 0.25]
    positions = []
    for i in range(n_peaks):
        # Use log-uniform distribution to avoid clustering
        # Map to domain [-0.25, 0.25] with proper spacing
        if i == 0:
            pos = -0.25 + random.uniform(0.01, 0.05)  # Near left edge
        elif i == n_peaks - 1:
            pos = 0.25 - random.uniform(0.01, 0.05)   # Near right edge
        else:
            # Logarithmic distribution in the middle
            log_min = np.log(0.02)
            log_max = np.log(0.2)
            log_pos = np.random.uniform(log_min, log_max)
            pos = np.exp(log_pos) * random.choice([-1, 1])  # Alternate sides
            pos = np.clip(pos, -0.24, 0.24)  # Keep within bounds
            
        positions.append(pos)
    
    # Sort positions
    positions.sort()
    
    # Ensure minimum separation between peaks to prevent narrow autoconvolution interference
    min_separation = max(20, n_steps // 50)
    adjusted_positions = []
    for i, pos in enumerate(positions):
        if i == 0:
            adjusted_positions.append(pos)
        else:
            # Ensure minimum gap from previous peak
            prev_pos = adjusted_positions[-1]
            if abs(pos - prev_pos) < min_separation * 0.1:
                # Adjust to maintain gap
                adjusted_positions.append(prev_pos + min_separation * 0.1 * np.sign(pos - prev_pos))
            else:
                adjusted_positions.append(pos)
    
    # Generate peak parameters
    for i, center_pos in enumerate(adjusted_positions):
        # Convert position to step index
        step_index = int((center_pos + 0.25) / (0.5 / n_steps))
        step_index = max(0, min(n_steps - 1, step_index))
        
        # Adaptive width and height based on position and peak importance
        # Peaks near center get narrower and higher amplitude for sharper autoconvolution
        # Outer peaks get broader and more moderate to avoid boundary artifacts
        if i == 0 or i == len(adjusted_positions) - 1:
            # Boundary peaks: broader and lower
            width = max(15, min(60, n_steps // 8))
            height = random.uniform(1.2, 2.0)
        else:
            # Inner peaks: narrower and higher  
            width = max(10, min(40, n_steps // 15))
            height = random.uniform(0.8, 1.5)
            
        # Scale height to avoid extremely dominant peaks that hurt C2 ratio
        height *= min(1.0, 100.0 / (width * (i + 1) + 20.0))
        
        # Create Gaussian-like peak centered at step_index
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - step_index) / width) ** 2)
        f_values += gaussian
    
    # Apply smoothing to reduce extreme variations
    if n_steps > 50:
        window_size = min(51, n_steps // 5)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_values = signal.savgol_filter(f_values, window_size, 3)
    
    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    if np.max(f_values) > 0:
        f_values = f_values / np.max(f_values) * 2.0
    
    # Apply constraint-aware normalization to prevent autoconvolution spikes
    _, _, norm_inf = compute_autoconvolution_norms(f_values.tolist())
    if norm_inf > 0:
        # Cap extreme values to keep autoconvolution manageable
        max_allowed = np.percentile(f_values, 90) if len(f_values) > 10 else 1.0
        if max_allowed > 0:
            f_values = np.minimum(f_values, max_allowed * 3.0)
    
    return f_values.tolist()

def optimize_function_locally(initial_function: List[float], 
                           max_iterations: int = 100) -> List[float]:
    """
    Apply local optimization to improve the function
    """
    current_function = initial_function.copy()
    current_score = evaluate_c2_score(current_function)
    
    best_function = current_function.copy()
    best_score = current_score
    
    # Simple hill-climbing approach
    for iteration in range(max_iterations):
        # Create neighbor by perturbing a subset of values
        neighbor = current_function.copy()
        
        # Sample some indices to modify
        n_modify = max(1, min(len(neighbor) // 10, 20))
        indices_to_modify = random.sample(range(len(neighbor)), n_modify)
        
        for idx in indices_to_modify:
            # Apply small multiplicative perturbation
            factor = random.uniform(0.95, 1.05)
            neighbor[idx] = max(0, neighbor[idx] * factor)
        
        # Evaluate neighbor
        neighbor_score = evaluate_c2_score(neighbor)
        
        # Accept if better
        if neighbor_score > current_score:
            current_function = neighbor
            current_score = neighbor_score
            if neighbor_score > best_score:
                best_function = neighbor
                best_score = neighbor_score
    
    return best_function

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Implements a hybrid approach combining adaptive peak construction
    with local optimization.
    """
    start_time = time.time()
    max_execution_time = 85  # seconds
    
    # Phase 1: Multiple adaptive constructions with different parameters
    best_result = []
    best_c2 = 0.0
    
    # Try several configurations to find a promising starting point
    for attempt in range(10):
        if time.time() - start_time > max_execution_time * 0.8:
            break
            
        try:
            # Create function with adaptive peak construction
            n_steps = random.randint(800, 2000)
            candidate_function = create_adaptive_peak_function(n_steps)
            
            # Evaluate candidate
            c2 = evaluate_c2_score(candidate_function)
            
            if c2 > best_c2:
                best_c2 = c2
                best_result = candidate_function.copy()
                
        except Exception as e:
            continue
    
    # Phase 2: Local optimization on best candidate
    if best_result and time.time() - start_time < max_execution_time * 0.9:
        try:
            optimized_result = optimize_function_locally(best_result, 100)
            optimized_c2 = evaluate_c2_score(optimized_result)
            
            if optimized_c2 > best_c2:
                best_result = optimized_result
                best_c2 = optimized_c2
                
        except Exception as e:
            pass
    
    # Phase 3: Fallback to robust construction if no good result found
    if len(best_result) == 0 or best_c2 < 0.8:
        try:
            # Use a more robust peak construction approach
            n_steps = 1000
            best_result = create_adaptive_peak_function(n_steps)
            
            # Final evaluation and optimization
            if best_result:
                final_c2 = evaluate_c2_score(best_result)
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    
        except Exception as e:
            # Last resort: simple symmetric function
            n_steps = 1000
            x = np.linspace(-1, 1, n_steps)
            base_shape = np.exp(-x**2 / 2)
            base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
            best_result = base_shape.tolist()
    
    # Final validation check
    if best_result:
        try:
            final_c2 = evaluate_c2_score(best_result)
            if final_c2 <= 0:
                # Fallback if evaluation fails
                n_steps = 1000
                x = np.linspace(-1, 1, n_steps)
                base_shape = np.exp(-x**2 / 2)
                base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
                best_result = base_shape.tolist()
        except:
            # Final fallback
            n_steps = 1000
            x = np.linspace(-1, 1, n_steps)
            base_shape = np.exp(-x**2 / 2)
            base_shape = 0.6 * (base_shape / np.max(base_shape)) + 0.2
            best_result = base_shape.tolist()
    
    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > max_execution_time:
        return best_result[:1000]  # Truncate if needed
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")