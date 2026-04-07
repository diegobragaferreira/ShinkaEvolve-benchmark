# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
import random
import time
from typing import List, Tuple
import math

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution efficiently using numba"""
    n = len(f_vals)
    # Create output array for autoconvolution
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually with numba optimization
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@numba.jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with numba"""
    n = len(g_vals)
    
    # L2 norm squared (using trapezoidal-like scheme)
    l2_sq = 0.0
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])
    
    # L-infinity norm
    linf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val
    
    return l2_sq, l1, linf

def evaluate_function(f_vals: List[float]) -> Tuple[float, float, float, float]:
    """
    Evaluate function and return C2 score with detailed metrics.
    Returns: (c2, norm_2_sq, norm_1, norm_inf)
    """
    try:
        # Convert to numpy array and ensure non-negative
        f_array = np.array(f_vals, dtype=np.float64)
        f_array = np.maximum(f_array, 0.0)
        
        # Skip if all zeros
        if np.sum(f_array) == 0:
            return (0.0, 0.0, 0.0, 0.0)
            
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_array)
        
        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return (0.0, l2_sq, l1, linf)
        
        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return (c2, l2_sq, l1, linf)
    except Exception as e:
        return (0.0, 0.0, 0.0, 0.0)

def adaptive_peak_construction(n_steps: int = None, peak_count: int = None) -> List[float]:
    """Construct function with adaptive Gaussian peak placement and parameters"""
    if n_steps is None:
        n_steps = random.randint(400, 1200)
    
    if peak_count is None:
        peak_count = max(3, min(20, n_steps // 50))
        
    # Create empty function
    f_vals = np.zeros(n_steps)
    
    # Generate peak positions with careful spacing
    # Use beta distribution to create clustered but well-spaced peaks
    positions_raw = np.random.beta(2, 2, peak_count)
    # Map to actual positions in [1, n_steps-2] to avoid boundary issues  
    peak_positions = (positions_raw * (n_steps - 2) + 1).astype(int)
    
    # Ensure uniqueness and sort
    peak_positions = np.unique(peak_positions)
    # If we don't have enough peaks, add more
    while len(peak_positions) < peak_count:
        new_pos = random.randint(1, n_steps - 2)
        if new_pos not in peak_positions:
            peak_positions = np.append(peak_positions, new_pos)
    peak_positions = np.sort(peak_positions)[:peak_count]
    
    # Generate parameters for each peak
    for i, center in enumerate(peak_positions):
        # Width based on position and peak importance
        if i == 0 or i == len(peak_positions) - 1:
            # Edge peaks wider to avoid boundary artifacts
            width = max(10, min(50, n_steps // 8))
        else:
            # Inner peaks narrower for sharper contributions
            width = max(8, min(30, n_steps // 15))
        
        # Height inversely related to width and peak importance
        if i == 0 or i == len(peak_positions) - 1:
            # Outer peaks higher to create better autoconvolution
            height = random.uniform(1.2, 2.5)
        else:
            # Inner peaks moderate height
            height = random.uniform(0.8, 1.8)
            
        # Apply adaptive scaling to prevent over-dominant peaks
        height *= min(1.0, 200.0 / (width * (i + 1) + 50.0))
        
        # Create Gaussian-like peak
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian
    
    # Apply smoothing to reduce extreme variations
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_vals = signal.savgol_filter(f_vals, window_size, 3)
    
    # Ensure non-negativity and normalize reasonably
    f_vals = np.maximum(f_vals, 0)
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 2.0
    
    return f_vals.tolist()

def simulated_annealing_optimization(initial_function: List[float], 
                                   max_iterations: int = 1000,
                                   initial_temp: float = 1.0,
                                   cooling_rate: float = 0.995) -> List[float]:
    """
    Simulated annealing for optimizing peak parameters
    """
    current_solution = initial_function.copy()
    current_score, _, _, _ = evaluate_function(current_solution)
    
    best_solution = current_solution.copy()
    best_score = current_score
    
    temperature = initial_temp
    
    # Track recent improvements
    recent_improvements = []
    
    for iteration in range(max_iterations):
        # Create neighbor by perturbing some peak parameters
        neighbor = current_solution.copy()
        
        # Select which elements to modify
        n_modify = min(10, len(neighbor) // 5)
        indices_to_modify = random.sample(range(len(neighbor)), n_modify)
        
        for idx in indices_to_modify:
            # Small random perturbation
            if random.random() < 0.7:
                # Multiplicative perturbation
                factor = random.uniform(0.9, 1.1)
                neighbor[idx] = max(0, neighbor[idx] * factor)
            else:
                # Additive perturbation
                delta = random.gauss(0, 0.05 * max(1, neighbor[idx]))
                neighbor[idx] = max(0, neighbor[idx] + delta)
        
        # Evaluate neighbor
        neighbor_score, _, _, _ = evaluate_function(neighbor)
        
        # Accept or reject
        if neighbor_score > current_score:
            current_solution = neighbor
            current_score = neighbor_score
            recent_improvements.append(neighbor_score)
        else:
            # Accept with probability based on temperature
            if temperature > 1e-10:
                prob_accept = math.exp((neighbor_score - current_score) / temperature)
                if random.random() < prob_accept:
                    current_solution = neighbor
                    current_score = neighbor_score
                    recent_improvements.append(neighbor_score)
        
        # Update best if improved
        if current_score > best_score:
            best_solution = current_solution.copy()
            best_score = current_score
            
        # Cool down
        temperature *= cooling_rate
        
        # Early stopping if no improvement recently
        if len(recent_improvements) > 20:
            recent_avg = sum(recent_improvements[-20:]) / 20
            if recent_avg < best_score * 0.9999:
                recent_improvements.clear()
    
    return best_solution

def multi_scale_optimization(starting_function: List[float],
                           max_time_seconds: float = 80.0) -> List[float]:
    """
    Performs optimization at multiple scales to find better local optima
    """
    start_time = time.time()
    
    best_result = starting_function.copy()
    best_score, _, _, _ = evaluate_function(best_result)
    
    # Multi-scale approach - start with coarse and refine
    scales = [1, 0.7, 0.5, 0.3]
    
    for scale in scales:
        if time.time() - start_time > max_time_seconds * 0.9:
            break
            
        # Create scaled version of current best
        if scale < 1.0:
            # Apply small random modifications to current best
            modified = best_result.copy()
            n_modify = max(1, int(len(modified) * scale * 0.1))
            indices_to_modify = random.sample(range(len(modified)), n_modify)
            
            for idx in indices_to_modify:
                if random.random() < 0.7:
                    factor = random.uniform(0.95, 1.05)
                    modified[idx] = max(0, modified[idx] * factor)
                else:
                    delta = random.gauss(0, 0.02 * max(1, modified[idx]))
                    modified[idx] = max(0, modified[idx] + delta)
        else:
            modified = best_result.copy()
            
        # Apply simulated annealing
        refined = simulated_annealing_optimization(modified, 
                                                 max_iterations=int(200 * scale),
                                                 initial_temp=0.5 * scale)
        
        # Evaluate and update best
        refined_score, _, _, _ = evaluate_function(refined)
        if refined_score > best_score:
            best_result = refined
            best_score = refined_score
    
    return best_result

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Implements a hybrid approach combining multi-scale optimization
    with adaptive peak construction.
    """
    start_time = time.time()
    
    # Phase 1: Multi-initialization with different configurations  
    best_result = []
    best_c2 = 0
    
    # Try several different initializations
    for attempt in range(5):
        if time.time() - start_time > 75:  # Leave time for final processing
            break
            
        # Try different combinations
        n_steps = random.randint(500, 1500)
        peak_count = random.randint(5, 25)
        
        # Create initial function
        initial_func = adaptive_peak_construction(n_steps, peak_count)
        
        # Improve with multi-scale optimization
        improved_func = multi_scale_optimization(initial_func, 
                                               max_time_seconds=75 - (time.time() - start_time))
        
        # Evaluate result
        c2, _, _, _ = evaluate_function(improved_func)
        if c2 > best_c2:
            best_c2 = c2
            best_result = improved_func
    
    # Phase 2: Final fine-tuning if needed
    if best_result and time.time() - start_time < 78:
        # Do one final round of focused optimization
        final_result = simulated_annealing_optimization(
            best_result, 
            max_iterations=300,
            initial_temp=0.2
        )
        
        # Check if final optimization improved
        final_c2, _, _, _ = evaluate_function(final_result)
        if final_c2 > best_c2:
            best_result = final_result
            best_c2 = final_c2
    
    # Phase 3: Fallback to robust construction if nothing worked well
    if len(best_result) == 0:
        # Use a more conservative approach
        n_steps = 800
        best_result = adaptive_peak_construction(n_steps, 15)
    
    # Final validation check
    if best_result:
        try:
            final_c2, _, _, _ = evaluate_function(best_result)
            if final_c2 <= 0:
                # If final evaluation failed, fallback to basic construction
                n_steps = 800
                best_result = adaptive_peak_construction(n_steps, 15)
        except:
            n_steps = 800
            best_result = adaptive_peak_construction(n_steps, 15)
    
    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")