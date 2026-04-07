# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
import random
import time

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

def evaluate_individual(individual):
    """Evaluate fitness of an individual (step function)"""
    try:
        # Convert to numpy array and ensure non-negative
        f_vals = np.array(individual, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)
        
        # Skip if all zeros
        if np.sum(f_vals) == 0:
            return (0.0,)
            
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return (0.0,)
        
        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return (c2,)
    except:
        return (0.0,)

def adaptive_gaussian_construction(n_steps=None):
    """
    Improved Gaussian peak construction with adaptive parameters and logarithmic spacing
    """
    if n_steps is None:
        n_steps = random.randint(800, 2000)
    
    # Create base function with logarithmic peak spacing
    f_vals = np.zeros(n_steps)
    
    # Use logarithmic distribution for peak positions to avoid regular patterns
    n_peaks = max(3, min(25, n_steps // 30))
    
    # Generate logarithmically spaced peak positions  
    log_positions = np.logspace(np.log10(0.1), np.log10(0.9), n_peaks, endpoint=True)
    peak_positions = (log_positions * n_steps).astype(int)
    # Ensure uniqueness and bounds
    peak_positions = np.unique(np.clip(peak_positions, 1, n_steps-2))
    
    # If we don't have enough peaks, add some at random positions
    if len(peak_positions) < n_peaks:
        additional = n_peaks - len(peak_positions)
        for _ in range(additional):
            pos = random.randint(1, n_steps-2)
            if pos not in peak_positions:
                peak_positions = np.append(peak_positions, pos)
        peak_positions = np.sort(peak_positions)
    
    # Build peaks with adaptive parameters
    for i, center in enumerate(peak_positions):
        # Adaptive width based on peak position and index
        # Peaks near edges get wider to avoid boundary artifacts
        if i == 0 or i == len(peak_positions) - 1:
            width_factor = 1.5  # Wider peaks at boundaries
        else:
            width_factor = 1.0
            
        # Width varies logarithmically with position to prevent clustering
        width_base = max(10, min(80, n_steps // (max(1, i + 2))))
        width = width_base * width_factor
        
        # Height inversely related to width and position importance
        # Outer peaks get higher amplitude to enhance convolution
        if i == 0 or i == len(peak_positions) - 1:
            height = random.uniform(1.2, 2.0)
        else:
            height = random.uniform(0.8, 1.5)
            
        # Apply adaptive scaling to prevent extreme autoconvolution peaks
        # This helps with numerical stability and C2 optimization
        height *= min(1.0, 1000.0 / (width * (i + 1) + 100.0))
        
        # Create Gaussian-like peak with adaptive shape
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian
    
    # Apply smoothing with adaptive window size
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_vals = signal.savgol_filter(f_vals, window_size, 3)
    
    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)
    
    # Apply final constraint-aware normalization
    if np.max(f_vals) > 0:
        # Adaptive thresholding to prevent autoconvolution spikes
        threshold = np.percentile(f_vals, 92) if len(f_vals) > 10 else np.max(f_vals)
        if threshold > 0:
            f_vals = np.minimum(f_vals, threshold * 2.5)
        f_vals = f_vals / np.max(f_vals) * 2.0 if np.max(f_vals) > 0 else f_vals
    
    return f_vals.tolist()

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Phase 1: Multiple adaptive harmonic constructions
    best_result = []
    best_c2 = 0
    
    # Try several adaptive harmonic constructions
    for attempt in range(10):
        # Check time limit
        if time.time() - start_time > 80:
            break
            
        try:
            # Create adaptive harmonic function
            adaptive_func = adaptive_gaussian_construction()
            f_vals = np.array(adaptive_func, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                
                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = adaptive_func.copy()
        except:
            continue
    
    # Phase 2: Local refinement with small mutations
    if best_result and time.time() - start_time < 85:
        try:
            # Apply focused refinement
            refined_result = best_result.copy()
            max_attempts = 30
            
            for attempt in range(max_attempts):
                # Check time limit
                if time.time() - start_time > 85:
                    break
                    
                test_result = refined_result.copy()
                # Select random indices to modify
                indices_to_change = random.sample(range(len(test_result)), 
                                                min(20, len(test_result) // 5))
                
                for idx in indices_to_change:
                    # Apply small adaptive changes
                    if random.random() < 0.7:
                        factor = random.uniform(0.95, 1.05)
                        test_result[idx] = max(0, test_result[idx] * factor)
                    else:
                        # Add small noise
                        noise = random.gauss(0, 0.03 * max(1, test_result[idx]))
                        test_result[idx] = max(0, test_result[idx] + noise)
                
                # Evaluate improvement
                current_score = evaluate_individual(refined_result)[0]
                test_score = evaluate_individual(test_result)[0]
                
                if test_score > current_score:
                    refined_result = test_result
                    if test_score > best_c2:
                        best_c2 = test_score
                        
            if evaluate_individual(refined_result)[0] > evaluate_individual(best_result)[0]:
                best_result = refined_result
        except:
            pass
    
    # Phase 3: Fallback to robust construction if needed
    if len(best_result) == 0 or best_c2 < 0.8:
        # Use more robust harmonic construction
        n_steps = random.randint(800, 2000)
        best_result = adaptive_gaussian_construction(n_steps)
    
    # Final validation check
    if best_result:
        try:
            f_vals = np.array(best_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                if l1 > 1e-15 and linf > 1e-15:
                    final_c2 = l2_sq / (l1 * linf)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
        except:
            pass
    
    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")