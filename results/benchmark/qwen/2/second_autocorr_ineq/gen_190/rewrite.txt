# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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

    # Compute autoconvolution using FFT for better performance
    # Pad arrays to avoid circular convolution effects
    padded_len = 2 * len(f) - 1
    f_padded = np.pad(f, (0, padded_len - len(f)), 'constant', constant_values=0)
    
    # Compute autoconvolution using FFT
    f_fft = np.fft.fft(f_padded)
    g_fft = f_fft * np.conj(f_fft)
    g = np.fft.ifft(g_fft).real[:padded_len]
    
    # Keep only the middle part (proper autoconvolution)
    g = g[len(f)-1:2*len(f)-1]
    
    # Compute norms with numerical stability checks
    g_abs = np.abs(g)

    # ||g||₂² (L2 norm squared) - use trapezoidal rule properly
    g_sq = g_abs ** 2
    if len(g_sq) < 2:
        norm_2_sq = 0.0 if len(g_sq) == 0 else g_sq[0]
    else:
        # Trapezoidal integration: sum((y[i] + y[i+1])/2 * delta_x)
        dx = 0.5 / (len(g_abs) - 1) if len(g_abs) > 1 else 0.5
        norm_2_sq = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0) * dx

    # ||g||₁ (L1 norm) - sum of absolute values
    norm_1 = np.sum(g_abs) * (0.5 / (len(g_abs) - 1)) if len(g_abs) > 1 else np.sum(g_abs)

    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs)

    # Numerical stability checks
    norm_2_sq = max(0.0, norm_2_sq)
    norm_1 = max(1e-12, norm_1)  # Avoid division by zero
    norm_inf = max(1e-12, norm_inf)  # Avoid division by zero

    return norm_2_sq, norm_1, norm_inf

def generate_refined_peaks(n_points, n_peaks):
    """
    Generate peak configuration with improved spacing and characteristics
    """
    # Create domain
    x = np.linspace(-0.25, 0.25, n_points)
    f_values = np.zeros_like(x)

    # Use logarithmic distribution for peak positions to avoid clustering
    peak_positions = []
    peak_amplitudes = []
    peak_widths = []

    # Create log-spaced positions with better distribution
    if n_peaks > 1:
        log_min = np.log(0.015)
        log_max = np.log(0.18)
        log_spaced_positions = np.logspace(log_min, log_max, max(1, n_peaks // 2))
        
        # Place peaks on both sides of center with logarithmic spacing
        for i in range(n_peaks):
            # Alternate sides to distribute evenly
            side = 1 if i % 2 == 0 else -1
            if i < len(log_spaced_positions):
                pos = side * log_spaced_positions[i // 2] + np.random.uniform(-0.01, 0.01)
            else:
                # Fill remaining peaks with uniform distribution
                pos = np.random.uniform(-0.23, 0.23)

            # Clip to valid range
            pos = np.clip(pos, -0.23, 0.23)

            # Avoid too close proximity to existing peaks
            valid_position = True
            for existing_pos in peak_positions:
                if abs(pos - existing_pos) < 0.015:  # Minimum gap of 0.015
                    valid_position = False
                    break

            if valid_position:
                peak_positions.append(pos)
    else:
        peak_positions = [0.0]

    # Generate amplitudes and widths with better distribution
    for pos in peak_positions:
        # Use distribution that decreases with distance from center
        center_distance = abs(pos)
        # Base amplitude with decay factor based on distance from center
        base_amp = np.random.exponential(0.5) * np.exp(-center_distance * 5.0)
        # Add some variation
        amp = base_amp * np.random.uniform(0.5, 1.5)
        
        # Cap amplitude to prevent extreme values
        amp = min(1.5, amp)
        
        peak_amplitudes.append(amp)

    # Create Gaussian peaks with optimized widths
    for i, (pos, amp) in enumerate(zip(peak_positions, peak_amplitudes)):
        # Vary widths to create more complex profile
        # Wider at center, narrower at edges
        center_distance = abs(pos)
        base_sigma = 0.02 + 0.03 * np.exp(-center_distance * 3.0)  
        sigma = np.clip(base_sigma * np.random.uniform(0.7, 1.3), 0.005, 0.08)
        
        gaussian_peak = amp * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
        f_values += gaussian_peak
        
    return f_values.tolist(), peak_positions

def construct_function_candidate(n_points, n_peaks):
    """
    Construct a single function candidate with adaptive peak characteristics
    """
    try:
        # Generate peaks
        f_values, peak_positions = generate_refined_peaks(n_points, n_peaks)
        
        # Apply adaptive smoothing with Gaussian filter
        if n_points > 100:
            # Determine smoothing window size
            window_size = max(3, min(21, int(n_points / 50)))
            if window_size % 2 == 0:
                window_size += 1
            
            # Convert to numpy for filtering
            f_array = np.array(f_values)
            # Apply Gaussian smoothing with nearest boundary condition
            f_array = gaussian_filter1d(f_array, sigma=window_size/3.0, mode='nearest')
            f_values = f_array.tolist()
        
        # Ensure non-negativity
        f_values = [max(0.0, val) for val in f_values]
        
        # Normalize to prevent extreme values
        max_val = max(f_values) if f_values else 1.0
        if max_val > 0:
            f_values = [val / (max_val * 1.5) for val in f_values]
            
        return f_values
        
    except Exception as e:
        warnings.warn(f"Failed to construct candidate: {str(e)}")
        # Fallback to simple construction
        return [0.0] * n_points

def evaluate_candidate(f_values):
    """
    Evaluate a single candidate function for its C2 value
    """
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
        
    except Exception as e:
        warnings.warn(f"Evaluation error: {str(e)}")
        return 0.0

def evaluate_candidates_parallel(candidates, max_workers=8):
    """
    Evaluate multiple candidates in parallel
    """
    results = []
    
    def evaluate_single_candidate(candidate):
        return evaluate_candidate(candidate)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_candidate, cand) for cand in candidates]
            results = [future.result() for future in as_completed(futures)]
    except Exception as e:
        warnings.warn(f"Parallel evaluation failed: {str(e)}")
        # Fall back to sequential evaluation
        results = [evaluate_single_candidate(cand) for cand in candidates]
    
    return results

def optimize_function_candidate(best_f, max_iterations=20):
    """
    Perform targeted local optimization on the best candidate
    """
    try:
        # Select key parameters for optimization (sample every 3rd point for efficiency)
        sample_indices = list(range(0, len(best_f), 3))
        if len(sample_indices) < 10:
            sample_indices = list(range(len(best_f)))
            
        # Objective function that operates on sampled parameters
        def objective(params):
            # Create full function with sampled parameters
            temp_f = best_f.copy()
            for i, idx in enumerate(sample_indices):
                if i < len(params):
                    temp_f[idx] = max(0.0, params[i])
            
            # Compute norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(temp_f)
            
            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return float('inf')
                
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return -c2 if not np.isnan(c2) and not np.isinf(c2) else float('inf')

        # Run differential evolution on selected parameters
        bounds = [(0.0, 2.0)] * len(sample_indices)
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_iterations,
            popsize=5,
            seed=42,
            strategy='best1bin',
            tol=1e-6
        )

        # Apply refined parameters back to original function
        final_f = best_f.copy()
        for i, idx in enumerate(sample_indices):
            if i < len(result.x):
                final_f[idx] = max(0.0, result.x[i])
                
        # Recompute C2 to verify improvement
        _, norm_1, norm_inf = compute_autoconvolution_norms(final_f)
        if norm_1 > 1e-12 and norm_inf > 1e-12:
            norm_2_sq, _, _ = compute_autoconvolution_norms(final_f)
            final_c2 = norm_2_sq / (norm_1 * norm_inf)
            return final_f, final_c2
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        pass
    
    return best_f, evaluate_candidate(best_f)

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Implements optimized evolutionary approach with FFT-based autoconvolution.
    """
    start_time = time.time()
    best_c2 = 0.0
    best_f = None
    
    # Multi-stage approach for better exploration
    strategies = [
        {'points': 2000, 'peaks': 15},
        {'points': 3000, 'peaks': 20},
        {'points': 1500, 'peaks': 12},
        {'points': 2500, 'peaks': 18}
    ]
    
    # Stage 1: Generate diverse candidates with different resolutions
    candidates = []
    candidate_configs = []
    
    # Generate candidates using different parameters
    for config in strategies:
        n_points = config['points']
        n_peaks = config['peaks']
        # Generate multiple candidates per configuration
        for _ in range(5):
            if time.time() - start_time > 85:  # Leave time for final processing
                break
            candidate = construct_function_candidate(n_points, n_peaks)
            candidates.append(candidate)
            candidate_configs.append(config)
    
    # Evaluate candidates in parallel
    if candidates:
        c2_scores = evaluate_candidates_parallel(candidates)
        
        # Find best candidate
        for i, (c2_score, candidate) in enumerate(zip(c2_scores, candidates)):
            if c2_score > best_c2:
                best_c2 = c2_score
                best_f = candidate.copy()
    
    # Stage 2: Local refinement of the best candidate
    if best_f is not None and time.time() - start_time < 80:
        try:
            # Perform targeted optimization
            refined_f, refined_c2 = optimize_function_candidate(best_f, max_iterations=15)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_f = refined_f
        except Exception as e:
            warnings.warn(f"Refinement failed: {str(e)}")
    
    # Stage 3: Final high-resolution construction if time permits
    if best_f is not None and time.time() - start_time < 80:
        try:
            # Create high-res version for fine-tuning
            high_res_f = construct_function_candidate(5000, 30)
            high_res_c2 = evaluate_candidate(high_res_f)
            
            if high_res_c2 > best_c2:
                best_c2 = high_res_c2
                best_f = high_res_f
                
                # Further refine if possible
                refined_f, refined_c2 = optimize_function_candidate(best_f, max_iterations=10)
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_f = refined_f
        except Exception as e:
            warnings.warn(f"High-res construction failed: {str(e)}")
    
    # Fallback if nothing worked well
    if best_f is None or len(best_f) == 0:
        # Create robust fallback
        n_points = 1000
        x = np.linspace(-0.25, 0.25, n_points)
        # Create a well-behaved Gaussian-like function
        f_values = np.exp(-0.5 * (x / 0.05)**2)
        f_values = f_values / (np.max(f_values) * 1.5)  # Normalize
        best_f = f_values.tolist()
    
    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")