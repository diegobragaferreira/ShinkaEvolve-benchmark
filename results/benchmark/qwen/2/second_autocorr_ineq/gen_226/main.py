# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from numba import jit
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
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

def fast_fft_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast autoconvolution computation using FFT for better performance on large arrays.
    """
    f = np.array(f_values, dtype=np.float64)
    n_steps = len(f)

    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n_steps

    # Use FFT for efficient convolution (f * f)
    # Pad to next power of 2 for efficiency
    n_pad = 2**(int(np.log2(n_steps)) + 1) if n_steps > 0 else 1
    f_padded = np.pad(f, (0, n_pad - n_steps), 'constant', constant_values=0)
    
    # FFT-based convolution (f * f)
    f_fft = np.fft.fft(f_padded)
    g_fft = f_fft * np.conj(f_fft)
    g = np.real(np.fft.ifft(g_fft))
    
    # Take the first n_steps elements (autoconvolution result)
    g = g[:n_steps]
    
    # Compute norms
    # L2 norm squared
    g_sq = g**2
    norm_2_squared = np.sum((g_sq[:-1] + g_sq[1:]) / 2.0) * dx  # Trapezoidal rule
    
    # L1 norm
    norm_1 = np.sum(np.abs(g))
    
    # Infinity norm
    norm_inf = np.max(np.abs(g))

    # Handle numerical edge cases
    if norm_1 <= 1e-15:
        norm_1 = 1e-15
    if norm_inf <= 1e-15:
        norm_inf = 1e-15

    return norm_2_squared, norm_1, norm_inf

def generate_multi_scale_gaussian_peaks(n_points: int, n_peaks: int = None) -> list[float]:
    """
    Generate step function using multi-scale Gaussian peaks with improved spacing.
    """
    if n_peaks is None:
        n_peaks = max(30, min(300, n_points // 15))
    
    # Create domain
    x_domain = np.linspace(-0.25, 0.25, n_points)
    
    # Generate peaks with different characteristics
    f_values = np.zeros_like(x_domain)
    
    # Create multi-scale peak distribution
    scales = np.logspace(np.log10(0.01), np.log10(0.15), 6)
    
    for scale_idx, scale in enumerate(scales):
        # Determine number of peaks per scale
        n_peaks_per_scale = max(2, int(n_peaks * scale / 0.15 * 0.5))
        
        # Generate positions with logarithmic spacing
        positions = np.logspace(np.log10(scale * 0.1), np.log10(scale), n_peaks_per_scale)
        positions = np.concatenate([-positions[::-1], positions])
        
        # Filter to domain
        positions = positions[(positions >= -0.25) & (positions <= 0.25)]
        
        # Apply adaptive amplitude scaling based on scale
        amplitude_factor = 1.0 / (scale * 10.0 + 1.0)
        
        for pos in positions:
            # Different peak widths based on scale
            width = max(0.005, scale * 0.5)
            amplitude = np.exp(-10 * pos**2) * amplitude_factor
            
            # Add Gaussian peak
            gaussian_peak = amplitude * np.exp(-0.5 * ((x_domain - pos) / width)**2)
            f_values += gaussian_peak
    
    # Add structured modulation to avoid degeneracy
    modulation = 0.1 * np.sin(20 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)
    f_values += modulation
    
    # Add adaptive noise
    noise = np.random.normal(0, 0.05, n_points)
    f_values += noise
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)
    
    # Normalize
    if np.sum(f_values) > 0:
        f_values = f_values / np.sum(f_values) * 10
    
    return f_values.tolist()

def generate_structured_peaks(n_points: int) -> list[float]:
    """
    Generate structured step function with balanced peak distribution.
    """
    x_domain = np.linspace(-0.25, 0.25, n_points)
    
    # Create more evenly distributed peaks
    n_peaks = max(50, min(200, n_points // 10))
    
    # More strategic peak placement using golden ratio
    phi = (1 + np.sqrt(5)) / 2
    peak_positions = []
    
    for i in range(n_peaks):
        ratio = (i * phi) % 1.0
        # Use sin transformation for better center concentration
        pos = np.sin(ratio * np.pi) * 0.25
        peak_positions.append(pos)
    
    # Create Gaussian peaks
    f_values = np.zeros(n_points)
    for i, pos in enumerate(peak_positions):
        # Vary widths and amplitudes for diversity
        width = np.random.uniform(0.01, 0.05)
        amplitude = np.random.uniform(0.5, 2.0)
        
        gaussian_peak = amplitude * np.exp(-0.5 * ((x_domain - pos) / width)**2)
        f_values += gaussian_peak
    
    # Add modulation
    mod = 0.05 * np.sin(15 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)
    f_values += mod
    
    # Noise for robustness
    noise = np.random.normal(0, 0.03, n_points)
    f_values += noise
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0)
    
    # Normalize
    if np.sum(f_values) > 0:
        f_values = f_values / np.sum(f_values) * 10
    
    return f_values.tolist()

def evaluate_candidate_function(f_values: list[float]) -> tuple[float, float, float, float]:
    """
    Enhanced evaluation that includes multiple norm computation methods for robustness.
    """
    try:
        # Use the more efficient FFT-based computation
        norm_2_sq, norm_1, norm_inf = fast_fft_autoconvolution_norms(f_values)
        
        # Calculate C2
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0, 0.0, 0.0, 0.0
        
        c2 = norm_2_sq / (norm_1 * norm_inf)
        
        return c2, norm_2_sq, norm_1, norm_inf
        
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def construct_multiple_candidates(n_points_list: list[int], n_candidates: int = 10) -> list[tuple[float, list[float]]]:
    """
    Generate multiple candidate functions in parallel for better exploration.
    """
    candidates = []
    
    def process_candidate(args):
        n_points, method_id = args
        try:
            if method_id == 0:
                f_values = generate_multi_scale_gaussian_peaks(n_points)
            elif method_id == 1:
                f_values = generate_structured_peaks(n_points)
            else:
                # Random construction
                f_values = [np.random.uniform(0, 1) for _ in range(n_points)]
            
            c2, _, _, _ = evaluate_candidate_function(f_values)
            return (c2, f_values)
        except Exception:
            return (0.0, [])
    
    # Prepare arguments for parallel processing
    args_list = [(n_points, i % 2) for n_points in n_points_list for i in range(int(n_candidates/2))]
    
    # Use process pool for parallel execution
    with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        futures = [executor.submit(process_candidate, arg) for arg in args_list]
        results = [future.result() for future in as_completed(futures)]
    
    # Filter valid candidates
    valid_candidates = [(c2, f_values) for c2, f_values in results if f_values and len(f_values) > 0 and c2 > 0]
    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    
    return valid_candidates[:n_candidates]

def adaptive_refinement(best_f: list[float], max_iter: int = 20) -> list[float]:
    """
    Perform adaptive local refinement on the best function.
    """
    try:
        # Simple refinement by slightly adjusting peak positions and amplitudes
        f_array = np.array(best_f)
        n_points = len(f_array)
        
        # Create a slightly modified version
        # Add small random perturbations
        perturbation = np.random.normal(0, 0.02, n_points)
        refined_f = f_array + perturbation
        
        # Ensure non-negativity
        refined_f = np.maximum(refined_f, 0)
        
        # Normalize
        if np.sum(refined_f) > 0:
            refined_f = refined_f / np.sum(refined_f) * 10
        
        return refined_f.tolist()
        
    except Exception:
        return best_f

def construct_function() -> list[float]:
    """
    Construct step function with optimized Gaussian-based peak placement and parallel processing.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()
    
    # Set maximum attempts for balancing quality vs time constraints
    max_attempts = 30
    
    # Generate candidate points to try different resolutions
    n_points_list = [np.random.randint(1000, 5000) for _ in range(20)]
    
    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break
        
        # Generate multiple candidates in parallel
        candidates = construct_multiple_candidates(n_points_list[:5], n_candidates=10)
        
        # Evaluate candidates
        for c2, f_values in candidates:
            if c2 > best_c2:
                best_c2 = c2
                best_function = f_values
        
        # Early exit if we've found a very good solution
        if best_c2 > 0.95:
            break
    
    # Final refinement of best function
    if best_function is not None and time.time() - start_time < 80:
        try:
            refined_function = adaptive_refinement(best_function)
            # Re-evaluate refined version
            c2_refined, _, _, _ = evaluate_candidate_function(refined_function)
            
            if c2_refined > best_c2:
                best_c2 = c2_refined
                best_function = refined_function
        except Exception:
            pass
    
    # Return the best function found, or fallback
    if best_function is not None:
        return best_function
    else:
        # Fallback to a simpler construction
        n_steps = 1000
        f_values = [1.0] * n_steps
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")