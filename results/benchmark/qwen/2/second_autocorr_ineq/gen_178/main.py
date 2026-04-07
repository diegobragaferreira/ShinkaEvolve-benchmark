# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy import signal
from numba import jit
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def enforce_minimum_peak_spacing(peak_params: list[float], domain_width: float, min_distance_ratio: float = 0.05) -> list[float]:
    """
    Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution interference.
    This helps maintain flatter autoconvolution profiles which improve C2 values.
    """
    if len(peak_params) < 3:
        return peak_params

    # Create list of peaks [amplitude, center, width]
    peaks = []
    for i in range(0, len(peak_params), 3):
        peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

    # Sort by center position
    peaks.sort(key=lambda x: x[1])

    # Ensure minimum spacing using a more sophisticated approach
    min_distance = min_distance_ratio * domain_width
    adjusted_peaks = []

    # First pass: place peaks with minimum spacing
    for i, (amp, center, width) in enumerate(peaks):
        if i == 0:
            adjusted_peaks.append([amp, center, width])
        else:
            prev_center = adjusted_peaks[-1][1]
            # Calculate required spacing
            required_space = min_distance

            # If the current peak is too close, move it
            if abs(center - prev_center) < required_space:
                new_center = prev_center + required_space
                # Ensure we stay within domain bounds
                new_center = np.clip(new_center, -domain_width/2, domain_width/2)
                adjusted_peaks.append([amp, new_center, width])
            else:
                adjusted_peaks.append([amp, center, width])

    # Second pass: if there are still overlaps, apply iterative adjustment
    # This handles cases where consecutive peaks were moved too close together
    final_peaks = [adjusted_peaks[0]]
    for i in range(1, len(adjusted_peaks)):
        prev_center = final_peaks[-1][1]
        amp, center, width = adjusted_peaks[i]
        required_space = min_distance

        if abs(center - prev_center) < required_space:
            # Move this peak further away
            new_center = prev_center + required_space
            # Ensure we stay within domain bounds
            new_center = np.clip(new_center, -domain_width/2, domain_width/2)
            final_peaks.append([amp, new_center, width])
        else:
            final_peaks.append([amp, center, width])

    # Flatten back to parameter list
    result = []
    for amp, center, width in final_peaks:
        result.extend([amp, center, width])

    return result

def create_gaussian_peak_function(x_domain: np.ndarray, peak_params: list[float]) -> np.ndarray:
    """
    Create function from Gaussian peak parameters with proper spacing enforcement.
    """
    f = np.zeros_like(x_domain)
    for i in range(0, len(peak_params), 3):
        amp, center, width = peak_params[i], peak_params[i+1], peak_params[i+2]
        if width > 1e-6:  # Avoid division by zero
            f += amp * np.exp(-0.5 * ((x_domain - center) / width)**2)
    return f

def logarithmic_peak_distribution(n_peaks: int) -> np.ndarray:
    """
    Generate peak positions using logarithmic distribution to better cover frequency space.
    """
    # Create logarithmic spacing for peak positions
    positions = np.logspace(-2, 0, n_peaks, base=10)  # Logarithmic distribution
    # Map to [-0.25, 0.25] interval
    positions = (positions - positions.min()) / (positions.max() - positions.min()) * 0.5 - 0.25
    # Randomly shuffle to avoid systematic bias
    np.random.shuffle(positions)
    return positions

def adaptive_gaussian_construction(n_steps: int, n_peaks: int = None) -> list[float]:
    """
    Construct a step function with adaptive Gaussian-based peak placement using logarithmic spacing.
    """
    x_domain = np.linspace(-0.25, 0.25, n_steps)
    domain_width = 0.5
    
    if n_peaks is None:
        n_peaks = max(5, min(30, n_steps // 100))  # Adaptive number of peaks
    
    # Generate positions using logarithmic distribution for better spectral coverage
    peak_positions = logarithmic_peak_distribution(n_peaks)
    
    # Initialize peak parameters [amplitude, center, width]
    peak_params = []
    for i, pos in enumerate(peak_positions):
        # Amplitude based on position - more peaks near center for better autoconvolution
        amplitude = np.exp(-20 * pos**2) * np.random.uniform(20, 80)
        center = pos
        # Width controlled to prevent very narrow or wide peaks
        width = np.random.uniform(0.015, 0.06)  
        peak_params.extend([amplitude, center, width])

    # Apply minimum peak spacing enforcement
    peak_params = enforce_minimum_peak_spacing(peak_params, domain_width)
    
    # Create function from these peaks
    f = create_gaussian_peak_function(x_domain, peak_params)
    
    # Add sinusoidal modulation for additional diversity
    f += 0.1 * np.sin(20 * np.pi * x_domain) * np.exp(-x_domain**2/0.05)
    
    # Add noise for robustness
    f += np.random.normal(0, 0.01, n_steps)
    
    # Enforce non-negativity and normalize
    f = np.maximum(f, 0)
    if np.sum(f) > 0:
        f = f / np.sum(f) * 15
        
    return f.tolist()

def evaluate_function_and_refine(f_values: list[float], max_evaluations: int = 10) -> tuple[list[float], float]:
    """
    Evaluate a function and perform local refinement if beneficial.
    """
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return f_values, 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        
        # Only refine if we have reasonable initial performance
        if c2 > 0.7 and len(f_values) > 100:
            # Try local refinement with differential evolution
            def objective(params):
                params = np.maximum(params, 0)
                try:
                    norm_2_sq_r, norm_1_r, norm_inf_r = compute_autoconvolution_norms_fast(params.tolist())
                    if norm_1_r <= 1e-15 or norm_inf_r <= 1e-15:
                        return 0.0
                    c2_r = norm_2_sq_r / (norm_1_r * norm_inf_r)
                    return -c2_r  # Negative because we minimize
                except:
                    return 0.0
            
            # Refinement bounds
            bounds = [(0, 10) for _ in range(len(f_values))]
            
            try:
                # Run differential evolution for refinement
                result = differential_evolution(
                    objective,
                    bounds=bounds,
                    maxiter=max_evaluations,
                    popsize=min(10, len(f_values) // 10),
                    seed=42,
                    strategy='best1bin',
                    disp=False
                )
                
                refined_params = result.x
                refined_params = np.maximum(refined_params, 0)
                
                # Check if refinement helped
                norm_2_sq_ref, norm_1_ref, norm_inf_ref = compute_autoconvolution_norms_fast(refined_params.tolist())
                if norm_1_ref <= 1e-15 or norm_inf_ref <= 1e-15:
                    return f_values, c2
                    
                c2_ref = norm_2_sq_ref / (norm_1_ref * norm_inf_ref)
                if c2_ref > c2:
                    return refined_params.tolist(), c2_ref
            except:
                pass
                
        return f_values, c2
    except Exception as e:
        warnings.warn(f"Evaluation error: {str(e)}")
        return f_values, 0.0

def construct_function() -> list[float]:
    """
    Construct step function using enhanced Gaussian peak optimization approach.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Multi-attempt selection to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()
    
    # Set maximum attempts to balance quality vs. time constraints
    max_attempts = 50
    
    # Track best so far to potentially exit early
    best_so_far_c2 = 0.0
    
    for attempt in range(max_attempts):
        # Early termination if time is running out
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
            break
            
        # Try different number of steps to find optimal
        n_steps = np.random.randint(1000, 5000)
        
        # Generate function with adaptive parameters
        f_values = adaptive_gaussian_construction(n_steps)
        
        # Evaluate and potentially refine the function
        refined_f_values, evaluated_c2 = evaluate_function_and_refine(f_values, max_evaluations=5)
        
        # Keep the best function
        if evaluated_c2 > best_c2:
            best_c2 = evaluated_c2
            best_function = refined_f_values
            
        # Early exit if we're consistently getting good results
        if best_c2 > best_so_far_c2:
            best_so_far_c2 = best_c2
            if best_c2 > 0.95:
                # Early termination if we have a very good solution
                break
    
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