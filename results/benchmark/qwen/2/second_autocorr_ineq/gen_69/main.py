# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import time
from numba import jit, prange

@jit(nopython=True)
def compute_autoconvolution_numba(f):
    """Numba-accelerated autoconvolution computation"""
    n = len(f)
    g = np.zeros(2*n - 1)

    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]

    return g[n-1:]  # Return positive lags only

@jit(nopython=True)
def compute_norms_numba(g):
    """Numba-accelerated norm computations"""
    n = len(g)

    # Compute norms
    norm_1 = 0.0
    norm_2_sq = 0.0
    norm_inf = 0.0

    for i in range(n):
        abs_g = abs(g[i])
        norm_1 += abs_g
        norm_2_sq += abs_g * abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g

    return norm_1, norm_2_sq, norm_inf

@jit(nopython=True)
def compute_c2_numba(norm_1, norm_2_sq, norm_inf):
    """Numba-accelerated C2 computation"""
    if norm_1 < 1e-12 or norm_inf < 1e-12:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

def evaluate_function(f):
    """Evaluate the function and compute C2"""
    try:
        # Fast autoconvolution
        g = compute_autoconvolution_numba(f)

        # Fast norm computations
        norm_1, norm_2_sq, norm_inf = compute_norms_numba(g)

        # C2 computation
        c2 = compute_c2_numba(norm_1, norm_2_sq, norm_inf)

        return c2, g
    except Exception:
        return 0.0, np.array([0.0])

def construct_function() -> list[float]:
    """
    Construct a step function that maximizes C2 = ||g||₂² / (||g||₁ · ||g||∞)
    where g = f*f (autoconvolution) and f is the step function.
    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters
    max_time = 90.0  # seconds
    start_time = time.time()

    # Initialize best solution
    best_c2 = 0.0
    best_f = None

    # Strategy 1: High-resolution uniform distribution
    n_uniform = 2000
    f_uniform = np.ones(n_uniform) * 0.5  # Flat profile
    c2_uniform, _ = evaluate_function(f_uniform)

    if c2_uniform > best_c2:
        best_c2 = c2_uniform
        best_f = f_uniform.copy()

    # Strategy 2: Enhanced Gaussian peaks with adaptive construction
    def adaptive_gaussian_construction():
        """Create optimized Gaussian-shaped peaks with adaptive parameters"""
        # Fixed resolution
        f = np.zeros(2000)
        x = np.linspace(-0.25, 0.25, len(f))

        # Try different numbers of peaks
        n_peaks_list = [6, 8, 10, 12]

        best_local_c2 = 0.0
        best_local_f = None

        for n_peaks in n_peaks_list:
            # Use logarithmic spacing for peak positions across domain
            # This ensures better spread across multiple scales
            log_min = np.log(0.01)  # Minimum spacing in log space
            log_max = np.log(0.4)   # Maximum spacing in log space
            log_spacings = np.logspace(log_min, log_max, n_peaks + 2)[1:-1]  # Exclude endpoints

            # Distribute peaks more evenly with log spacing
            peak_positions = []
            for i in range(n_peaks):
                # Position based on logarithmic distribution
                pos = -0.25 + np.sum(log_spacings[:i+1])  # cumulative log spacing
                # Add small randomness to avoid perfect periodicity
                pos += np.random.uniform(-0.005, 0.005)
                peak_positions.append(pos)

            # Ensure minimum spacing between peaks (at least 0.05 of domain width)
            min_spacing = 0.05
            adjusted_positions = []
            for i, pos in enumerate(peak_positions):
                # Check against previous peaks
                if i > 0:
                    prev_pos = adjusted_positions[-1]
                    min_allowed = prev_pos + min_spacing
                    pos = max(pos, min_allowed)
                adjusted_positions.append(pos)

            # Create peaks with adaptive height adjustment based on intermediate evaluation
            temp_f = np.zeros_like(f)

            for i, pos in enumerate(adjusted_positions):
                # Start with moderate amplitude and adjust based on feedback
                peak_height = 0.7 + np.random.random() * 0.3  # Range: 0.7-1.0
                gaussian = peak_height * np.exp(-0.5 * ((x - pos) / 0.04)**2)
                temp_f += gaussian

            # Ensure non-negative and normalize properly
            temp_f = np.maximum(temp_f, 0)
            if np.sum(temp_f) > 0:
                temp_f = temp_f / np.sum(temp_f) * 10  # Scale appropriately
            else:
                temp_f = np.ones_like(temp_f) * 0.1

            # Quick intermediate check to avoid wasting time on obviously poor candidates
            try:
                test_c2, _ = evaluate_function(temp_f)
                if test_c2 > best_local_c2:
                    best_local_c2 = test_c2
                    best_local_f = temp_f.copy()
            except Exception:
                continue

        return best_local_f if best_local_f is not None else np.ones(2000) * 0.1

    # Try adaptive Gaussian configurations with multiple attempts
    for attempt in range(15):  # Reduced from 20 to allow more time for better strategies
        if time.time() - start_time > max_time * 0.8:
            break

        try:
            f_gaussian = adaptive_gaussian_construction()

            c2_gaussian, _ = evaluate_function(f_gaussian)

            if c2_gaussian > best_c2:
                best_c2 = c2_gaussian
                best_f = f_gaussian.copy()

        except Exception:
            continue

    # Strategy 3: Adaptive optimization around best so far
    if best_f is not None:
        # Refinement using gradient ascent (simplified version)
        current_f = best_f.copy()
        best_refined_f = current_f.copy()
        best_refined_c2 = best_c2

        # Simple hill climbing approach
        for iteration in range(100):
            if time.time() - start_time > max_time * 0.95:
                break

            # Random perturbation
            perturbed_f = current_f.copy()

            # Apply small perturbations to random elements
            indices_to_modify = np.random.choice(len(current_f),
                                               size=max(1, len(current_f) // 50),
                                               replace=False)

            for idx in indices_to_modify:
                # Add small noise
                perturbed_f[idx] += np.random.normal(0, 0.01)
                perturbed_f[idx] = max(0, perturbed_f[idx])  # Keep non-negative

            # Evaluate perturbed function
            c2_new, _ = evaluate_function(perturbed_f)

            if c2_new > best_refined_c2:
                best_refined_c2 = c2_new
                best_refined_f = perturbed_f.copy()
                current_f = perturbed_f.copy()
            else:
                # Sometimes accept worse solutions with low probability (simulated annealing)
                if np.random.random() < 0.05:
                    current_f = perturbed_f.copy()

        if best_refined_c2 > best_c2:
            best_c2 = best_refined_c2
            best_f = best_refined_f.copy()

    # Strategy 4: Final optimization with FFT for large functions
    if best_f is not None and len(best_f) > 1000:
        # For very large functions, use better optimization
        final_f = best_f.copy()

        # Normalize
        if np.sum(final_f) > 0:
            final_f = final_f / np.sum(final_f) * 10

        # Final check
        final_c2, _ = evaluate_function(final_f)
        if final_c2 > best_c2:
            best_c2 = final_c2
            best_f = final_f.copy()

    # If nothing found, return a reasonable default
    if best_f is None:
        best_f = np.ones(1000) * 0.5

    # Convert to list format
    result = best_f.tolist()

    # Post-processing to make sure it meets requirements
    result = [max(0, x) for x in result]

    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")