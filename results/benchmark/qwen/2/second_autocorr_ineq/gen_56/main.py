# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time
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

def create_individual(size):
    """Create a random individual"""
    return [random.uniform(0, 1) for _ in range(size)]

def mutate_individual(individual):
    """Mutate an individual"""
    for i in range(len(individual)):
        if random.random() < 0.1:  # 10% mutation rate
            individual[i] = max(0, individual[i] + random.gauss(0, 0.1))
    return individual

def adaptive_gaussian_construction():
    """
    Construct function using adaptive Gaussian-based peak placement.
    This approach places peaks strategically to maximize C2 ratio by creating
    well-behaved autoconvolutions with balanced L2², L1, and L-infinity norms.
    """
    # Number of steps - use larger number for better resolution
    n_steps = random.randint(1000, 5000)

    # Create positions using log-uniform distribution to avoid clustering
    # This helps distribute peaks more evenly across the domain
    positions = []
    n_peaks = max(5, min(50, n_steps // 100))  # Dynamic number of peaks

    # Generate log-uniform distributed positions across [-0.25, 0.25]
    for i in range(n_peaks):
        # Use log-uniform distribution for better spatial distribution
        if i == 0:
            pos = random.uniform(-0.25, -0.05)  # First peak near left boundary
        elif i == n_peaks - 1:
            pos = random.uniform(0.05, 0.25)   # Last peak near right boundary
        else:
            # Middle peaks distributed log-uniformly
            log_min = math.log(0.05) if i < n_peaks//2 else math.log(0.05)
            log_max = math.log(0.25) if i < n_peaks//2 else math.log(0.25)
            log_pos = random.uniform(log_min, log_max)
            pos = math.exp(log_pos) * (-1 if i % 2 == 0 else 1)  # Alternate sides
            # Keep within bounds
            pos = max(-0.25, min(0.25, pos))
        positions.append(pos)

    # Sort positions for proper step function construction
    positions.sort()

    # Ensure minimum gap between peaks to prevent narrow autoconvolution interference
    min_gap = 0.05  # At least 0.05 units between peaks
    adjusted_positions = []
    for i, pos in enumerate(positions):
        if i == 0:
            adjusted_positions.append(pos)
        else:
            # Ensure minimum gap from previous peak
            prev_pos = adjusted_positions[-1]
            if abs(pos - prev_pos) < min_gap:
                # Shift to maintain gap
                adjusted_positions.append(prev_pos + min_gap * (1 if pos >= prev_pos else -1))
            else:
                adjusted_positions.append(pos)

    # Generate amplitudes with adaptive scaling based on position
    amplitudes = []
    for i, pos in enumerate(adjusted_positions):
        # Amplitude inversely related to distance from center (higher at center)
        distance_from_center = abs(pos)
        # Use exponential decay from center
        amp = max(0.1, 1.0 * math.exp(-distance_from_center * 8))
        # Add some randomness for variety
        amp *= random.uniform(0.8, 1.2)
        amplitudes.append(amp)

    # Apply adaptive peak reduction to prevent L-infinity dominance
    # If peaks are too sharp, reduce their amplitudes
    peak_heights = [a for a in amplitudes]
    max_peak = max(peak_heights) if peak_heights else 1.0

    if max_peak > 0:
        # Check if any peaks are disproportionately high
        total_amplitude = sum(amplitudes)
        avg_amp = total_amplitude / len(amplitudes) if len(amplitudes) > 0 else 1.0

        # If max peak is more than 3x average, reduce it
        if max_peak > avg_amp * 3:
            # Reduce all amplitudes proportionally
            reduction_factor = 1.0
            for i in range(len(amplitudes)):
                if amplitudes[i] > avg_amp * 1.5:
                    reduction_factor = min(reduction_factor, avg_amp / (amplitudes[i] * 0.7))

            if reduction_factor < 1.0:
                amplitudes = [amp * reduction_factor for amp in amplitudes]

    # Now create the step function on [-1/4, 1/4]
    # Initialize all steps to zero
    result = [0.0] * n_steps

    # Map peak positions to step indices
    step_width = 0.5 / n_steps
    half_domain = 0.25

    for i, (pos, amp) in enumerate(zip(adjusted_positions, amplitudes)):
        # Map position to step index
        step_index = int((pos + half_domain) / step_width)
        step_index = max(0, min(n_steps - 1, step_index))

        # Apply Gaussian-like shape around that point
        # Width parameter for the Gaussian (controlled by number of steps)
        sigma = max(1, n_steps // 100)

        # Apply Gaussian influence to nearby steps
        for j in range(max(0, step_index - 3*sigma), min(n_steps, step_index + 3*sigma + 1)):
            distance = abs(j - step_index)
            gaussian_val = amp * math.exp(-0.5 * (distance/sigma)**2)
            result[j] = max(result[j], gaussian_val)

    # Ensure all values are non-negative
    result = [max(0, val) for val in result]

    # Normalize to reasonable scale
    max_val = max(result) if max(result) > 0 else 1.0
    if max_val > 0:
        result = [val / max_val * 2.0 for val in result]

    # Apply smoothing to reduce oscillations
    if len(result) > 50:
        result = signal.savgol_filter(result, min(51, len(result)-1), 3)
        result = [max(0, val) for val in result]

    return result

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Try multiple approaches to find good solution
    best_result = []
    best_c2 = 0

    # Approach 1: Adaptive Gaussian Construction (NEW APPROACH)
    try:
        gaussian_result = adaptive_gaussian_construction()
        if gaussian_result:
            # Evaluate gaussian result
            f_vals = np.array(gaussian_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)

                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = gaussian_result
    except Exception as e:
        pass

    # Approach 2: Evolutionary algorithm for backup
    if len(best_result) == 0 or best_c2 < 0.9:
        try:
            evolved_result = evolve_step_function()
            if evolved_result:
                # Evaluate evolved result
                f_vals = np.array(evolved_result, dtype=np.float64)
                f_vals = np.maximum(f_vals, 0.0)
                if np.sum(f_vals) > 0:
                    g_vals = compute_autoconvolution_numba(f_vals)
                    l2_sq, l1, linf = compute_norms_numba(g_vals)

                    if l1 > 1e-15 and linf > 1e-15:
                        c2 = l2_sq / (l1 * linf)
                        if c2 > best_c2:
                            best_c2 = c2
                            best_result = evolved_result
        except Exception as e:
            pass

    # If no good result from evolution, fallback to a more informed approach
    if len(best_result) == 0:
        # Use a heuristic approach with more structured sampling
        n_steps = 500  # Fixed size for consistency
        # Create a step function that balances peaks and flat regions
        # This is a simplified version but more principled than pure random
        f_values = np.random.gamma(2, 2, n_steps)  # Gamma distribution gives positive values
        f_values = f_values / np.max(f_values) * 2  # Scale to reasonable range
        f_values = np.maximum(f_values, 0)

        # Apply some smoothing to reduce extreme variations
        f_values = signal.savgol_filter(f_values, min(51, len(f_values)-1), 3) if len(f_values) > 50 else f_values
        f_values = np.maximum(f_values, 0)

        best_result = f_values.tolist()

    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")