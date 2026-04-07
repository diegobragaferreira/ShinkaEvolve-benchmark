# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import jit
import time
import optuna
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import warnings

# Global constants for optimization
MAX_ITERATIONS = 150
INITIAL_POPSIZE = 20
ADAPTIVE_POPSIZES = [10, 15, 20, 25, 30]
CONVERGENCE_WINDOW = 20
MIN_C2_IMPROVEMENT = 1e-8

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """
    Compute autoconvolution g = f*f using efficient Numba implementation
    """
    n = len(f_vals)
    # Convolution result has length 2*n-1
    conv_len = 2 * n - 1
    g = np.zeros(conv_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < conv_len:
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """
    Compute C2 value using precise Numba implementation with correct integration
    """
    if len(g_vals) == 0:
        return 0.0

    # Compute norms using proper trapezoidal integration for L2^2
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (maximum absolute value)
    for i in range(len(g_vals)):
        abs_val = abs(g_vals[i])
        if abs_val > g_max:
            g_max = abs_val

    # For L2^2 norm using trapezoidal integration
    # For convolution on domain [-1/2, 1/2], with len(g_vals) points
    # Step width h = 1.0 / (len(g_vals) - 1) if len > 1, else 0.001
    if len(g_vals) >= 2:
        # Trapezoidal rule: h * (y0^2 + 2*y1^2 + ... + yn-1^2)/2
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]
    else:
        g_l2_sq = 0.0

    # Compute C2 with numerical stability
    if g_l1 > 1e-15 and g_max > 1e-15:
        return g_l2_sq / (g_l1 * g_max)
    else:
        return 0.0

def compute_autoconvolution_fft(f_vals):
    """
    Compute autoconvolution using FFT for better numerical stability.
    This is more efficient and numerically stable for large arrays.
    """
    n = len(f_vals)
    # Pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    f_padded = np.pad(f_vals, (0, padded_length - n), mode='constant', constant_values=0)

    # FFT-based convolution
    F = fft(f_padded)
    G_fft = F * F  # Point-wise multiplication in frequency domain
    g = ifft(G_fft).real[:padded_length]

    return g

def compute_c2_fft(f_vals):
    """Compute C2 value using FFT-based autoconvolution"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Compute autoconvolution using FFT
        g_vals = compute_autoconvolution_fft(f_vals)

        # Compute C2
        c2 = compute_c2_numba(g_vals)
        return c2

    except Exception as e:
        return 0.0

def fourier_domain_initialization(size):
    """
    Create initialization based on Fourier domain analysis for better spectral properties.
    This approach aims to create functions that naturally produce high C2 values
    by leveraging known relationships between spectral properties and C2.
    """
    # Create a pattern that tends to have good autoconvolution behavior
    # Based on the principle that functions with specific spectral characteristics
    # often yield better C2 values

    # Generate base pattern with multiple frequency components
    x = np.linspace(0, 1, size)

    # Low frequency dominant signal (strongest component)
    base_low = 0.8 + 0.2 * np.cos(2 * np.pi * x * 3)  # 3 periods

    # Medium frequency components
    medium = 0.1 * np.sin(2 * np.pi * x * 10)  # 10 periods
    medium2 = 0.05 * np.cos(2 * np.pi * x * 6)  # 6 periods

    # High frequency for complexity (but not too much to avoid noise)
    high = 0.05 * np.sin(2 * np.pi * x * 20)  # 20 periods

    # Combine all components
    combined = base_low + medium + medium2 + high

    # Add structured variations to make it more interesting
    for i in range(size):
        if i % 7 == 0:
            combined[i] += 0.15
        elif i % 7 == 3:
            combined[i] -= 0.1

    # Ensure non-negativity and normalize
    combined = np.maximum(combined, 0)

    # Normalize to reasonable scale
    if np.sum(combined) > 0:
        combined = combined / np.sum(combined) * 2.0

    return combined.tolist()

def sophisticated_initialization(size):
    """
    Create a sophisticated initial population with structured patterns
    """
    # Try different initialization strategies and pick the best
    strategies = [
        lambda s: fourier_domain_initialization(s),
        lambda s: [0.5 + 0.3 * np.sin(i * 0.5) for i in range(s)],
        lambda s: [1.0 if i % 3 == 0 else 0.2 for i in range(s)],
        lambda s: [0.5 + 0.4 * np.sin(i * 0.3) + 0.1 * np.random.randn() for i in range(s)]
    ]

    best_c2 = -1.0
    best_params = None

    for strategy in strategies:
        try:
            params = strategy(size)
            c2 = compute_c2_fft(params)
            if c2 > best_c2:
                best_c2 = c2
                best_params = params[:]
        except:
            continue

    # If no strategy worked, fall back to basic initialization
    if best_params is None:
        # Create alternating high-low segments with sinusoidal modulation
        segments = max(1, size // 20)  # At least 1 segment
        f_vals = []

        for i in range(segments):
            segment_size = size // segments
            # Pattern based on position - creating structured variation
            base_height = 0.5 + 0.3 * np.sin(i * 0.7)
            if i % 3 == 0:
                height = 1.0  # Peaks
            elif i % 3 == 1:
                height = 0.3  # Valleys
            else:
                height = base_height  # Middle values

            segment_vals = [height] * segment_size
            f_vals.extend(segment_vals)

        # Trim or extend to exact size
        if len(f_vals) > size:
            f_vals = f_vals[:size]
        elif len(f_vals) < size:
            # Pad with middle values
            padding = [0.5] * (size - len(f_vals))
            f_vals.extend(padding)

        # Add controlled noise for diversity
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, size)
        f_vals = np.array(f_vals) + noise
        f_vals = np.maximum(f_vals, 0)  # Ensure non-negative

        best_params = f_vals.tolist()

    return best_params

def adaptive_evolutionary_optimization(size, max_iter=MAX_ITERATIONS, adaptive=True):
    """
    Perform evolutionary optimization with adaptive parameters based on convergence behavior
    """
    if adaptive:
        # Implement adaptive population sizing based on convergence
        initial_popsize = INITIAL_POPSIZE
        popsize_schedule = ADAPTIVE_POPSIZES
    else:
        initial_popsize = INITIAL_POPSIZE
        popsize_schedule = [INITIAL_POPSIZE]

    # Start with sophisticated initialization
    x0 = sophisticated_initialization(size)

    # Set bounds for optimization
    bounds = [(0, 10)] * size

    # Track convergence for adaptive parameters
    prev_best_c2 = -np.inf
    convergence_streak = 0
    last_improvement_iteration = 0

    # Initialize with the first population size
    current_popsize = popsize_schedule[0] if adaptive and len(popsize_schedule) > 0 else initial_popsize

    # Parameters for differential evolution
    de_params = {
        'mutation': (0.5, 1.0),
        'recombination': 0.7,
        'popsize': current_popsize,
        'maxiter': max_iter,
        'seed': 42,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'disp': False
    }

    # Run optimization with adaptive parameters
    try:
        result = differential_evolution(
            lambda x: -compute_c2_fft(x),
            bounds,
            **de_params
        )

        return result.x
    except Exception:
        # Fallback to simpler approach if needed
        return x0

def stochastic_local_search(initial_params, max_iter=100, perturbation_scale=0.01):
    """
    Local search with stochastic perturbations to escape local optima
    """
    def objective(x):
        return -compute_c2_fft(x)

    # Start with the initial parameters
    current_params = np.array(initial_params)
    best_params = current_params.copy()
    best_c2 = compute_c2_fft(best_params)

    # Perform gradient-based local search first
    try:
        result = minimize(
            objective,
            current_params,
            method='L-BFGS-B',
            bounds=[(0, 10) for _ in range(len(current_params))],
            options={'maxiter': max_iter//2, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        current_params = result.x
        current_c2 = compute_c2_fft(current_params)
        if current_c2 > best_c2:
            best_params = current_params.copy()
            best_c2 = current_c2
    except:
        pass

    # Then apply stochastic perturbations to escape local optima
    try:
        # Apply multiple stochastic perturbations
        for iteration in range(max_iter//2):
            # Create a perturbed copy
            perturbed_params = current_params + np.random.normal(0, perturbation_scale, len(current_params))
            perturbed_params = np.maximum(perturbed_params, 0)  # Ensure non-negative

            # Evaluate the perturbed version
            perturbed_c2 = compute_c2_fft(perturbed_params)

            # Accept if better
            if perturbed_c2 > best_c2:
                best_params = perturbed_params.copy()
                best_c2 = perturbed_c2
                current_params = perturbed_params.copy()

    except Exception:
        pass

    return best_params

def multi_stage_optimization():
    """
    Run multi-stage optimization pipeline with enhanced strategies
    """
    start_time = time.time()
    best_c2 = -np.inf
    best_params = None

    # Stage 1: Multi-config evolutionary optimization with adaptive population sizing
    configurations = [
        (200, 50),
        (400, 75),
        (600, 100),
        (800, 125),
        (1000, 150)
    ]

    # Add some randomness
    for _ in range(10):
        dim = np.random.randint(200, 1000)
        iterations = np.random.randint(50, 150)
        configurations.append((dim, iterations))

    for n_steps, n_iterations in configurations:
        if time.time() - start_time > 85:
            break

        try:
            # Try different population sizes to see what works best
            for popsize in [10, 15, 20, 25, 30]:
                if time.time() - start_time > 85:
                    break

                # Run evolutionary optimization with adaptive parameter
                params = adaptive_evolutionary_optimization(
                    n_steps,
                    max_iter=n_iterations,
                    adaptive=True
                )

                # Apply stochastic local refinement
                refined_params = stochastic_local_search(params, max_iter=50)

                # Evaluate final result
                c2 = compute_c2_fft(refined_params)

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_params.copy()

        except Exception as e:
            continue

    # Stage 2: Additional fine-tuning with specialized refinements
    if best_params is not None and time.time() - start_time < 85:
        try:
            # Use a more aggressive refinement approach
            final_params = stochastic_local_search(best_params, max_iter=100, perturbation_scale=0.02)
            final_c2 = compute_c2_fft(final_params)

            if final_c2 > best_c2:
                best_c2 = final_c2
                best_params = final_params
        except Exception:
            pass

    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    try:
        # Main optimization strategy
        best_params, best_c2 = multi_stage_optimization()

        # If we found a good solution, use it
        if best_params is not None and best_c2 > 0:
            return best_params

        # Fallback to deterministic approach
        size = 1000
        best_f_vals = adaptive_evolutionary_optimization(size, MAX_ITERATIONS, adaptive=True)
        best_f_vals = stochastic_local_search(best_f_vals, max_iter=50)
        c2_val = compute_c2_fft(best_f_vals)
        print(f"Best C2 found: {c2_val}")

        # Return the optimized values
        return best_f_vals.tolist()

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Final fallback to structured initialization
        return sophisticated_initialization(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")