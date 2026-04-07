# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from numba import njit
import warnings
warnings.filterwarnings('ignore')

@njit
def compute_autoconvolution_norms_numba(f_array: np.ndarray) -> tuple:
    """
    Compute the L2, L1, and L-infinity norms of the autoconvolution of f.
    JIT compiled version for speed.

    Args:
        f_array: Numpy array of step heights

    Returns:
        Tuple of (||g||₂², ||g||₁, ||g||∞)
    """
    if len(f_array) == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution g = f * f (discrete convolution)
    # Manual implementation for better control and JIT compatibility
    g_len = 2 * len(f_array) - 1
    g = np.zeros(g_len, dtype=np.float64)

    # Direct convolution computation - fully JIT compiled
    for i in range(len(f_array)):
        for j in range(len(f_array)):
            g[i + j] += f_array[i] * f_array[j]

    # Compute norms using manual loop for JIT compatibility
    # ||g||₂² - integrate g² using trapezoidal rule approximation
    g_squared = g * g
    trapz_sum = 0.0

    # Use trapezoidal integration for ||g||₂²
    if len(g) >= 2:
        h = 1.0 / (len(g) - 1)  # Normalized spacing
        for i in range(len(g) - 1):
            y1, y2 = g_squared[i], g_squared[i+1]
            # Correct trapezoidal formula for g² integration: (h/3)(y₁² + y₁y₂ + y₂²)
            trapz_sum += (y1*y1 + y1*y2 + y2*y2) * h / 3.0
    else:
        trapz_sum = g_squared[0] if len(g_squared) > 0 else 0.0

    # ||g||₁ - integrate |g| using trapezoidal rule
    g_abs = np.abs(g)
    trapz_l1_sum = 0.0

    if len(g) >= 2:
        h = 1.0 / (len(g) - 1)  # Normalized spacing
        for i in range(len(g) - 1):
            y1, y2 = g_abs[i], g_abs[i+1]
            trapz_l1_sum += (y1 + y2) * h / 2.0
    else:
        trapz_l1_sum = g_abs[0] if len(g_abs) > 0 else 0.0

    # ||g||∞ - infinity norm (maximum absolute value)
    g_max = np.max(np.abs(g)) if len(g) > 0 else 0.0

    return trapz_sum, trapz_l1_sum, g_max

def compute_autoconvolution_norms(f: list) -> tuple:
    """
    Compute the L2, L1, and L-infinity norms of the autoconvolution of f.

    Args:
        f: List of step heights

    Returns:
        Tuple of (||g||₂², ||g||₁, ||g||∞)
    """
    if not f:
        return 0.0, 0.0, 0.0

    # Convert to numpy array for easier manipulation
    f_array = np.array(f, dtype=np.float64)

    return compute_autoconvolution_norms_numba(f_array)

def calculate_c2(f: list) -> float:
    """
    Calculate C₂ = ||g||₂² / (||g||₁ · ||g||∞) where g = f * f.

    Args:
        f: List of step heights

    Returns:
        C₂ value
    """
    try:
        g_norm2_sq, g_norm1, g_norm_inf = compute_autoconvolution_norms(f)

        # Avoid division by zero
        if g_norm1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0

        return g_norm2_sq / (g_norm1 * g_norm_inf)
    except Exception as e:
        return 0.0

def initialize_latin_hypercube_parameters(size: int, n_samples: int = 12) -> np.ndarray:
    """
    Initialize step function parameters using Latin Hypercube Sampling for better diversity.
    """
    patterns = []
    
    # Create LHS samples for different scale and amplitude parameters
    for _ in range(n_samples):
        pattern = np.zeros(size)
        
        # Multiple Gaussian bumps with different characteristics
        num_bumps = np.random.randint(3, 8)
        for _ in range(num_bumps):
            # Random scale (log-uniform)
            scale = int(np.random.uniform(1, size//8))
            # Random amplitude
            amp = np.random.uniform(0.5, 2.0)
            # Random position
            pos = np.random.randint(0, size)
            
            # Generate Gaussian bump
            indices = np.arange(size)
            gaussian = amp * np.exp(-0.5 * (indices - pos)**2 / scale**2)
            pattern += gaussian
            
        # Add harmonic structure
        t = np.linspace(-2, 2, size)
        harmonic = (
            0.8 +
            0.3 * np.sin(2 * np.pi * t) +
            0.2 * np.cos(4 * np.pi * t) +
            0.1 * np.sin(6 * np.pi * t)
        )
        pattern += harmonic
        
        # Ensure non-negativity and normalize
        pattern = np.maximum(pattern, 0.0)
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * 100
            
        patterns.append(pattern)
        
    return patterns

def initialize_smart_parameters(size: int) -> np.ndarray:
    """
    Initialize step function parameters with enhanced multi-scale pattern.
    """
    # Create multi-scale Gaussian pattern for structured initialization
    pattern = np.zeros(size)

    # Generate multiple scales with logarithmic distribution
    scales = np.logspace(np.log10(2), np.log10(size//4), 6, base=2.0)
    scales = np.unique(scales.astype(int))  # Remove duplicates

    # Amplitude decreases with scale for hierarchical structure
    amplitudes = 1.0 / (np.arange(1, len(scales) + 1) * 2.0)

    # Place Gaussian bumps strategically
    for i, (scale, amp) in enumerate(zip(scales, amplitudes)):
        if scale >= 1:
            # Position evenly across domain
            position = int((i + 1) * size / (len(scales) + 2))
            position = max(0, min(position, size - 1))
            indices = np.arange(size)
            gaussian = amp * np.exp(-0.5 * (indices - position)**2 / scale**2)
            pattern += gaussian

    # Add harmonic structure
    t = np.linspace(-2, 2, size)
    harmonic_pattern = (
        0.8 +
        0.3 * np.sin(2 * np.pi * t) +
        0.2 * np.cos(4 * np.pi * t) +
        0.1 * np.sin(6 * np.pi * t)
    )

    # Combine all components and ensure non-negativity
    pattern = np.maximum(pattern + harmonic_pattern, 0.0)

    # Normalize to reasonable magnitude
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 100

    return pattern

def create_discrete_projection(x: np.ndarray) -> np.ndarray:
    """
    Project continuous values to valid discrete step function heights.
    Ensures non-negativity and reasonable scaling.
    """
    # Ensure non-negativity
    x_proj = np.maximum(x, 0.0)

    # Normalize to prevent extreme values that might cause numerical issues
    if np.sum(x_proj) > 0:
        x_proj = x_proj / np.sum(x_proj) * 100

    return x_proj

def smooth_objective(params: np.ndarray, size: int) -> float:
    """
    Smoothed version of the objective function for optimization.
    """
    # Convert to step function
    f = create_discrete_projection(params)

    # If we're dealing with very few parameters, pad with zeros
    if len(f) < size:
        f = np.pad(f, (0, size - len(f)), 'constant', constant_values=0)
    elif len(f) > size:
        f = f[:size]

    return -calculate_c2(f.tolist())  # Negative because we minimize

def adaptive_local_search(initial_params: np.ndarray, size: int, max_iter: int = 50) -> list:
    """
    Enhanced local search with adaptive perturbation sizes and early termination.
    """
    current_params = initial_params.copy()
    current_value = -smooth_objective(current_params, size)

    # Track improvement for early termination
    last_improvement = 0
    consecutive_no_improvement = 0

    # Start with larger perturbations and decrease over time
    for iteration in range(max_iter):
        # Adaptive perturbation size - starts large, decreases over time
        adaptive_perturbation = max(0.01, 0.5 * (1.0 - iteration / max_iter))

        # Randomly select which parameters to perturb
        num_perturb = max(1, int(len(current_params) * 0.1))
        perturb_indices = np.random.choice(len(current_params), num_perturb, replace=False)

        # Create perturbed version
        perturbed_params = current_params.copy()
        for idx in perturb_indices:
            # Adaptive perturbation with decreasing magnitude
            noise = np.random.normal(0, adaptive_perturbation)
            perturbed_params[idx] = max(0, current_params[idx] + noise)

        # Project to valid space
        perturbed_params = create_discrete_projection(perturbed_params)

        # Evaluate
        perturbed_value = -smooth_objective(perturbed_params, size)

        # Accept improvement or accept with probability based on difference
        if perturbed_value > current_value:
            current_params = perturbed_params.copy()
            current_value = perturbed_value
            last_improvement = iteration
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        # Early termination if no improvement for many iterations
        if consecutive_no_improvement > 10:
            break

    return current_params.tolist()

def coordinate_wise_refinement(initial_params: np.ndarray, size: int, max_iter: int = 30) -> list:
    """
    Perform coordinate-wise refinement on the solution.
    """
    current_params = initial_params.copy()
    current_value = -smooth_objective(current_params, size)

    # Track improvement for early termination
    last_improvement = 0
    consecutive_no_improvement = 0

    for iteration in range(max_iter):
        improved = False

        # Try improving each coordinate individually
        for i in range(len(current_params)):
            original_value = current_params[i]

            # Try small positive and negative changes
            for delta in [original_value * 0.05, -original_value * 0.05]:
                if abs(delta) < 1e-6:
                    continue

                new_params = current_params.copy()
                new_params[i] = max(0, original_value + delta)

                # Project to valid space
                new_params = create_discrete_projection(new_params)

                new_value = -smooth_objective(new_params, size)

                if new_value > current_value:
                    current_params = new_params.copy()
                    current_value = new_value
                    improved = True
                    last_improvement = iteration
                    consecutive_no_improvement = 0
                    break

            if improved:
                break

        if not improved:
            consecutive_no_improvement += 1

        # Early termination if no improvement for many iterations
        if consecutive_no_improvement > 5:
            break

    return current_params.tolist()

def multi_start_optimization(size: int, n_starts: int = 20, max_iter_per_start: int = 500) -> list:
    """
    Perform multi-start optimization with diverse initializations.
    """
    best_c2 = -float('inf')
    best_solution = None

    # Generate diverse initial solutions using LHS and smart initialization
    lhs_initializations = initialize_latin_hypercube_parameters(size, n_starts // 2)
    
    # Fill remaining slots with smart patterns
    remaining_slots = n_starts - len(lhs_initializations)
    smart_initializations = [initialize_smart_parameters(size) for _ in range(remaining_slots)]

    # Combine both types of initializations
    all_initializations = lhs_initializations + smart_initializations

    # Limit to the requested number of starts
    all_initializations = all_initializations[:n_starts]

    for i, init_params in enumerate(all_initializations):
        try:
            # Optimization with L-BFGS
            result = minimize(
                smooth_objective,
                init_params,
                args=(size,),
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_start, 'ftol': 1e-8, 'gtol': 1e-6},
                bounds=[(0, 1000) for _ in range(size)]
            )

            # Get optimized parameters
            optimized_params = result.x

            # Project to discrete solution
            final_solution = create_discrete_projection(optimized_params)

            # Enhanced local search
            refined_solution = adaptive_local_search(final_solution, size, max_iter=30)

            # Final coordinate-wise refinement
            final_refined_solution = coordinate_wise_refinement(refined_solution, size, max_iter=15)

            # Evaluate final solution
            c2_value = calculate_c2(final_refined_solution)

            if c2_value > best_c2:
                best_c2 = c2_value
                best_solution = final_refined_solution

        except Exception as e:
            print(f"Optimization failed for start {i}: {e}")
            continue

    return best_solution if best_solution is not None else [1.0] * size

def construct_function() -> list:
    """
    Main function to construct step-function with high C2 value using hybrid optimization.

    Returns:
        List of step heights that maximize C2
    """
    # Try different sizes for better results
    sizes_to_try = [1000, 1250, 1500]  # Focused on medium-to-large sizes for better resolution
    best_c2 = -float('inf')
    best_solution = None

    for size in sizes_to_try:
        try:
            # Use multi-start approach with adaptive refinement
            solution = multi_start_optimization(size, n_starts=20, max_iter_per_start=500)

            c2_value = calculate_c2(solution)
            print(f"Size {size}: C2 = {c2_value:.6f}")

            if c2_value > best_c2:
                best_c2 = c2_value
                best_solution = solution
        except Exception as e:
            print(f"Failed at size {size}: {e}")
            continue

    return best_solution if best_solution is not None else [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")