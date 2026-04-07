# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit
import random
import time

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values):
    """Optimized computation of autoconvolution norms using numba"""
    n = len(f_values)

    # Initialize autoconvolution array
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Keep only center portion
    half_len = n - 1
    g_center = g[half_len:-half_len]

    # Compute norms
    norm_2_squared = np.sum(g_center**2)
    norm_1 = np.sum(np.abs(g_center))
    norm_inf = np.max(np.abs(g_center))

    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation with improved numerical stability"""
    try:
        if not f_values:
            return 0.0, 0.0, 0.0, 0.0

        # Convert to numpy array for easier manipulation
        f = np.array(f_values, dtype=np.float64)

        # Ensure non-negative values
        f = np.maximum(f, 0.0)

        # Use direct computation for better control over numeric precision
        # For step functions, we can compute convolution more carefully
        n = len(f)
        g = np.zeros(2*n - 1, dtype=np.float64)

        # Manual convolution with careful indexing
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < len(g):
                    g[idx] += f[i] * f[j]

        # Keep only the valid convolution part (middle)
        half_len = n - 1
        g_valid = g[half_len:-half_len] if half_len < len(g) else g

        # Compute norms with better numerical handling
        norm_2_squared = np.sum(g_valid**2)
        norm_1 = np.sum(np.abs(g_valid))
        norm_inf = np.max(np.abs(g_valid))

        # Avoid division by zero with small epsilon
        epsilon = 1e-15
        if norm_1 < epsilon or norm_inf < epsilon:
            return 0.0, 0.0, 0.0, 0.0

        # C2 = ||g||₂² / (||g||₁ · ||g||∞)
        c2 = norm_2_squared / (norm_1 * norm_inf)

        return c2, norm_2_squared, norm_1, norm_inf
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_c2(individual):
    """Evaluate fitness of individual (step function) for maximizing C2"""
    try:
        # Ensure non-negative values
        individual = np.maximum(individual, 0.0)

        # Compute C2 value
        c2, _, _, _ = compute_autoconvolution_norms(individual)

        # Return negative because we want to maximize
        return -c2
    except Exception:
        # Return very poor fitness if error occurs
        return 1e10

def advanced_structured_initialization(n_steps):
    """Create advanced structured initial step function based on mathematical insights"""
    # Create a pattern inspired by known optimal structures for convolution
    # This uses a combination of energy concentration and systematic variation

    f_values = []

    # Create a base pattern with alternating high/low regions but with strategic energy concentration
    # We'll create a pattern where high-value regions are spaced to promote constructive interference
    # in convolution while maintaining enough variation to avoid overfitting to particular configurations

    # Divide into sections
    section_size = max(1, n_steps // 8)

    # Base values - alternating between high and low
    base_high = 1.0
    base_low = 0.1

    # Create pattern with progressive modulation
    for i in range(n_steps):
        # Determine section
        section = i // section_size

        # Alternate high/low with some systematic modulation
        if section % 2 == 0:
            value = base_high
        else:
            value = base_low

        # Add periodic modulation for complexity
        modulation = 0.1 * np.sin(2 * np.pi * i / (n_steps // 4))
        value = max(0, value + modulation)

        # Add small random component for diversity
        if i % 10 == 0:  # Only add noise occasionally to preserve structure
            noise = np.random.normal(0, 0.02)
            value = max(0, value + noise)

        f_values.append(value)

    # Apply a gentle Gaussian smoothing to reduce sharp edges
    # This helps prevent numerical artifacts during convolution
    smoothed = []
    sigma = max(1, n_steps // 50)

    for i in range(n_steps):
        weighted_sum = 0.0
        weight_sum = 0.0

        # Apply Gaussian weighting from neighboring points
        for j in range(n_steps):
            distance = abs(i - j)
            weight = np.exp(-0.5 * (distance / sigma)**2)
            weighted_sum += weight * f_values[j]
            weight_sum += weight

        smoothed.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)

    # Add structured noise for additional diversity
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, n_steps)
    smoothed = np.array(smoothed) + noise
    smoothed = np.maximum(smoothed, 0.0)  # Ensure non-negative

    # Normalize to maintain reasonable magnitude
    total = np.sum(smoothed)
    if total > 0:
        smoothed = smoothed * n_steps / total

    return smoothed.tolist()

def multi_start_differential_evolution(n_steps, max_evaluations=2000):
    """Run multiple differential evolution runs with optimized initializations"""
    best_c2 = -float('inf')
    best_solution = None

    # Run fewer but more strategic optimizations to save time
    num_starts = 3  # Reduced from 5 for efficiency

    for start_idx in range(num_starts):
        # Use different seed for each start
        np.random.seed(start_idx * 100 + 42)

        # Use more focused initialization strategies
        if start_idx == 0:
            # Advanced structured initialization (most promising)
            initial_guess = advanced_structured_initialization(n_steps)
        elif start_idx == 1:
            # Simple alternating pattern with noise
            initial_guess = [1.0 if i % 2 == 0 else 0.1 for i in range(n_steps)]
            # Add noise
            np.random.seed(start_idx)
            noise = np.random.normal(0, 0.03, n_steps)
            initial_guess = np.array(initial_guess) + noise
            initial_guess = np.maximum(initial_guess, 0.0)
        else:
            # Random initialization with some structure
            initial_guess = [np.random.random() * 0.5 + 0.25 for _ in range(n_steps)]

        # Adjust population size based on iteration count
        # Use smaller populations for faster convergence
        popsize = min(15, max(5, n_steps // 150 + 5))

        # Set bounds for each parameter
        bounds = [(0.0, 10.0) for _ in range(n_steps)]

        try:
            # Use differential evolution with optimized parameters
            result = differential_evolution(
                evaluate_c2,
                bounds,
                maxiter=max_evaluations // num_starts,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=start_idx * 42,
                disp=False,
                polish=False  # Skip polishing to save time
            )

            # Check if this solution is better
            current_c2 = -result.fun
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = result.x.tolist()

        except Exception:
            continue

    # Fallback to advanced initialization if nothing worked
    return best_solution if best_solution is not None else advanced_structured_initialization(n_steps)

def local_refinement(initial_solution, max_iter=20):
    """Apply local optimization to refine the solution"""
    try:
        bounds = [(0.0, 10.0) for _ in range(len(initial_solution))]

        # Apply bounds constraint and local minimization
        result = minimize(
            evaluate_c2,
            initial_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter},
            tol=1e-6
        )

        if result.success:
            return result.x.tolist()
    except:
        pass
    return initial_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid optimization"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()

    try:
        # Increase problem size for better solution quality within time limits
        # This gives us more degrees of freedom to create effective step functions
        n_steps = 1500  # Increased from 1000 for better resolution
        max_evaluations = 2000  # Total evaluations across all runs

        # Multi-start differential evolution
        intermediate_solution = multi_start_differential_evolution(n_steps, max_evaluations)

        # Local refinement with more iterations for better convergence
        refined_solution = local_refinement(intermediate_solution, max_iter=30)

        # Ensure non-negative values
        refined_solution = [max(0, x) for x in refined_solution]

        # Normalize to avoid extreme values that might cause numerical issues
        total = sum(refined_solution)
        if total > 0:
            refined_solution = [x / total * len(refined_solution) for x in refined_solution]

        # Final check to ensure reasonable bounds
        refined_solution = [min(10.0, max(0.0, x)) for x in refined_solution]

        return refined_solution

    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Fallback due to error: {e}")
        return advanced_structured_initialization(500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")