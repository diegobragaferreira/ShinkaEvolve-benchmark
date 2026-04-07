# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import math

def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    Optimized for performance with pre-computed convolution.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using fast convolution
    g = convolve(np.array(f_values), np.array(f_values), mode='full')

    # Extract the central part (valid convolution) which has length n_steps
    mid = len(g) // 2
    g_valid = g[mid : mid + n_steps]

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g_valid)-1):
        g2_sq += (dx/3) * (g_valid[i]**2 + g_valid[i]*g_valid[i+1] + g_valid[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g_valid)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g_valid))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def generate_patterned_initial_function(n_steps):
    """Generate an initial function based on mathematical insight about optimal convolution shapes"""
    # Create a function designed to produce uniform convolution profiles
    # This pattern balances peak and flat regions to encourage good C2 values

    f_values = []

    # Create a pattern that starts low, rises to a peak, then falls back down
    # but with enough variation to be interesting
    half = n_steps // 2
    quarter = n_steps // 4

    # Base pattern with multiple regions
    for i in range(n_steps):
        if i < quarter:
            # Rising edge
            f_values.append(i / quarter)
        elif i < half:
            # Peak region
            f_values.append(1.0)
        elif i < 3*quarter:
            # Falling edge
            f_values.append((3*quarter - i) / quarter)
        else:
            # Low tail
            f_values.append((n_steps - i) / quarter)

    # Apply some smoothing to reduce sharp transitions
    smoothed = []
    for i in range(n_steps):
        if i == 0 or i == n_steps - 1:
            smoothed.append(f_values[i])
        else:
            # Weighted average
            smoothed.append(0.2 * f_values[i-1] + 0.6 * f_values[i] + 0.2 * f_values[i+1])

    # Normalize to ensure reasonable magnitude
    total_area = sum(smoothed) * (0.5 / n_steps)
    if total_area > 0:
        smoothed = [x / total_area * 2.0 for x in smoothed]

    return smoothed

def adaptive_optimization_search():
    """Perform adaptive optimization with multiple resolutions and strategies"""

    # Try different configurations with varying complexity
    resolutions = [200, 300, 400, 500]

    best_solution = None
    best_c2 = -np.inf

    # Multi-start approach with different initial patterns
    for res in resolutions:
        # Try multiple random initializations for each resolution
        for start_iter in range(3):
            np.random.seed(start_iter * 1000 + res)

            # Generate initial function using pattern-based approach
            if res <= 300:
                # For smaller problems, use patterned construction
                f_values = generate_patterned_initial_function(res)
            else:
                # For larger problems, use a more structured approach
                f_values = generate_patterned_initial_function(res)
                # Add some noise for exploration
                noise_level = 0.1
                for i in range(len(f_values)):
                    if np.random.random() < 0.3:
                        f_values[i] *= (1 + np.random.normal(0, noise_level))

            # Ensure non-negativity
            f_values = [max(0, x) for x in f_values]

            # Normalize for better numerical behavior
            total = sum(f_values)
            if total > 0:
                f_values = [x / total * 10 for x in f_values]

            # Simple local improvement
            current_f = f_values.copy()
            current_c2 = compute_c2(current_f)

            # Gradient-like local search
            for _ in range(15):
                test_f = current_f.copy()
                # Modify a few points randomly
                indices = np.random.choice(len(test_f), min(8, len(test_f)//5), replace=False)
                for idx in indices:
                    # Small perturbation
                    change = np.random.normal(0, 0.05 * test_f[idx])
                    test_f[idx] = max(0, test_f[idx] + change)

                test_c2 = compute_c2(test_f)
                if test_c2 > current_c2:
                    current_c2 = test_c2
                    current_f = test_f

            # Check if this is our best solution
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = current_f.copy()

    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive approach."""

    # Try the adaptive optimization approach
    try:
        solution = adaptive_optimization_search()

        # If we got a solution, return it
        if solution is not None and len(solution) > 0:
            return solution

    except Exception as e:
        pass

    # Fallback: use simpler approach with fixed pattern
    n_steps = 300
    f_values = generate_patterned_initial_function(n_steps)

    # Final refinement
    current_f = f_values.copy()
    current_c2 = compute_c2(current_f)

    for _ in range(10):
        test_f = current_f.copy()
        # Make small random changes
        indices = np.random.choice(len(test_f), min(5, len(test_f)//4), replace=False)
        for idx in indices:
            change = np.random.normal(0, 0.1 * test_f[idx])
            test_f[idx] = max(0, test_f[idx] + change)

        test_c2 = compute_c2(test_f)
        if test_c2 > current_c2:
            current_c2 = test_c2
            current_f = test_f

    return current_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")