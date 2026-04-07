# EVOLVE-BLOCK-START

import numpy as np
from numba import njit

@njit
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    JIT compiled version with correct piecewise linear integration.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using piecewise linear integration
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution - JIT compiled loop with proper dx scaling
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                # Contribution scaled by dx for proper integration
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using trapezoidal rule for ||g||₂²
    # Using trapezoidal rule: ∫ g(x)² dx ≈ (dx/3)(g₀² + g₀g₁ + g₁²) + ...
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)  
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

@njit
def compute_c2_numba(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞) - JIT compiled version"""
    g2_sq, g1, ginf = compute_convolution_norms_numba(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid approach."""
    # Use a fixed number of steps for reproducibility and efficiency
    n_steps = 1000  # Increased from previous to improve resolution

    # Start with a mathematically informed construction that promotes flat convolutions
    # Use a symmetric 'bump' pattern - creates a reasonably flat autoconvolution
    base_height = 1.0
    f_values = []
    
    half = n_steps // 2
    for i in range(n_steps):
        if i < half:
            # Increasing ramp
            f_values.append(base_height * (i / half))
        else:
            # Decreasing ramp  
            f_values.append(base_height * ((n_steps - i) / half))

    # Normalize to reasonable values to avoid extreme peaks that could hurt C2
    total_area = sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = [x / total_area for x in f_values]

    # Apply smoothing to make the function less sensitive to small perturbations
    # This helps avoid getting trapped in poor local optima
    smoothed_f = []
    for i in range(len(f_values)):
        # Simple averaging with neighbors to smooth the function
        left = max(0, i - 1)
        right = min(len(f_values), i + 2)
        avg = sum(f_values[left:right]) / (right - left)
        smoothed_f.append(avg)
    f_values = smoothed_f

    # Now refine using adaptive local search similar to the best performer
    best_f = f_values.copy()
    best_c2 = compute_c2_numba(best_f)

    # Adaptive local search parameters - tuned for speed and effectiveness
    max_iterations = 50  # Reduced to stay within time limits
    improvement_threshold = 0.001
    patience = 8

    # Track recent improvements for adaptive behavior
    recent_improvements = []
    current_patience = 0

    # Perform adaptive local optimization
    for iteration in range(max_iterations):
        # Try small random modifications
        test_f = best_f.copy()

        # Modify a few random positions with adaptive strategy
        num_modifications = max(1, min(20, len(test_f) // 15))  # Adjusted based on performance
        
        if len(recent_improvements) > 5:
            avg_improvement = np.mean(recent_improvements[-5:])
            # Reduce modifications if we're making slow progress
            if avg_improvement < improvement_threshold * 0.1:
                num_modifications = max(1, num_modifications // 2)

        mod_indices = np.random.choice(len(test_f), num_modifications, replace=False)
        for idx in mod_indices:
            # Add small random change with bounded support
            change = np.random.normal(0, 0.05 * best_f[idx])  # Slightly reduced magnitude
            test_f[idx] = max(0, test_f[idx] + change)  # Ensure non-negativity

        # Evaluate and accept improvement
        test_c2 = compute_c2_numba(test_f)
        improvement = test_c2 - best_c2

        if test_c2 > best_c2:
            best_c2 = test_c2
            best_f = test_f
            recent_improvements.append(improvement)
            current_patience = 0
        else:
            current_patience += 1
            recent_improvements.append(improvement)

        # Early stopping if no significant improvement for several iterations
        if current_patience >= patience:
            break

    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")