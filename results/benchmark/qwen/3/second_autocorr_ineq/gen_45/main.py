# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
from typing import List, Tuple
import random

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the norms of the autoconvolution g = f*f using fast numba compilation
    """
    n = len(f_vals)
    # Create convolution result array - size 2*n-1
    g = np.zeros(2 * n - 1)

    # Compute convolution manually (f*f) - JIT compiled
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)
    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

@njit
def calculate_c2(f_vals):
    """
    Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞) using JIT compilation
    """
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_2_sq / (norm_1 * norm_inf)

@njit
def compute_trapz_norm2(g_values):
    """Compute L2 norm squared using piecewise linear integration"""
    if len(g_values) < 2:
        return g_values[0] * g_values[0] if len(g_values) > 0 else 0.0

    # Simplified trapezoidal integration for g²
    h = 1.0 / (len(g_values) - 1)  # Normalized spacing
    total = 0.0

    # For piecewise linear integration of g², use (y₁² + y₁y₂ + y₂²)/3 * h per segment
    for i in range(len(g_values) - 1):
        y1, y2 = g_values[i], g_values[i+1]
        total += (y1*y1 + y1*y2 + y2*y2) * h / 3.0

    return total

@njit
def compute_trapz_norm1(g_values):
    """Compute L1 norm using trapezoidal rule"""
    if len(g_values) < 2:
        return g_values[0] if len(g_values) > 0 else 0.0

    h = 1.0 / (len(g_values) - 1)  # Normalized spacing
    total = 0.0

    # Trapezoidal rule for ∫|g|
    for i in range(len(g_values) - 1):
        y1, y2 = g_values[i], g_values[i+1]
        total += (y1 + y2) * h / 2.0

    return total

def construct_function() -> list[float]:
    """
    Construct a step-function with high C2 value using evolved optimization strategy
    """
    # Use a better heuristic approach inspired by successful patterns
    # Generate initial seed pattern - alternating high/low values with structure
    
    # Try to create patterns that might yield high C2 values
    n = random.randint(200, 1000)  # Reasonable size range
    
    # Create structured pattern with some randomness
    f_values = np.zeros(n)
    
    # Create alternating pattern with some peaks to encourage good convolutions
    for i in range(n):
        if i % 4 == 0:
            f_values[i] = random.uniform(0.5, 1.0)  # High values
        elif i % 4 == 1:
            f_values[i] = random.uniform(0.2, 0.6)  # Medium values
        elif i % 4 == 2:
            f_values[i] = random.uniform(0.0, 0.3)  # Low values
        else:  # i % 4 == 3
            f_values[i] = random.uniform(0.0, 0.5)  # Low to medium
    
    # Add some random noise to avoid local minima trapping
    noise_level = 0.1
    f_values += np.random.normal(0, noise_level, n)

    # Ensure all values are non-negative
    f_values = np.maximum(f_values, 0)

    # Now run a simple local search to refine this pattern
    best_f = f_values.copy()
    best_c2 = calculate_c2(best_f)
    
    # Try a few mutations to see if we can improve
    for _ in range(50):
        # Small random perturbation
        mutated = best_f.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance to mutate
                mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.05))
        
        mutated_c2 = calculate_c2(mutated)
        if mutated_c2 > best_c2:
            best_f = mutated
            best_c2 = mutated_c2
    
    return best_f.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")