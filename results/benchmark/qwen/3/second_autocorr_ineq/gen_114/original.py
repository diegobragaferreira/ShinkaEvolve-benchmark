# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import njit

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

    # Direct convolution computation
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
        if g_norm1 <= 1e-12 or g_norm_inf <= 1e-12:
            return 0.0

        return g_norm2_sq / (g_norm1 * g_norm_inf)
    except Exception as e:
        return 0.0

def initialize_smart_parameters(size: int) -> np.ndarray:
    """
    Initialize step function parameters with smart patterns that tend to perform well.
    """
    # Create a combination of smooth and sharp features
    t = np.linspace(-1, 1, size)

    # Base pattern with multiple frequency components
    pattern = (
        1.0 +
        0.5 * np.sin(2 * np.pi * t) +
        0.3 * np.cos(4 * np.pi * t) +
        0.2 * np.sin(6 * np.pi * t) +
        0.1 * np.cos(8 * np.pi * t)
    )

    # Add some sharp transitions to encourage good autoconvolution behavior
    # Create a few localized spikes that can create strong peaks in autoconvolution
    spike_positions = np.random.choice(size, size=min(5, size//10), replace=False)
    for pos in spike_positions:
        pattern[pos] += np.random.uniform(0.5, 1.5)

    # Ensure non-negativity and normalize
    pattern = np.maximum(pattern, 0.0)

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

def gradient_free_optimization(size: int, max_iter: int = 1000) -> list:
    """
    Perform gradient-free optimization to find high C2 step function.
    """
    # Initialize with smart pattern
    initial_params = initialize_smart_parameters(size)

    # Optimization with L-BFGS
    result = minimize(
        smooth_objective,
        initial_params,
        args=(size,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6},
        bounds=[(0, 1000) for _ in range(size)]
    )

    # Get optimized parameters
    optimized_params = result.x

    # Project to discrete solution
    final_solution = create_discrete_projection(optimized_params)

    # Further refine with local search around the optimum
    best_solution = final_solution.copy()
    best_value = -smooth_objective(final_solution, size)

    # Local refinement by perturbing randomly
    for _ in range(20):
        # Create small random perturbation
        perturbed = final_solution + np.random.normal(0, 0.1, len(final_solution))
        # Project
        perturbed = create_discrete_projection(perturbed)

        # Evaluate
        value = -smooth_objective(perturbed, size)
        if value > best_value:
            best_value = value
            best_solution = perturbed.copy()

    return best_solution.tolist()

def evolve_with_population(size: int, generations: int = 30) -> list:
    """Evolutionary approach to find good initial solution."""
    # Population-based approach for exploration
    population_size = 15
    population = []
    
    # Initialize population with smart patterns
    for _ in range(population_size):
        individual = initialize_smart_parameters(size)
        population.append(individual)
    
    best_c2 = -float('inf')
    best_solution = None
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            f = create_discrete_projection(individual)
            c2 = calculate_c2(f.tolist())
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = f.tolist()
        
        # Tournament selection, crossover, and mutation
        new_population = []
        
        # Elitism
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            selected_indices = np.random.choice(len(population), size=tournament_size, replace=False)
            winner_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
            
            # Clone winner
            parent = population[winner_idx].copy()
            
            # Mutate
            mutated = parent.copy()
            for i in range(len(mutated)):
                if np.random.random() < 0.1:
                    mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.1))
            
            new_population.append(mutated)
        
        population = new_population[:population_size]
    
    return best_solution if best_solution is not None else [1.0] * size

def construct_function() -> list:
    """
    Main function to construct step-function with high C2 value using hybrid optimization.
    
    Returns:
        List of step heights that maximize C2
    """
    # Try different sizes for better results
    sizes_to_try = [750, 1000, 1250]  # Focus on medium-to-large sizes for better resolution
    best_c2 = -float('inf')
    best_solution = None
    
    for size in sizes_to_try:
        try:
            # First try evolutionary approach for good starting point
            evol_solution = evolve_with_population(size, 30)
            evol_c2 = calculate_c2(evol_solution)
            
            # Then refine with gradient-free optimization
            refined_solution = gradient_free_optimization(size, max(500, 1000 - (size // 100)))
            refined_c2 = calculate_c2(refined_solution)
            
            # Use the better of the two
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_solution
            elif evol_c2 > best_c2:
                best_c2 = evol_c2
                best_solution = evol_solution
                
            print(f"Size {size}: Best C2 = {best_c2:.6f}")
        except Exception as e:
            print(f"Failed at size {size}: {e}")
            continue
    
    # Final refinement on best solution
    if best_solution is not None:
        # Apply additional local refinement
        refined_solution = best_solution.copy()
        for _ in range(10):
            # Simple hill climbing
            idx = np.random.randint(len(refined_solution))
            old_val = refined_solution[idx]

            # Try small changes
            new_val = max(0, old_val + np.random.normal(0, 0.1))
            refined_solution[idx] = new_val

            # Test if this improves the solution
            test_c2 = calculate_c2(refined_solution)
            if test_c2 > best_c2:
                best_c2 = test_c2
            else:
                refined_solution[idx] = old_val  # Revert if worse

        best_solution = refined_solution

    return best_solution if best_solution is not None else [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
