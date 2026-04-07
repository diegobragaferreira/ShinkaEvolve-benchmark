# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
import numba
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# JIT compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    n = len(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    # The resulting g will have length 2*n - 1 where n is the length of f
    g_length = 2 * n - 1
    g = np.zeros(g_length)
    
    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute the norms
    # ||g||₂² = sum(g[i]²) using proper piecewise integration
    norm_g_2_squared = 0.0
    
    # For piecewise linear integration, we use trapezoidal-like approach:
    # for consecutive pairs of points (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    
    # ||g||₁ = sum(|g[i]|)
    norm_g_1 = 0.0
    for i in range(g_length):
        norm_g_1 += abs(g[i])
    
    # ||g||∞ = max(|g[i]|)
    norm_g_inf = 0.0
    for i in range(g_length):
        abs_g = abs(g[i])
        if abs_g > norm_g_inf:
            norm_g_inf = abs_g
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂² as specified in requirements
    """
    f = np.array(f_values)
    
    # Use the fast JIT-compiled version
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_fast(f)
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero with stricter thresholds
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def generate_multi_scale_initialization(n_steps: int, coarse_size: int = 100) -> List[float]:
    """
    Generate initial configuration using multi-scale sampling approach for better structure
    """
    # Start with coarse resolution to understand the landscape
    coarse_f = np.zeros(coarse_size)
    
    # Create a balanced alternating pattern with some randomness
    segment_size = max(1, coarse_size // 12)
    for i in range(0, coarse_size, segment_size):
        end_idx = min(i + segment_size, coarse_size)
        if (i // segment_size) % 2 == 0:
            # High region with some variation
            coarse_f[i:end_idx] = 0.6 + np.random.random(end_idx - i) * 0.3
        else:
            # Low region with some variation
            coarse_f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.15

    # Interpolate to full resolution
    coarse_x = np.linspace(-1, 1, coarse_size)
    fine_x = np.linspace(-1, 1, n_steps)
    
    # Use spline interpolation for smooth transition
    from scipy.interpolate import interp1d
    interpolate_func = interp1d(coarse_x, coarse_f, kind='linear', fill_value='extrapolate')
    f = interpolate_func(fine_x)
    
    # Add Gaussian-shaped smoothing to reduce artifacts and add structure
    gaussian_width = 0.2 + np.random.random() * 0.15
    gaussian = np.exp(-0.5 * (fine_x / gaussian_width)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4
    
    # Add some additional structure
    n_peaks = 2 + np.random.randint(0, 3)
    for _ in range(n_peaks):
        peak_pos = np.random.randint(0, n_steps)
        peak_width = max(1, n_steps // 15 + np.random.randint(-2, 3))
        start = max(0, peak_pos - peak_width // 2)
        end = min(n_steps, peak_pos + peak_width // 2)
        peak_height = 0.3 + np.random.random() * 0.4
        f[start:end] = np.maximum(f[start:end], peak_height * 0.7 + np.random.random(end - start) * 0.3)
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def generate_pattern_based_initialization(n_steps: int) -> List[float]:
    """
    Generate a sophisticated initial configuration using multiple pattern combinations
    """
    f = np.zeros(n_steps)

    # Pattern 1: Main alternating structure with variation
    segment_size = max(1, n_steps // 10)
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region
            f[i:end_idx] = 0.6 + np.random.random(end_idx - i) * 0.3
        else:
            # Low region
            f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.1

    # Pattern 2: Add Gaussian envelope for smoothness
    x = np.linspace(-1, 1, n_steps)
    gaussian_width = 0.15 + np.random.random() * 0.2
    gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
    f = f * gaussian * 0.4 + gaussian * 0.6

    # Pattern 3: Add some peak structures for extra complexity
    n_peaks = 2 + np.random.randint(0, 3)
    for _ in range(n_peaks):
        peak_pos = np.random.randint(0, n_steps)
        peak_width = max(1, n_steps // 20 + np.random.randint(-2, 3))
        start = max(0, peak_pos - peak_width // 2)
        end = min(n_steps, peak_pos + peak_width // 2)
        peak_height = 0.3 + np.random.random() * 0.4
        f[start:end] = np.maximum(f[start:end], peak_height)

    # Ensure non-negativity
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def generate_diverse_initial_population(n_individuals: int, n_steps: int) -> List[List[float]]:
    """
    Generate diverse initial population for evolutionary algorithm with enhanced variety
    """
    population = []

    # Create various types of initial configurations
    for i in range(n_individuals):
        # Type 1: Alternating segments with smooth transitions
        if i % 5 == 0:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 10)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region
                    f[j:end_idx] = 0.8 + np.random.random(end_idx - j) * 0.15
                else:
                    # Low region
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15

            # Smooth with Gaussian
            x = np.linspace(-1, 1, n_steps)
            gaussian_width = 0.2 + np.random.random() * 0.15
            gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
            f = f * gaussian * 0.6 + gaussian * 0.4

            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

        # Type 2: Multi-peak distribution
        elif i % 5 == 1:
            f = np.ones(n_steps) * 0.1  # Base low values
            # Add multiple peaks at different positions
            n_peaks = 3 + np.random.randint(0, 3)
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, n_steps)
                peak_width = max(1, n_steps // 15 + np.random.randint(-2, 3))
                start = max(0, peak_pos - peak_width // 2)
                end = min(n_steps, peak_pos + peak_width // 2)
                f[start:end] = np.maximum(f[start:end], 0.7 + np.random.random(end - start) * 0.2)

            # Add smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.25)**2)
            f = f * gaussian * 0.5 + gaussian * 0.5

            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

        # Type 3: Gaussian-like distribution
        elif i % 5 == 2:
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.15 + np.random.random() * 0.2
            mu = np.random.random() * 0.3 - 0.15  # Centered around -0.15 to 0.15
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

        # Type 4: Uniform distribution with some structure  
        elif i % 5 == 3:
            f = np.random.random(n_steps)
            # Add some structure with clustering
            clusters = 3 + np.random.randint(0, 3)
            for _ in range(clusters):
                center = np.random.randint(0, n_steps)
                width = max(1, n_steps // 10 + np.random.randint(-2, 3))
                start = max(0, center - width // 2)
                end = min(n_steps, center + width // 2)
                f[start:end] = np.maximum(f[start:end], 0.5 + np.random.random(end - start) * 0.3)
            f = np.clip(f, 0, 1)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

        # Type 5: High-low alternating with enhanced transitions
        else:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 15)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region
                    f[j:end_idx] = 0.75 + np.random.random(end_idx - j) * 0.2
                else:
                    # Low region
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15

            # Apply more aggressive smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.3)**2)
            f = f * gaussian * 0.4 + gaussian * 0.6

            # Ensure non-negativity
            f = np.clip(f, 0, None)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            population.append(f.tolist())

    return population

def enhanced_evolutionary_optimization(max_generations: int = 30) -> List[float]:
    """
    Enhanced evolutionary algorithm with adaptive population sizing and diversity preservation
    """
    n_steps = 500

    # Initial population size
    pop_size = 12
    max_pop_size = 20
    min_pop_size = 6

    # Track convergence
    previous_best = -np.inf
    no_improvement_count = 0
    max_no_improvement = 8

    # Generate initial population
    population = generate_diverse_initial_population(pop_size, n_steps)

    best_solution = None
    best_c2 = -np.inf

    for generation in range(max_generations):
        # Evaluate fitness of current population
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)

            if c2 > best_c2:
                best_c2 = c2
                best_solution = individual.copy()

        # Check for convergence
        current_best = max(fitness_scores)
        if current_best > previous_best:
            previous_best = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Adaptively adjust population size based on convergence
        if no_improvement_count > max_no_improvement:
            if pop_size < max_pop_size:
                pop_size = min(pop_size + 1, max_pop_size)
            elif pop_size > min_pop_size:
                pop_size = max(pop_size - 1, min_pop_size)
            no_improvement_count = 0

        # Selection: Keep top half of individuals
        sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size//2]
        selected_population = [population[i] for i in sorted_indices]

        # Elitism: keep the best individual
        if best_solution is not None:
            selected_population.append(best_solution)

        # Generate offspring through crossover and mutation
        new_population = selected_population.copy()

        # Create new individuals through crossover and mutation
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]

            # Crossover (uniform)
            child = []
            for i in range(n_steps):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            # Mutation with adaptive rate
            mutation_rate = 0.15 * np.exp(-generation/max_generations)  # Decreasing rate
            mutation_intensity = 0.05 * (1 - generation/max_generations)  # Decreasing intensity

            for i in range(n_steps):
                if np.random.random() < mutation_rate:
                    delta = np.random.normal(0, mutation_intensity)
                    child[i] = max(0, child[i] + delta)

            # Normalize
            child_sum = sum(child)
            if child_sum > 0:
                child = [val / child_sum for val in child]

            new_population.append(child)

        # Trim to population size
        population = new_population[:pop_size]

    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def local_refinement_optimization(initial_f: List[float], max_iter: int = 20) -> List[float]:
    """
    Local refinement using gradient-like approach to fine-tune the solution
    """
    f = np.array(initial_f)
    n_steps = len(f)

    # Simple gradient-like approach with small perturbations
    for iteration in range(max_iter):
        current_c2 = evaluate_c2(f.tolist())

        # Try small perturbations
        best_f = f.copy()
        best_c2 = current_c2

        for _ in range(30):  # Try fewer random perturbations for speed
            perturbed_f = f.copy()
            # Apply small random changes with controlled variance
            idx = np.random.randint(0, n_steps)
            delta = np.random.normal(0, 0.01)
            perturbed_f[idx] = max(0, perturbed_f[idx] + delta)

            # Normalize
            if np.sum(perturbed_f) > 0:
                perturbed_f = perturbed_f / np.sum(perturbed_f)

            new_c2 = evaluate_c2(perturbed_f.tolist())

            if new_c2 > best_c2:
                best_c2 = new_c2
                best_f = perturbed_f

        f = best_f

        # Early stopping if improvement is minimal
        if abs(best_c2 - current_c2) < 1e-6:
            break

    return f.tolist()

def hybrid_optimization_approach() -> List[float]:
    """
    Hybrid optimization combining multiple strategies for better results
    """
    n_steps = 500

    # Strategy 1: Generate multiple diverse initial solutions with better patterns
    initial_solutions = []

    # Multiple initialization strategies
    for i in range(4):
        if i == 0:
            # Multi-scale initialization for broad exploration
            initial_solutions.append(generate_multi_scale_initialization(n_steps))
        elif i == 1:
            # Pattern-based initialization
            initial_solutions.append(generate_pattern_based_initialization(n_steps))
        elif i == 2:
            # Diverse population initialization
            population = generate_diverse_initial_population(1, n_steps)
            initial_solutions.append(population[0])
        else:
            # Random initialization
            f = np.random.random(n_steps)
            f = f / np.sum(f) if np.sum(f) > 0 else f
            initial_solutions.append(f.tolist())

    # Evaluate all initial solutions
    best_c2 = -np.inf
    best_solution = None

    for sol in initial_solutions:
        c2 = evaluate_c2(sol)
        if c2 > best_c2:
            best_c2 = c2
            best_solution = sol

    # Strategy 2: Enhanced evolutionary optimization
    evolved_solution = enhanced_evolutionary_optimization(30)
    evolved_c2 = evaluate_c2(evolved_solution)

    if evolved_c2 > best_c2:
        best_c2 = evolved_c2
        best_solution = evolved_solution

    # Strategy 3: Local refinement of the best solution
    if best_solution is not None:
        refined_solution = local_refinement_optimization(best_solution, 15)
        refined_c2 = evaluate_c2(refined_solution)

        if refined_c2 > best_c2:
            best_c2 = refined_c2
            best_solution = refined_solution

    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def advanced_gradient_refinement(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Advanced gradient-based refinement with adaptive learning rates and bounds
    """
    try:
        # Use scipy's L-BFGS-B optimization
        def objective(x):
            return -evaluate_c2(x.tolist())
        
        # Define bounds for each parameter (step height)
        bounds = [(0, 1.0) for _ in range(len(initial_f))]
        
        # Run L-BFGS-B optimization
        result = minimize(
            objective,
            initial_f,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'gtol': 1e-6},
            callback=None
        )
        
        if result.success:
            optimized_f = np.maximum(result.x, 0)
            if np.sum(optimized_f) > 0:
                optimized_f = optimized_f / np.sum(optimized_f)
            return optimized_f.tolist()
    except Exception as e:
        pass  # Fall back to original if gradient optimization fails
        
    return initial_f

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using hybrid optimization
    """
    try:
        # Use hybrid optimization approach
        final_solution = hybrid_optimization_approach()
        
        # Apply advanced gradient refinement
        refined_solution = advanced_gradient_refinement(final_solution, max_iter=20)
        
        # Final validation and normalization
        if not refined_solution:
            refined_solution = [1.0/500] * 500
            
        # Ensure non-negativity and normalization
        refined_solution = np.array(refined_solution)
        refined_solution = np.clip(refined_solution, 0, None)
        if np.sum(refined_solution) > 0:
            refined_solution = refined_solution / np.sum(refined_solution)
            
        return refined_solution.tolist()
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        return [1.0/500] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")