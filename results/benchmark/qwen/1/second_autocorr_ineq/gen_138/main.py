# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
from numba import jit
import warnings
warnings.filterwarnings('ignore')

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

    # Compute the norms with proper piecewise integration for ||g||₂²
    # ||g||₂² = sum of (g[i]^2 + g[i]*g[i+1] + g[i+1]^2)/3
    # This is a more accurate trapezoidal-like integration for quadratic function
    norm_g_2_squared = 0.0

    # Trapezoidal integration for g^2: for interval [x_i, x_{i+1}] with values g_i, g_{i+1}
    # integral of g^2 dx ≈ (Δx/3)(g_i^2 + g_i*g_{i+1} + g_{i+1}^2) where Δx = 1
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

        # Avoid division by zero
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def generate_harmonic_initialization(n_steps: int) -> List[float]:
    """
    Generate initial configuration using harmonic pattern generation
    Creates functions with sinusoidal components that are likely to produce favorable autoconvolutions
    """
    # Create a base function with harmonic components
    f = np.zeros(n_steps)
    
    # Use a combination of sinusoidal components for rich spectral content
    x = np.linspace(-1, 1, n_steps)
    
    # Main harmonic component (fundamental frequency)
    freq1 = 2.0 + np.random.random() * 3.0
    f += np.sin(freq1 * np.pi * x) * 0.3
    
    # Second harmonic component
    freq2 = 4.0 + np.random.random() * 4.0
    f += np.sin(freq2 * np.pi * x) * 0.2
    
    # Third harmonic component
    freq3 = 6.0 + np.random.random() * 3.0
    f += np.cos(freq3 * np.pi * x) * 0.15
    
    # Add a main envelope for structure
    envelope = np.exp(-0.5 * (x / 0.3)**2)
    f = f * envelope * 0.5 + envelope * 0.5
    
    # Add random variations to break symmetry
    f += np.random.normal(0, 0.02, n_steps)
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def generate_multiresolution_initialization(n_steps: int, coarse_resolution: int = 200) -> List[float]:
    """
    Generate initial configuration using multi-resolution approach
    """
    # Start with coarse resolution to understand the landscape
    coarse_f = np.zeros(coarse_resolution)
    
    # Create a pattern with alternating high/low sections
    segment_size = max(1, coarse_resolution // 8)
    for i in range(0, coarse_resolution, segment_size):
        end_idx = min(i + segment_size, coarse_resolution)
        if (i // segment_size) % 2 == 0:
            # High region
            coarse_f[i:end_idx] = 0.6 + np.random.random(end_idx - i) * 0.3
        else:
            # Low region
            coarse_f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.15
    
    # Interpolate to full resolution
    coarse_x = np.linspace(-1, 1, coarse_resolution)
    fine_x = np.linspace(-1, 1, n_steps)
    
    # Use spline interpolation for smooth transition
    from scipy.interpolate import interp1d
    interpolate_func = interp1d(coarse_x, coarse_f, kind='linear', fill_value='extrapolate')
    f = interpolate_func(fine_x)
    
    # Add some smoothing
    gaussian_width = 0.1 + np.random.random() * 0.2
    gaussian = np.exp(-0.5 * (fine_x / gaussian_width)**2)
    f = f * gaussian * 0.4 + gaussian * 0.6
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def generate_alternating_initialization(n_steps: int) -> List[float]:
    """
    Generate alternating high/low segments with structured randomness
    """
    f = np.zeros(n_steps)
    
    # Create a base alternating pattern with more varied segment sizes
    segment_sizes = [max(1, n_steps // 12), max(1, n_steps // 8), max(1, n_steps // 10)]
    segment_size = segment_sizes[np.random.choice(len(segment_sizes))]
    
    # Alternate between high and low values to create interesting convolution behavior
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region with variation
            amplitude = 0.7 + np.random.random() * 0.25
            f[i:end_idx] = amplitude + np.random.random(end_idx - i) * 0.1
        else:
            # Low region with variation
            amplitude = 0.1 + np.random.random() * 0.15
            f[i:end_idx] = amplitude + np.random.random(end_idx - i) * 0.1
    
    # Add Gaussian-like structure for smoothness and structure preservation
    x = np.linspace(-1, 1, n_steps)
    gaussian_width = 0.2 + np.random.random() * 0.2
    gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4
    
    # Add some noise for diversity
    noise_level = 0.01 + np.random.random() * 0.02
    noise = np.random.normal(0, noise_level, n_steps)
    f = f + noise
    
    # Ensure non-negativity
    f = np.clip(f, 0, None)
    
    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def adaptive_evolutionary_search(n_steps: int = 500, max_iterations: int = 50) -> List[float]:
    """
    Custom evolutionary algorithm with adaptive parameters and enhanced operators
    """
    # Initialize with diverse patterns
    population = []
    pop_size = 12  # Increased base population size
    
    # Generate diverse initial population
    for i in range(pop_size):
        if i % 4 == 0:
            f = generate_harmonic_initialization(n_steps)
        elif i % 4 == 1:
            f = generate_multiresolution_initialization(n_steps)
        elif i % 4 == 2:
            f = generate_alternating_initialization(n_steps)
        else:
            # Random initialization
            f = np.random.random(n_steps)
            f = np.clip(f, 0, 1)
            f = f / np.sum(f)
            f = f.tolist()
        population.append(f)
    
    best_solution = None
    best_c2 = -np.inf
    no_improvement_count = 0
    max_no_improvement = 10
    
    # Evolutionary search loop
    for iteration in range(max_iterations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = individual.copy()
        
        # Selection - keep top 50% (but at least 3)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_count = max(3, pop_size // 2)
        selected_indices = sorted_indices[:selected_count]
        selected_population = [population[i] for i in selected_indices]
        
        # Create offspring through crossover and mutation
        new_population = selected_population.copy()
        
        # Elitism: keep the best individual
        if best_solution is not None:
            new_population.append(best_solution)
        
        # Adaptive population size management
        current_best = max(fitness_scores)
        if current_best > best_c2:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Adjust population size based on convergence
        if no_improvement_count > max_no_improvement:
            if pop_size < 20:  # Max population size
                pop_size += 1
            elif pop_size > 8:  # Min population size
                pop_size -= 1
            no_improvement_count = 0
        
        # Generate new individuals through crossover and mutation
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]
            
            # Crossover (uniform with some structure preservation)
            child = []
            for i in range(n_steps):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Mutation with adaptive rate
            mutation_rate = 0.15 * np.exp(-iteration/max_iterations)  # Decreasing rate
            for i in range(n_steps):
                if np.random.random() < mutation_rate:
                    # Add small random perturbation with variance that decreases
                    variance = 0.05 * (1 - iteration/max_iterations)
                    delta = np.random.normal(0, variance)
                    child[i] = max(0, child[i] + delta)
            
            # Normalize
            child_sum = sum(child)
            if child_sum > 0:
                child = [val / child_sum for val in child]
            
            new_population.append(child)
        
        # Trim to population size
        population = new_population[:pop_size]
    
    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def adaptive_local_refinement(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Enhanced local refinement with multiple strategies
    """
    f = np.array(initial_f)
    n_steps = len(f)
    
    # Multi-stage local refinement
    for stage in range(3):
        # Stage 1: Gradient-free local search with random perturbations
        if stage == 0:
            for iteration in range(max_iter // 3):
                current_c2 = evaluate_c2(f.tolist())
                
                # Try small perturbations
                best_f = f.copy()
                best_c2 = current_c2
                
                # Try multiple random perturbations
                for _ in range(30):
                    perturbed_f = f.copy()
                    # Apply small random changes
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
                if abs(best_c2 - current_c2) < 1e-7:
                    break
        
        # Stage 2: Gradient-based local optimization
        elif stage == 1:
            try:
                def local_objective(f_vals):
                    return -evaluate_c2(f_vals.tolist())
                
                bounds_local = [(0, 1.0) for _ in range(n_steps)]
                local_result = minimize(
                    local_objective,
                    f,
                    method='L-BFGS-B',
                    bounds=bounds_local,
                    options={'maxiter': max_iter // 3}
                )
                
                if local_result.success:
                    refined_f = np.maximum(local_result.x, 0)
                    if np.sum(refined_f) > 0:
                        refined_f = refined_f / np.sum(refined_f)
                    f = refined_f
            except:
                pass
                
        # Stage 3: Stochastic perturbation to escape local optima
        elif stage == 2:
            # Add small Gaussian noise to escape local optima
            noise = np.random.normal(0, 0.005, n_steps)
            f = f + noise
            f = np.clip(f, 0, None)
            if np.sum(f) > 0:
                f = f / np.sum(f)
    
    return f.tolist()

def improved_construct_function() -> list[float]:
    """
    Improved function construction with multiple strategies and adaptive optimization
    """
    n_steps = 500
    
    try:
        # Strategy 1: Multi-start with different initialization strategies
        best_solution = None
        best_c2 = -np.inf
        
        # Try multiple initialization strategies
        strategies = [
            ("harmonic", lambda: generate_harmonic_initialization(n_steps)),
            ("multires", lambda: generate_multiresolution_initialization(n_steps)),
            ("alternating", lambda: generate_alternating_initialization(n_steps)),
            ("random", lambda: np.random.random(n_steps).tolist())
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                # Generate initial solution
                initial_f = strategy_func()
                
                # Run evolutionary optimization
                optimized_f = adaptive_evolutionary_search(n_steps, max_iterations=35)
                c2_optimized = evaluate_c2(optimized_f)
                
                if c2_optimized > best_c2:
                    best_c2 = c2_optimized
                    best_solution = optimized_f
                    
            except Exception as e:
                continue  # Skip strategy if it fails
        
        # Strategy 2: Refine the best solution found so far
        if best_solution is not None:
            # Apply adaptive local refinement
            refined_solution = adaptive_local_refinement(best_solution, max_iter=25)
            refined_c2 = evaluate_c2(refined_solution)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_solution
        
        # If still no solution found, fall back to a simple initialization
        if best_solution is None:
            best_solution = [1.0/n_steps] * n_steps
            
        return best_solution
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")