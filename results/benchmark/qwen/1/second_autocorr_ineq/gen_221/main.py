# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from typing import List
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# Import JAX for automatic differentiation
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp

@njit
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

# JAX-compiled autoconvolution computation
@jit
def compute_autoconvolution_jax(f_values):
    """JAX-compiled autoconvolution computation"""
    f = jnp.array(f_values)
    # Compute autoconvolution using JAX's convolution
    g = jnp.convolve(f, f, mode='full')
    return g

@jit
def compute_autoconvolution_norms_jax(f_values):
    """JAX-compiled computation of autoconvolution norms"""
    g = compute_autoconvolution_jax(f_values)

    # ||g||₂² using proper piecewise integration
    # For consecutive pairs (y1, y2) with unit spacing: integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    g_squares = g[:-1]**2 + g[:-1]*g[1:] + g[1:]**2
    norm_g_2_squared = jnp.sum(g_squares) / 3.0

    # ||g||₁ = sum(|g[i]|)
    norm_g_1 = jnp.sum(jnp.abs(g))

    # ||g||∞ = max(|g[i]|)
    norm_g_inf = jnp.max(jnp.abs(g))

    return norm_g_2_squared, norm_g_1, norm_g_inf

@jit
def evaluate_c2_jax(f_values):
    """JAX-compiled evaluation of C₂"""
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_jax(f_values)

    # Avoid division by zero
    epsilon = 1e-15
    norm_g_1 = jnp.where(norm_g_1 <= epsilon, epsilon, norm_g_1)
    norm_g_inf = jnp.where(norm_g_inf <= epsilon, epsilon, norm_g_inf)

    c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
    return c2

# Gradient-based optimization setup
@jit
def c2_gradient_jax(f_values):
    """Compute gradient of C2 with respect to f_values using JAX automatic differentiation"""
    # Use automatic differentiation
    return grad(evaluate_c2_jax)(f_values)

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

def generate_pattern_based_initialization(n_steps: int) -> List[float]:
    """
    Generate a sophisticated initial configuration using multiple pattern combinations
    """
    f = np.zeros(n_steps)

    # Pattern 1: Main alternating structure
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

def enhanced_evolutionary_optimization(max_generations: int = 50) -> List[float]:
    """
    Enhanced evolutionary algorithm with adaptive population sizing and diversity preservation
    """
    n_steps = 500

    # Initial population size
    pop_size = 12
    max_pop_size = 20
    min_pop_size = 8

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

        # More intelligent adaptive population sizing
        if no_improvement_count > max_no_improvement:
            if pop_size < max_pop_size:
                pop_size = min(pop_size + 1, max_pop_size)
            elif pop_size > min_pop_size:
                pop_size = max(pop_size - 1, min_pop_size)
            no_improvement_count = 0

        # Selection: Keep top third of individuals for better exploration
        sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size//3]
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

            # Crossover (uniform with probabilistic bias towards better parents)
            child = []
            for i in range(n_steps):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            # Mutation with adaptive rate and stronger intensification near later generations
            mutation_rate = 0.2 * np.exp(-generation/max_generations)  # Increased base rate
            mutation_intensity = 0.08 * (1 - generation/max_generations)  # Increased intensity

            for i in range(n_steps):
                if np.random.random() < mutation_rate:
                    # Adjust intensity based on generation progress
                    delta = np.random.normal(0, mutation_intensity * (1 + generation/max_generations))
                    child[i] = max(0, child[i] + delta)

            # Normalize
            child_sum = sum(child)
            if child_sum > 0:
                child = [val / child_sum for val in child]

            new_population.append(child)

        # Trim to population size
        population = new_population[:pop_size]

    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def jax_gradient_refinement(initial_f: List[float], max_iter: int = 50) -> List[float]:
    """
    Enhanced local refinement using JAX-based automatic differentiation for precise gradient computation
    with adaptive learning rate and momentum for improved convergence
    """
    f_start = np.array(initial_f)
    n_steps = len(f_start)

    # Use JAX for gradient computation and optimization
    def objective_jax(x):
        return -evaluate_c2_jax(x.tolist())

    # Compute gradient using JAX automatic differentiation
    def compute_jax_gradient(x):
        return -np.array(c2_gradient_jax(x))

    # Perform gradient descent with JAX - enhanced with adaptive learning rate and momentum
    f_current = f_start.astype(float)

    # Momentum term for stable updates
    velocity = np.zeros_like(f_current)
    momentum_coeff = 0.9  # Momentum coefficient

    # Initial learning rate and decay factor
    initial_lr = 0.02
    lr_decay = 0.95

    for iteration in range(max_iter):
        current_c2 = evaluate_c2(f_current.tolist())

        # Compute precise gradient using JAX
        grad_vals = compute_jax_gradient(f_current)

        # Adaptive learning rate with exponential decay
        learning_rate = initial_lr * (lr_decay ** iteration)

        # Update velocity with momentum
        velocity = momentum_coeff * velocity + learning_rate * grad_vals

        # Apply update with momentum
        f_new = f_current - velocity

        # Ensure non-negativity
        f_new = np.maximum(f_new, 0)

        # Normalize
        if np.sum(f_new) > 0:
            f_new = f_new / np.sum(f_new)

        # Check for convergence with more stringent criteria
        new_c2 = evaluate_c2(f_new.tolist())

        # Convergence check - if improvement is minimal or negative, stop
        if new_c2 <= current_c2 + 1e-8:  # Allow small numerical improvements
            break

        f_current = f_new

    return f_current.tolist()

def hybrid_optimization_approach() -> List[float]:
    """
    Hybrid optimization combining multiple strategies for better results
    """
    n_steps = 500

    # Strategy 1: Generate multiple diverse initial solutions with enhanced variety
    initial_solutions = []

    # Multiple initialization strategies - increased diversity
    for i in range(6):
        if i == 0:
            # Pattern-based initialization
            initial_solutions.append(generate_pattern_based_initialization(n_steps))
        elif i == 1:
            # Diverse population initialization
            population = generate_diverse_initial_population(1, n_steps)
            initial_solutions.append(population[0])
        elif i == 2:
            # Structured initialization with peaks
            f = np.zeros(n_steps)
            # Create more structured pattern with defined peaks
            n_peaks = 3 + np.random.randint(1, 3)
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, n_steps)
                peak_width = max(1, n_steps // 20 + np.random.randint(-2, 3))
                start = max(0, peak_pos - peak_width // 2)
                end = min(n_steps, peak_pos + peak_width // 2)
                peak_height = 0.4 + np.random.random() * 0.4
                f[start:end] = np.maximum(f[start:end], peak_height)
            f = np.clip(f, 0, None)
            if np.sum(f) > 0:
                f = f / np.sum(f)
            initial_solutions.append(f.tolist())
        else:
            # Random initialization with variation
            f = np.random.random(n_steps)
            # Add some structure to make it less random
            f = f * (0.7 + np.random.random() * 0.3)
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

    # Strategy 2: Run multiple evolutionary optimizations with different settings
    # This increases chances of finding better solutions
    best_evolved_solution = None
    best_evolved_c2 = -np.inf

    # Run multiple evolutionary runs with different parameters
    for run in range(3):
        # Vary the number of generations for each run
        generations = 30 + run * 10
        evolved_solution = enhanced_evolutionary_optimization(generations)
        evolved_c2 = evaluate_c2(evolved_solution)

        if evolved_c2 > best_evolved_c2:
            best_evolved_c2 = evolved_c2
            best_evolved_solution = evolved_solution

    if best_evolved_c2 > best_c2:
        best_c2 = best_evolved_c2
        best_solution = best_evolved_solution

    # Strategy 3: JAX-based gradient refinement with enhanced precision
    if best_solution is not None:
        # Use the more accurate JAX-based refinement
        refined_solution = jax_gradient_refinement(best_solution, max_iter=20)
        refined_c2 = evaluate_c2(refined_solution)

        if refined_c2 > best_c2:
            best_c2 = refined_c2
            best_solution = refined_solution

    # Final validation check
    if best_solution is None:
        # Return a default solution rather than None
        return [1.0/n_steps] * n_steps

    return best_solution

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using hybrid optimization
    """
    try:
        # Use hybrid optimization approach
        final_solution = hybrid_optimization_approach()

        return final_solution

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")