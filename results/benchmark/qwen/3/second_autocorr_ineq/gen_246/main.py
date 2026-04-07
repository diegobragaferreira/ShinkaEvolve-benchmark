# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time
import numba
from numba import jit, prange

@jit(nopython=True, parallel=False)
def compute_convolution_norms_fast(f_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fast computation of autoconvolution norms using numba-compiled loop
    """
    n = len(f_values)
    # Pre-allocate convolution result
    g = np.zeros(2*n - 1)

    # Manual convolution computation for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Extract valid convolution (center part)
    center = len(g) // 2
    g_valid = g[center - n + 1:center + n]

    # Compute norms
    norm_l2_sq = 0.0
    norm_l1 = 0.0
    norm_linf = 0.0

    # Use piecewise linear integration for L2 norm
    dx = 1.0 / len(g_valid)  # Step size
    for i in range(len(g_valid)-1):
        norm_l2_sq += (dx/3) * (g_valid[i]**2 + g_valid[i]*g_valid[i+1] + g_valid[i+1]**2)

    for val in g_valid:
        norm_l1 += abs(val) * dx
        if abs(val) > norm_linf:
            norm_linf = abs(val)

    return norm_l2_sq, norm_l1, norm_linf

@jit(nopython=True)
def calculate_c2_fast(f_values: np.ndarray) -> float:
    """Fast calculation of C2 with numerical stability"""
    norm_l2_sq, norm_l1, norm_linf = compute_convolution_norms_fast(f_values)

    # Avoid division by zero
    if norm_l1 <= 1e-12 or norm_linf <= 1e-12:
        return 0.0

    c2 = norm_l2_sq / (norm_l1 * norm_linf)
    return c2

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array for efficient computation
    f = np.array(f_values)

    # Compute autoconvolution g = f * f using fast convolution
    g = signal.convolve(f, f, mode='full')

    # Only keep the middle portion corresponding to valid convolution
    center_idx = len(g) // 2
    half_len = len(f)
    g = g[center_idx - half_len + 1:center_idx + half_len]

    # Compute norms using trapezoidal rule
    g_squared = g * g
    dx = 1.0 / len(g)  # Step size for integration
    norm_l2_sq = 0.0
    for i in range(len(g)-1):
        norm_l2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)
    
    # L1 norm
    norm_l1 = np.sum(np.abs(g)) * dx
    
    # L-infinity norm
    norm_linf = np.max(np.abs(g))

    return norm_l2_sq, norm_l1, norm_linf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    try:
        # Use fast numba version for speed during optimization
        if len(f_values) < 500:
            f_array = np.array(f_values)
            return calculate_c2_fast(f_array)
        else:
            # Use more accurate scipy version for complex cases
            norm_l2_sq, norm_l1, norm_linf = compute_autoconvolution_norms(f_values)
            if norm_l1 <= 1e-12 or norm_linf <= 1e-12:
                return 0.0
            c2 = norm_l2_sq / (norm_l1 * norm_linf)
            return c2
    except Exception:
        return 0.0

def construct_multiscale_gaussian_initial_function(n_steps: int = 200) -> List[float]:
    """
    Construct step function using multi-scale Gaussian patterns.
    Creates a more principled geometric foundation that systematically varies scales.
    """
    f_values = np.zeros(n_steps)

    # Create multiple Gaussian bumps with varying scales and positions
    num_bumps = min(20, n_steps // 8)  # Increased number of bumps for better structure
    bump_positions = np.linspace(0.1, 0.9, num_bumps)  # Spread bumps across domain
    bump_scales = np.logspace(-1.5, 0, num_bumps, base=10)  # Varying scales (from coarse to fine)
    bump_heights = np.logspace(0, 1.5, num_bumps, base=10)  # Varying heights

    # Normalize heights to avoid extreme values
    bump_heights = bump_heights / np.sum(bump_heights) * 5.0

    # Create each Gaussian bump
    x = np.linspace(0, 1, n_steps)
    for i in range(num_bumps):
        # Position bump according to logarithmic spacing
        position = bump_positions[i]
        scale = bump_scales[i] * 0.1  # Scale factor to control spread
        height = bump_heights[i]

        # Create Gaussian bump
        bump = height * np.exp(-((x - position)**2) / (2 * scale**2))
        f_values += bump

    # Add some additional smooth variation
    smooth_variation = 0.3 * np.sin(4 * np.pi * x) + 0.7
    f_values = f_values * smooth_variation

    # Ensure non-negativity and normalize
    f_values = np.clip(f_values, 0, None)

    # Normalize to reasonable range
    total = np.sum(f_values)
    if total > 0:
        f_values = f_values / total * 10

    # Add some controlled noise for diversification
    noise = np.random.normal(0, 0.01, n_steps)
    f_values = np.clip(f_values + noise, 0, None)

    return f_values.tolist()

def construct_structured_initial_function(n_steps: int = 200) -> List[float]:
    """
    Construct a more structured initial function with explicit geometric properties
    """
    # Create a combination of sinusoidal and polynomial patterns
    x = np.linspace(0, 1, n_steps)
    
    # Base pattern with peaks
    pattern1 = 0.5 + 0.5 * np.sin(4 * np.pi * x)
    
    # Add additional structure
    pattern2 = 0.3 * np.sin(8 * np.pi * x) + 0.7
    
    # Combine patterns
    combined = pattern1 * pattern2
    
    # Add central peak
    center = n_steps // 2
    width = n_steps // 6
    central_peak = np.exp(-((np.arange(n_steps) - center)**2) / (2 * (width/2)**2))
    combined = combined + 0.2 * central_peak
    
    # Ensure non-negativity
    combined = np.clip(combined, 0, None)
    
    # Normalize
    total = np.sum(combined)
    if total > 0:
        combined = combined / total * 10
    
    # Add noise for diversity
    noise = np.random.normal(0, 0.01, n_steps)
    combined = np.clip(combined + noise, 0, None)
    
    return combined.tolist()

def tournament_selection(population: List[List[float]], fitness: List[float],
                         k: int = 3) -> List[float]:
    """Select individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda i: fitness[i])
    return population[best_idx]

def uniform_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform uniform crossover between two parents"""
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

def mutate_individual(individual: List[float], mutation_rate: float) -> List[float]:
    """Mutate an individual with given probability"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise with adaptive strength
            mutation_strength = 0.1 * mutated[i] if mutated[i] > 0 else 0.1
            mutated[i] = max(0, mutated[i] + random.gauss(0, mutation_strength))
    return mutated

def adaptive_evolutionary_optimization(n_steps: int = 1000):
    """Main optimization routine using adaptive evolutionary strategy"""

    # Generate diverse initial population with more variety
    pop_size = 100  # Increased population size for better exploration
    population = []
    
    # Use multi-scale Gaussian function for majority of population
    for _ in range(int(pop_size * 0.6)):
        individual = construct_multiscale_gaussian_initial_function(n_steps)
        # Ensure non-negativity and reasonable scaling
        individual = [max(0, x) for x in individual]
        total = sum(individual)
        if total > 0:
            individual = [x / total * 10 for x in individual]
        population.append(individual)

    # Use structured initialization for some diversity
    for _ in range(int(pop_size * 0.2)):
        individual = construct_structured_initial_function(n_steps)
        # Ensure non-negativity and reasonable scaling
        individual = [max(0, x) for x in individual]
        total = sum(individual)
        if total > 0:
            individual = [x / total * 10 for x in individual]
        population.append(individual)

    # Add some diversity with random initialization
    for _ in range(pop_size - len(population)):
        individual = np.random.random(n_steps).tolist()
        # Ensure non-negativity and reasonable scaling
        individual = [max(0, x) for x in individual]
        total = sum(individual)
        if total > 0:
            individual = [x / total * 10 for x in individual]
        population.append(individual)

    # Track best solution
    best_individual = None
    best_c2 = -1
    max_generations = 50  # Increased generations for better optimization
    
    # Adaptive parameters
    mutation_rate = 0.25  # Slightly reduced mutation rate for better convergence
    crossover_rate = 0.8
    elite_count = 8  # More elites for better preservation

    # Early stopping variables
    best_fitness_history = []
    early_stop_window = 15
    early_stop_threshold = 1e-6

    # Evolution loop
    for generation in range(max_generations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            c2 = calculate_c2(individual)
            fitness_scores.append(c2)
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Store best fitness for early stopping
        best_fitness_history.append(best_c2)
        if len(best_fitness_history) > early_stop_window:
            best_fitness_history.pop(0)
            
        # Check for early stopping
        if len(best_fitness_history) >= early_stop_window:
            improvement = abs(best_fitness_history[-1] - best_fitness_history[0])
            if improvement < early_stop_threshold:
                print(f"Early stopping at generation {generation}")
                break

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(elite_count):
            new_population.append(sorted_population[i].copy())

        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents
            parent1 = tournament_selection(sorted_population, sorted_fitness, 3)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 3)

            # Crossover
            if np.random.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            mutate_individual(child1, mutation_rate)
            mutate_individual(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

        # Adapt mutation rate: decrease over time to fine-tune
        mutation_rate = max(0.05, mutation_rate * 0.95)

    return best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Try adaptive evolutionary optimization first
    start_time = time.time()
    try:
        result = adaptive_evolutionary_optimization(1000)
        elapsed = time.time() - start_time
        if elapsed < 85:  # Leave some margin for final calculations
            return result
    except Exception as e:
        print(f"Evolutionary optimization failed: {e}")
        pass

    # Final fallback to simpler approach if optimization fails or times out
    return [random.uniform(0, 1) for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")