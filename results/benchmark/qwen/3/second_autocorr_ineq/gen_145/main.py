# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import math
from numba import njit
from sklearn.cluster import KMeans
import random
import time

@njit
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Fast computation of autoconvolution norms using numba-compiled loop with hybrid approach
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # For small arrays, use direct convolution - for larger arrays, consider FFT
    if n_steps < 100:
        # Manual convolution computation for speed
        g = np.zeros(2*n_steps - 1)
        for i in range(n_steps):
            for j in range(n_steps):
                g[i + j] += f_values[i] * f_values[j]

        # Extract valid convolution (center part)
        center = len(g) // 2
        g_valid = g[center - n_steps + 1:center + n_steps]
    else:
        # For larger arrays, use FFT-based approach
        padded_size = 2 * n_steps - 1
        f_padded = np.zeros(padded_size)
        f_padded[:n_steps] = f_values
        
        # FFT-based convolution: g = f * f
        f_fft = np.fft.fft(f_padded)
        g_fft = f_fft * f_fft.conj()  # Pointwise multiplication in frequency domain
        g = np.fft.ifft(g_fft).real  # Inverse FFT to get convolution result
        
        # Extract the valid convolution part (center portion)
        g = g[n_steps-1:2*n_steps-1]

    # Compute norms using piecewise linear integration
    # For ||g||₂²: use trapezoidal-like formula
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
    """Fast calculation of C2 with numerical stability"""
    g2_sq, g1, ginf = compute_convolution_norms_numba(f_values)

    # Avoid division by zero
    if g1 <= 1e-12 or ginf <= 1e-12:
        return 0.0

    c2 = g2_sq / (g1 * ginf)
    return c2

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
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

    # Compute norms - using trapezoidal method for consistency with numba version
    g_squared = g * g
    dx = 1.0 / len(g)  # Step size for integration
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)
    
    # L1 norm
    g1 = np.sum(np.abs(g)) * dx
    
    # L-infinity norm
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def calculate_c2(f_values: list[float]) -> float:
    """Calculate C2 value for given step function"""
    try:
        # Use fast numba version for speed during optimization
        if len(f_values) < 1000:
            return compute_c2_numba(f_values)
        else:
            # Use more accurate scipy version for complex cases
            norm_l2_sq, norm_l1, norm_linf = compute_autoconvolution_norms(f_values)
            if norm_l1 <= 1e-12 or norm_linf <= 1e-12:
                return 0.0
            c2 = norm_l2_sq / (norm_l1 * norm_linf)
            return c2
    except Exception:
        return 0.0

def construct_geometric_initial_function(n_steps: int = 200) -> list[float]:
    """
    Construct step function using geometric patterns and Chebyshev nodes
    """
    f_values = np.zeros(n_steps)

    # Base pattern: smooth transition with some oscillation
    x = np.linspace(0, 1, n_steps)

    # Create a multi-scale pattern that helps avoid extreme peaks
    scale1 = 0.3 * np.sin(2 * np.pi * x) + 0.7
    scale2 = 0.2 * np.sin(4 * np.pi * x) + 0.8
    scale3 = 0.1 * np.sin(8 * np.pi * x) + 0.9

    # Combine scales with decreasing influence
    combined = 0.5 * scale1 + 0.3 * scale2 + 0.2 * scale3

    # Normalize to reasonable values
    combined = np.clip(combined, 0, 1)

    # Add some structured variation to make it less regular
    center = n_steps // 2
    width = n_steps // 4
    bump = np.exp(-((np.arange(n_steps) - center)**2) / (2 * (width/3)**2))
    bump = bump / np.max(bump) * 0.3
    combined = np.clip(combined + bump, 0, 1)

    # Add some controlled randomness
    noise = np.random.normal(0, 0.02, n_steps)
    combined = np.clip(combined + noise, 0, 1)

    return combined.tolist()

def smart_step_function_constructor(n_steps: int = 200) -> list[float]:
    """
    Construct step function using Chebyshev nodes for smoother convolution
    This creates a more favorable convolution structure
    """
    # Generate Chebyshev nodes for better distribution
    theta = np.pi * np.arange(1, n_steps + 1) / (n_steps + 1)
    cheb_nodes = np.cos(theta)
    
    # Map to [-1/4, 1/4] interval
    scaled_nodes = cheb_nodes * 0.125
    
    # Create step function - use a smooth profile that will produce good convolution properties
    # We'll use a triangular shape centered at 0 with peaks at the Chebyshev nodes
    # But since the nodes are positions, we create a step function with heights based on a smooth function
    base_heights = 0.5 + 0.5 * np.sin(2 * np.pi * scaled_nodes)  # Oscillating pattern
    # Apply a smooth weighting function that gives higher weights to central values
    central_weight = 1.0 + 0.5 * np.exp(-10 * scaled_nodes**2)  
    heights = base_heights * central_weight
    
    # Ensure non-negativity and normalize
    heights = np.maximum(heights, 0)
    # Normalize to prevent extreme values that might cause numerical issues
    heights = heights / (np.sum(heights) + 1e-10) * 10
    
    return heights.tolist()

def tournament_selection(population: list[list[float]], fitness: list[float], 
                         k: int = 3) -> list[float]:
    """Select individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda i: fitness[i])
    return population[best_idx]

def uniform_crossover(parent1: list[float], parent2: list[float]) -> list[float]:
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

def mutate_individual(individual: list[float], mutation_rate: float) -> list[float]:
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
    
    # Generate diverse initial population with multiple strategies
    pop_size = 50
    population = []
    
    # Use a mix of strategies for better exploration
    strategies = [
        ('geometric', lambda n: construct_geometric_initial_function(n)),
        ('chebyshev', lambda n: smart_step_function_constructor(n)),
        ('uniform', lambda n: np.random.random(n).tolist()),
        ('gaussian', lambda n: [max(0, random.gauss(0.5, 0.3)) for _ in range(n)])
    ]
    
    for i in range(pop_size):
        strategy_name, strategy_func = random.choice(strategies)
        individual = strategy_func(n_steps)
        
        # Ensure non-negativity and reasonable scaling
        individual = [max(0, x) for x in individual]
        # Normalize to avoid extremely large peaks
        total = sum(individual)
        if total > 0:
            individual = [x / total * 10 for x in individual]

        population.append(individual)

    # Track best solution
    best_individual = None
    best_c2 = -1
    max_generations = 25  # Limit to keep within time constraints

    # Adaptive parameters
    mutation_rate = 0.3
    crossover_rate = 0.8
    elite_count = 5

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

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    np.random.seed(42)  # For reproducibility

    # Try adaptive evolutionary optimization first
    try:
        result = adaptive_evolutionary_optimization(1000)
        if result is not None:
            return result
    except Exception as e:
        print(f"Evolutionary optimization failed: {e}")

    # Fall back to simpler approach if evolutionary optimization fails
    n_steps = 500

    # Use constructed function with geometric pattern
    f_values = construct_geometric_initial_function(n_steps)

    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * 10 for x in f_values]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")