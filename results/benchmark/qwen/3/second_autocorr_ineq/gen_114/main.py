# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import warnings
warnings.filterwarnings('ignore')

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

    # Direct convolution computation - fully JIT compiled
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
        if g_norm1 <= 1e-15 or g_norm_inf <= 1e-15:
            return 0.0

        return g_norm2_sq / (g_norm1 * g_norm_inf)
    except Exception as e:
        return 0.0

def initialize_smart_parameters(size: int) -> np.ndarray:
    """
    Initialize step function parameters with smart patterns that tend to perform well.
    """
    # Create a combination of smooth and sharp features using geometric progression
    # This creates better structured patterns than purely random approaches
    indices = np.arange(size)
    
    # Base pattern with geometric decay to create smooth yet structured shape
    base_pattern = np.exp(-0.5 * (indices - size/2)**2 / (size/4)**2)
    
    # Add harmonic components for complexity
    t = np.linspace(-2, 2, size)
    harmonic_pattern = (
        0.8 + 
        0.3 * np.sin(2 * np.pi * t) +
        0.2 * np.cos(4 * np.pi * t) +
        0.1 * np.sin(6 * np.pi * t)
    )
    
    # Combine and ensure non-negativity
    pattern = np.maximum(base_pattern + harmonic_pattern, 0.0)
    
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

def evolve_individual(parent: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
    """
    Evolve an individual with adaptive mutation.
    """
    child = parent.copy()
    
    # Apply mutations to random positions
    mutation_mask = np.random.random(len(child)) < 0.1
    if np.any(mutation_mask):
        noise = np.random.normal(0, mutation_strength, len(child))
        child[mutation_mask] += noise[mutation_mask]
        child = np.maximum(child, 0.0)  # Ensure non-negativity
    
    return child

def tournament_selection(population: list, fitness_scores: list, tournament_size: int = 3) -> np.ndarray:
    """
    Select an individual using tournament selection.
    """
    tournament_indices = np.random.choice(len(population), 
                                         size=tournament_size, 
                                         replace=False)
    best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
    return population[best_idx].copy()

def evolve_generation(population: list, fitness_scores: list, generation: int) -> list:
    """
    Evolve one generation with adaptive parameters.
    """
    new_population = []
    pop_size = len(population)
    
    # Elitism: keep best individual
    best_idx = np.argmax(fitness_scores)
    new_population.append(population[best_idx].copy())
    
    # Adaptive mutation strength based on generation
    if generation < 10:
        mutation_strength = 0.3  # High exploration
    elif generation < 20:
        mutation_strength = 0.15  # Balanced
    else:
        mutation_strength = 0.05  # High exploitation
    
    # Generate rest through selection, crossover, and mutation
    for _ in range(pop_size - 1):
        # Selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        
        # Simple crossover (average of two parents)
        child = (parent1 + parent2) / 2.0
        
        # Mutation
        child = evolve_individual(child, mutation_strength)
        
        new_population.append(child)
    
    return new_population

def evolutionary_optimization(size: int, generations: int = 25) -> list:
    """
    Perform evolutionary optimization to find good step function.
    """
    # Initialize population
    population = [initialize_smart_parameters(size) for _ in range(15)]
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Evolve
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [calculate_c2(create_discrete_projection(individual).tolist()) 
                         for individual in population]
        
        # Track best
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[fitness_scores.index(max_fitness)].copy()
        
        # Evolve
        population = evolve_generation(population, fitness_scores, gen)
    
    return best_individual if best_individual is not None else np.ones(size)

def construct_function() -> list:
    """
    Main function to construct step-function with high C2 value using hybrid optimization.
    
    Returns:
        List of step heights that maximize C2
    """
    # Try different sizes for better results
    sizes_to_try = [1000, 1250, 1500]  # Focus on medium-to-large sizes
    best_c2 = -float('inf')
    best_solution = None
    
    for size in sizes_to_try:
        try:
            # First use evolutionary optimization for global search
            evol_solution = evolutionary_optimization(size, 25)
            
            # Convert to discrete form
            discrete_solution = create_discrete_projection(evol_solution)
            
            # Evaluate
            c2_value = calculate_c2(discrete_solution.tolist())
            print(f"Size {size}: C2 = {c2_value:.6f}")
            
            if c2_value > best_c2:
                best_c2 = c2_value
                best_solution = discrete_solution.tolist()
                
        except Exception as e:
            print(f"Failed at size {size}: {e}")
            continue
    
    # Final local refinement on best solution
    if best_solution is not None:
        # Apply simple hill climbing for final improvement
        refined_solution = best_solution.copy()
        current_c2 = calculate_c2(refined_solution)
        
        for _ in range(15):  # Limited iterations for time efficiency
            # Try small random changes
            idx = np.random.randint(len(refined_solution))
            old_val = refined_solution[idx]
            
            # Try small changes
            new_val = max(0, old_val + np.random.normal(0, 0.1))
            refined_solution[idx] = new_val
            
            # Test if this improves the solution
            test_c2 = calculate_c2(refined_solution)
            if test_c2 > current_c2:
                current_c2 = test_c2
            else:
                refined_solution[idx] = old_val  # Revert if worse

    return best_solution if best_solution is not None else [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
