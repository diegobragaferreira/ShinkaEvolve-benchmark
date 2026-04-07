# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import random
import time
from scipy.optimize import differential_evolution
import math

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms(f_values):
    """
    Compute the autoconvolution g = f*f and return its L2, L1, and L-infinity norms.
    Uses piecewise linear integration for L2 norm.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Create convolution using discrete convolution (equivalent to autoconvolution)
    # The convolution will have length 2*n - 1
    g = np.zeros(2 * n - 1)
    
    # Compute autoconvolution manually for efficiency with Numba
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute norms
    # L2 norm squared
    l2_norm_squared = 0.0
    if len(g) >= 2:
        # Piecewise linear integration using trapezoidal rule approximation
        # For intervals, we use (h/3)(y1^2 + y1*y2 + y2^2) for each adjacent pair
        h = 1.0  # Since step size is normalized to 1 for simplicity
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            l2_norm_squared += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)
    
    # L1 norm
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)  # Normalize by number of intervals
    
    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))
    
    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2(l2_norm_squared, l1_norm, l_inf_norm):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def construct_structured_initial_function(length=1000):
    """Construct an initial function with structured patterns that tend to produce good C2 values."""
    # Create a function with alternating high and low regions following a specific pattern
    # This follows the geometric approach but with better structure
    f_values = []
    for i in range(length):
        # Create a pattern that balances high and low values with smooth transitions
        pattern_pos = i % 8
        if pattern_pos < 2:  # Two high peaks
            f_values.append(np.random.uniform(0.8, 1.0))
        elif pattern_pos < 4:  # Two medium regions
            f_values.append(np.random.uniform(0.4, 0.8))
        elif pattern_pos < 6:  # Two low regions
            f_values.append(np.random.uniform(0.1, 0.4))
        else:  # Two very low regions
            f_values.append(np.random.uniform(0.0, 0.2))
    
    # Add some smoothing to reduce numerical artifacts and improve convergence
    smoothed = []
    for i in range(len(f_values)):
        # Apply a simple moving average
        window_size = min(3, len(f_values))
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(f_values), i + window_size//2 + 1)
        avg = np.mean(f_values[start_idx:end_idx])
        smoothed.append(avg)

    return smoothed

def compute_c2(f_values):
    """Wrapper function to compute C2 from step function values"""
    try:
        l2, l1, l_inf = compute_autoconvolution_norms(f_values)
        return calculate_c2(l2, l1, l_inf)
    except Exception:
        return 0.0

def optimize_with_de():
    """Use differential evolution to optimize the function."""
    # Start with a good structured initial function
    initial_f = construct_structured_initial_function(1000)

    # Define bounds for each parameter (height of each step)
    bounds = [(0.0, 2.0) for _ in range(len(initial_f))]

    # Objective function for differential evolution
    def objective(params):
        # Clip negative values
        params = [max(0.0, p) for p in params]
        return -compute_c2(params)  # Negative because we want to maximize

    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False
    )

    # Return the best solution found
    optimized_params = [max(0.0, p) for p in result.x]
    return optimized_params

def generate_initial_population(pop_size, min_length=500, max_length=2000):
    """Generate initial population with enhanced hybrid initialization."""
    population = []
    for _ in range(pop_size):
        # Random length within range
        length = np.random.randint(min_length, max_length + 1)
        
        # Enhanced hybrid initialization with structured patterns
        individual = []
        for i in range(length):
            # Create structured pattern with alternating high/low regions
            pattern_pos = i % 8
            if pattern_pos < 2:  # High peaks
                individual.append(np.random.uniform(0.8, 1.0))
            elif pattern_pos < 4:  # Medium peaks
                individual.append(np.random.uniform(0.4, 0.8))
            elif pattern_pos < 6:  # Low valleys
                individual.append(np.random.uniform(0.1, 0.4))
            else:  # Very low
                individual.append(np.random.uniform(0.0, 0.2))
        
        # Add structured noise for diversity
        for i in range(len(individual)):
            if np.random.random() < 0.3:  # 30% chance to modify
                individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.1))
        
        population.append(individual)
    
    return population

def mutate_individual(individual, generation, max_generations):
    """Mutate an individual with improved adaptive mutation rate."""
    # Adaptive mutation rate that decreases more gradually and has better shaping
    base_mutation_rate = 0.4
    decay_factor = 0.005
    mutation_rate = base_mutation_rate - (generation / max_generations) * (base_mutation_rate - decay_factor)
    # Add a minimum mutation rate to maintain diversity
    mutation_rate = max(mutation_rate, 0.05)
    
    mutated = individual.copy()
    
    # Mutate each element with probability mutation_rate
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Add normally distributed noise with adaptive scale
            noise_scale = 0.1 + (0.05 * (generation / max_generations))
            # Make noise proportional to current value to prevent extreme changes
            noise_scale *= (mutated[i] + 0.1)  # Avoid zero division
            mutated[i] = max(0.0, mutated[i] + np.random.normal(0, noise_scale))
    
    return mutated

def crossover(parent1, parent2):
    """Perform improved crossover between two individuals with preference for good traits."""
    # Ensure both parents have same length
    min_len = min(len(parent1), len(parent2))
    child1, child2 = [], []
    
    # Improved crossover with weighted selection bias
    for i in range(min_len):
        # With 70% chance to inherit from parent1, 30% from parent2 to preserve good traits
        if np.random.random() < 0.7:
            child1.append(parent1[i])
        else:
            child1.append(parent2[i])
            
        # Inverse bias for second child
        if np.random.random() < 0.7:
            child2.append(parent2[i])
        else:
            child2.append(parent1[i])
    
    # Handle differing lengths by extending with random values
    if len(parent1) > min_len:
        for i in range(min_len, len(parent1)):
            child1.append(np.random.uniform(0, 1))
    elif len(parent2) > min_len:
        for i in range(min_len, len(parent2)):
            child1.append(np.random.uniform(0, 1))
        
    if len(parent2) > min_len:
        for i in range(min_len, len(parent2)):
            child2.append(np.random.uniform(0, 1))
    elif len(parent1) > min_len:
        for i in range(min_len, len(parent1)):
            child2.append(np.random.uniform(0, 1))
    
    return child1, child2

def evaluate_fitness(individual):
    """Evaluate fitness of an individual (C2 value)."""
    try:
        l2, l1, l_inf = compute_autoconvolution_norms(individual)
        c2 = calculate_c2(l2, l1, l_inf)
        return c2
    except Exception:
        return 0.0

def evolutionary_optimization(max_generations=30, pop_size=100):
    """Main evolutionary optimization loop with enhanced parameters."""
    # Initialize population with improved settings
    population = generate_initial_population(pop_size, 500, 2000)
    
    best_c2 = 0.0
    best_individual = None
    stagnation_counter = 0
    max_stagnation = 10
    
    # Evolutionary process
    for generation in range(max_generations):
        # Evaluate fitness of all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append((individual, fitness))
        
        # Sort by fitness descending
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best solution so far
        current_best = fitness_scores[0][1]
        if current_best > best_c2:
            best_c2 = current_best
            best_individual = fitness_scores[0][0].copy()
            stagnation_counter = 0  # Reset stagnation counter on improvement
        else:
            stagnation_counter += 1
        
        # Select top individuals (higher selection pressure)
        top_count = int(pop_size * 0.25)  # Top 25% for selection pressure
        selected = [ind for ind, _ in fitness_scores[:top_count]]
        
        # Preserve multiple elites for stability
        elites = [ind for ind, _ in fitness_scores[:int(pop_size * 0.05)]]  # Top 5% as elites
        
        # Generate new population through crossover and mutation
        new_population = elites.copy()  # Elitism with multiple elites
        
        # Continue generating offspring until population is filled
        while len(new_population) < pop_size:
            # Selection with tournament size adaptation based on generation
            tournament_size = max(3, 7 - generation // 5)  # Decrease tournament size over generations
            parent1 = max(random.sample(selected, tournament_size), key=lambda x: evaluate_fitness(x))
            parent2 = max(random.sample(selected, tournament_size), key=lambda x: evaluate_fitness(x))
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1, generation, max_generations)
            child2 = mutate_individual(child2, generation, max_generations)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:pop_size]
        
        # Early stopping check - if we're stagnating, stop early
        if stagnation_counter >= max_stagnation:
            break
    
    return best_individual, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Try optimization approach first (differential evolution)
    try:
        # Use differential evolution to find a good global solution
        f_values = optimize_with_de()
        
        # If we have time left, do a quick local refinement with evolutionary algorithm
        elapsed = time.time() - start_time
        if elapsed < 80:  # Still have time for refinement
            # Run a small evolutionary optimization for local refinement
            refined_individual, refined_c2 = evolutionary_optimization(
                max_generations=15,
                pop_size=50
            )
            
            # Check if evolutionary improvement is significant
            if refined_c2 > compute_c2(f_values):
                f_values = refined_individual
    except Exception as e:
        # Fallback to evolutionary approach if optimization fails
        try:
            f_values, _ = evolutionary_optimization(
                max_generations=30,
                pop_size=80
            )
        except Exception:
            # Final fallback to structured initialization
            f_values = construct_structured_initial_function(500)

    # Ensure we don't exceed time limits
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for final processing
        # Return a reasonable solution
        return construct_structured_initial_function(500)

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
