# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from collections import deque

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the autoconvolution g = f*f and its norms efficiently.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4] with equal spacing
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Step size in x domain [-1/4, 1/4]
    dx = 0.5 / (n - 1) if n > 1 else 0.5

    # Compute autoconvolution using numpy's convolution
    g = np.convolve(f_values, f_values, mode='full')

    # Properly extract the convolution result on [-1/2, 1/2]
    # For two functions of length n on [-1/4, 1/4], convolution produces 2*n-1 points
    center_start = len(g) // 2 - (n - 1)
    center_end = center_start + (2 * n - 1)
    g = g[center_start:center_end]

    # Now we compute the three norms
    # ||g||∞ = max of |g|
    norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0

    # ||g||₁ = sum of |g| * dx
    if len(g) <= 1:
        norm_1 = 0.0
    else:
        # Trapezoidal approximation for ||g||₁
        norm_1 = np.sum(np.abs(g)) * dx

    # ||g||₂² = ∫ g² dx ≈ (dx/3) * Σ (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    if len(g) <= 1:
        norm_2_squared = 0.0
    else:
        # Piecewise linear integration (trapezoidal-like for quadratic form)
        norm_2_squared = 0.0
        for i in range(len(g)-1):
            y1, y2 = g[i], g[i+1]
            norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)

    return norm_2_squared, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute the C2 value for given step function."""
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def adaptive_gaussian_construction(n_points: int = None) -> List[float]:
    """
    Construct step function using adaptive Gaussian shaping with strategic positioning
    and dynamic amplitude adjustment.
    """
    if n_points is None:
        n_points = np.random.randint(300, 1500)  # Increased range for better exploration

    # Generate x coordinates from -1/4 to 1/4
    x = np.linspace(-0.25, 0.25, n_points)

    # Create multiple Gaussian peaks with strategic positioning
    f_values = np.zeros(n_points)

    # Use logarithmic spacing for peak positions to avoid clustering
    num_peaks = np.random.randint(4, 10)
    peak_positions = []

    # Generate logarithmically spaced positions to ensure good distribution
    log_min = np.log(0.01)
    log_max = np.log(0.2)
    log_positions = np.logspace(log_min, log_max, num_peaks)

    # Distribute peaks along the domain with some randomness for diversity
    for i in range(num_peaks):
        # Place peaks in the middle third of domain to avoid edge effects
        side = np.random.choice([-1, 1])
        base_pos = side * log_positions[i] * 0.5
        # Add variation to avoid perfect symmetry
        peak_pos = base_pos + np.random.uniform(-0.015, 0.015)
        # Ensure peak stays within domain bounds
        if abs(peak_pos) <= 0.25:
            peak_positions.append(peak_pos)

    # Generate peaks with adaptive amplitudes
    peak_width = 0.04  # Fixed peak width for consistency

    for peak_pos in peak_positions:
        # Adaptive amplitude based on peak position to promote flatter autoconvolutions
        center_distance = abs(peak_pos)
        if center_distance < 0.05:  # Near center
            amp_factor = 1.8
        elif center_distance < 0.1:  # Middle area
            amp_factor = 1.3
        else:  # Outer areas
            amp_factor = 0.9

        peak_amplitude = np.random.uniform(0.7, 1.8) * amp_factor

        # Create Gaussian peak with dynamic width adjustment
        gaussian_peak = peak_amplitude * np.exp(-0.5 * ((x - peak_pos) / peak_width)**2)
        f_values += gaussian_peak

    # Apply adaptive smoothing with optimized Gaussian kernel
    if len(f_values) > 10:
        # Use Gaussian filtering with adaptive sigma
        sigma = max(1.0, len(f_values) / 200.0)  # Dynamic smoothing strength
        f_values = gaussian_filter1d(f_values, sigma=sigma, mode='constant', cval=0.0)

    # Ensure non-negative values
    f_values = np.clip(f_values, 0, None)

    # Normalize to control overall magnitude
    if np.max(f_values) > 1e-6:
        f_values = f_values / np.max(f_values) * 3.0  # Scale to reasonable range

    return f_values.tolist()

def selective_local_optimization(individual: List[float], max_iter: int = 30) -> List[float]:
    """
    Apply selective optimization focusing on key parameters that influence autoconvolution shape.
    """
    # Convert to numpy for easier manipulation
    x = np.array(individual)
    n = len(x)
    
    # If function is too small, just return
    if n <= 20:
        return individual
    
    # Apply more sophisticated smoothing to preserve feature importance
    # Use larger windows for more extensive smoothing
    window_size = min(7, max(3, n // 15))
    if window_size > 1:
        smoothed = np.convolve(x, np.ones(window_size)/window_size, mode='same')
        
        # Preserve significant peaks by thresholding
        threshold = np.percentile(x, 70)
        # Only smooth low-value regions to reduce artifacts
        mask = x > threshold * 0.2
        result = np.where(mask, x, smoothed)
    else:
        result = x
        
    # Ensure non-negativity
    result = np.clip(result, 0, None)
    
    return result.tolist()

def adaptive_mutation(individual: List[float], generation: int, best_c2: float, 
                     recent_improvements: deque) -> List[float]:
    """Enhanced mutation strategy with adaptive parameters."""
    mutated = individual.copy()
    n = len(mutated)

    # Dynamic mutation parameters based on performance and generation
    if best_c2 > 0.97:
        mutation_rate = 0.03
        noise_sigma = 0.015
    elif best_c2 > 0.95:
        mutation_rate = 0.05
        noise_sigma = 0.02
    elif best_c2 > 0.92:
        mutation_rate = 0.08
        noise_sigma = 0.03
    else:
        mutation_rate = 0.12
        noise_sigma = 0.04

    # Apply enhanced mutation strategy
    for i in range(n):
        if random.random() < mutation_rate:
            # Use mixed noise types for robust exploration
            if random.random() < 0.7:  # 70% Gaussian noise
                mutated[i] += np.random.normal(0, noise_sigma * np.mean(mutated) if np.mean(mutated) > 0 else 0.01)
            else:  # 30% Cauchy noise for heavy-tailed exploration
                mutated[i] += np.random.standard_cauchy() * noise_sigma * 2
            
            # Ensure non-negativity
            mutated[i] = max(0.0, mutated[i])

    # Local smoothing with adaptive intensity
    if random.random() < 0.3 and n > 20:
        # Apply selective smoothing based on recent improvement trends
        if len(recent_improvements) > 2 and np.std(list(recent_improvements)[-3:]) < 0.001:
            window_size = min(5, max(2, n // 25))
        else:
            window_size = min(5, max(1, n // 15))
            
        if window_size > 1:
            smoothed = np.convolve(mutated, np.ones(window_size)/window_size, mode='same')
            # Mix with original using adaptive alpha
            alpha = random.uniform(0.2, 0.6)
            mutated = [alpha * old + (1 - alpha) * new for old, new in zip(mutated, smoothed)]

    return mutated

def evolve_population(population: List[List[float]],
                      fitnesses: List[float],
                      generation: int,
                      population_size: int,
                      best_fitness: float,
                      recent_improvements: deque) -> List[List[float]]:
    """Generate next generation using tournament selection and enhanced mutation."""
    # Sort by fitness
    sorted_indices = np.argsort(fitnesses)[::-1]

    # Keep top 35%
    elite_count = max(1, population_size // 3)
    elites = [population[i] for i in sorted_indices[:elite_count]]

    # Generate offspring through tournament selection and enhanced mutation
    offspring = []

    while len(offspring) < population_size - elite_count:
        # Tournament selection
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]

        # Mutate the winner with adaptive strategy
        mutated = adaptive_mutation(population[winner_index], generation, best_fitness, recent_improvements)
        offspring.append(mutated)

    # Combine elites and offspring
    new_population = elites + offspring

    return new_population

def evaluate_population(population: List[List[float]]) -> List[float]:
    """Evaluate fitness for entire population in parallel."""
    def evaluate_single(individual):
        try:
            return compute_c2(individual)
        except Exception:
            return 0.0

    # Parallel evaluation
    fitnesses = Parallel(n_jobs=-1, backend='threading')(
        delayed(evaluate_single)(ind) for ind in population
    )

    return fitnesses

def construct_function() -> List[float]:
    """Optimized function to construct step-function with high C2 value."""
    # Parameters for evolution
    population_size = 60
    generations = 300
    max_time_seconds = 85  # Leave room for finalization

    start_time = time.time()

    # Initialize population with adaptive Gaussian constructions
    def create_adaptive_individual():
        return adaptive_gaussian_construction()

    # Generate initial population with varied strategies
    population = [create_adaptive_individual() for _ in range(population_size)]

    best_fitness = -float('inf')
    best_individual = None
    
    # Track recent improvements for adaptive strategies
    recent_improvements = deque(maxlen=10)

    # Precompute initial fitnesses
    fitnesses = evaluate_population(population)

    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break

        # Update best solution
        current_best_idx = np.argmax(fitnesses)
        current_fitness = fitnesses[current_best_idx]
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = population[current_best_idx].copy()
            recent_improvements.append(current_fitness)

        # Apply local optimization to best individual periodically
        if gen % 10 == 0 and best_fitness > 0.95 and best_individual is not None:
            optimized = selective_local_optimization(best_individual)
            optimized_c2 = compute_c2(optimized)
            if optimized_c2 > best_fitness:
                best_individual = optimized
                # Update fitness in population
                for i, individual in enumerate(population):
                    if individual == population[current_best_idx]:
                        fitnesses[i] = optimized_c2
                        break

        # Evolve population
        population = evolve_population(population, fitnesses, gen, population_size, 
                                     best_fitness, recent_improvements)

        # Evaluate new population
        fitnesses = evaluate_population(population)

        # Early stopping if no improvement
        if gen > 10 and len(recent_improvements) > 1:
            recent_avg = np.mean(list(recent_improvements)[-3:])
            current_avg = np.mean(list(recent_improvements)[-2:])
            if abs(current_avg - recent_avg) < 1e-6:
                if gen > 50:
                    break

        # Adaptive population size adjustment based on diversity
        if gen % 50 == 0 and gen > 0:
            # Calculate diversity metric
            diversity = np.std([np.mean(ind) for ind in population])
            if diversity < 0.005 and len(recent_improvements) > 2:
                # If very homogeneous, increase diversity
                population_size = min(200, population_size + 10)
            elif len(recent_improvements) > 2 and recent_improvements[-1] > np.mean(list(recent_improvements)[:-1]):
                # If improving, maintain population size
                pass
            else:
                # If not improving much, reduce population size slightly
                population_size = max(20, population_size - 5)
        else:
            # Keep population size consistent unless needed
            population_size = min(200, max(20, population_size))

    # Final refinement of best individual if it looks promising
    if best_individual is not None:
        final_c2 = compute_c2(best_individual)
        if final_c2 > 0.95:
            refined = selective_local_optimization(best_individual)
            refined_c2 = compute_c2(refined)
            if refined_c2 > final_c2:
                best_individual = refined

    return best_individual if best_individual is not None else []

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")