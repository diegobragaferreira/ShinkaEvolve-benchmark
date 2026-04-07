# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from deap import base, creator, tools, algorithms
import random
from scipy import signal

def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
    """Compute the autoconvolution g = f * f of step function f."""
    n = len(f_values)
    if n == 0:
        return np.array([])

    # Create step function with proper spacing
    step_width = 0.5 / n  # interval [-1/4, 1/4] has width 0.5
    f_array = np.array(f_values)

    # Compute convolution using numpy's convolve (valid mode)
    g = np.convolve(f_array, f_array, mode='full')

    # Trim to appropriate size (should be 2*n-1 elements)
    g = g[n-1:-(n-1)] if n > 1 else g

    return g

def compute_norms(g_values: np.ndarray) -> tuple:
    """Compute the three required norms for C2 calculation."""
    if len(g_values) == 0:
        return 0.0, 0.0, 0.0

    # ||g||₂² using trapezoidal-like piecewise linear integration
    if len(g_values) <= 1:
        norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
    else:
        # Using trapezoidal rule for integration
        # Convert to piecewise linear integration
        norm_2_sq = 0.0
        for i in range(len(g_values)-1):
            # For segment from i to i+1 with values g[i] and g[i+1]
            # Contribution = (h/3)*(g[i]^2 + g[i]*g[i+1] + g[i+1]^2)
            h = 1.0  # Normalized width for unit spacing
            norm_2_sq += (h/3.0) * (g_values[i]**2 + g_values[i]*g_values[i+1] + g_values[i+1]**2)

    # ||g||₁: L1 norm, approximated as sum(|g|) / (len(g) + 1)
    if len(g_values) > 0:
        norm_1 = np.sum(np.abs(g_values)) / (len(g_values) + 1)
    else:
        norm_1 = 0.0

    # ||g||∞: Infinity-norm
    norm_inf = np.max(np.abs(g_values)) if len(g_values) > 0 else 0.0

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 for given step function values."""
    g = compute_autoconvolution(f_values)
    norm_2_sq, norm_1, norm_inf = compute_norms(g)

    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def gaussian_peak_function(x: np.ndarray, peak_params: List[float]) -> np.ndarray:
    """
    Generate a function composed of multiple Gaussian peaks.
    peak_params: list of [amplitude, center, width] triplets
    """
    result = np.zeros_like(x)
    for amp, center, width in zip(peak_params[::3], peak_params[1::3], peak_params[2::3]):
        # Ensure width is positive
        width = max(width, 1e-6)
        result += amp * np.exp(-0.5 * ((x - center) / width)**2)
    return result

def create_individual(num_peaks: int, domain_width: float = 0.5) -> List[float]:
    """Create a random individual with specified number of Gaussian peaks"""
    # Each peak: [amplitude, center, width]
    individual = []
    for _ in range(num_peaks):
        # Amplitude: 0 to 100
        individual.append(random.uniform(10.0, 100.0))
        # Center: -domain_width/2 to domain_width/2
        individual.append(random.uniform(-domain_width/2, domain_width/2))
        # Width: 0.01 to 0.2
        individual.append(random.uniform(0.01, 0.2))
    return individual

def evaluate_individual(individual: List[float], domain_points: np.ndarray) -> float:
    """Evaluate the fitness of an individual (C2 value)"""
    try:
        # Generate function from peak parameters
        func_values = gaussian_peak_function(domain_points, individual)
        # Convert to step function values (take discrete samples)
        step_values = func_values.tolist()
        # Compute C2
        c2_value = compute_c2(step_values)
        return (c2_value,)
    except Exception:
        return (0.0,)

def optimize_with_evolutionary_algorithm(num_peaks: int, domain_width: float = 0.5,
                                       population_size: int = 50, generations: int = 50) -> List[float]:
    """Optimize peak parameters using evolutionary algorithm"""

    # Define the bounds for each parameter
    # [amplitude, center, width] for each peak
    toolbox = base.Toolbox()

    def create_valid_individual():
        return create_individual(num_peaks, domain_width)

    toolbox.register("individual", tools.initIterate, creator.Individual, create_valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register("evaluate", evaluate_individual, domain_points=np.linspace(-domain_width/2, domain_width/2, 500))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population and run evolution
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                          ngen=generations, stats=stats, halloffame=hof,
                                          verbose=False)
    except:
        # Fallback if evolution fails
        if len(pop) > 0:
            return list(pop[0])
        else:
            # Return random individual
            return create_individual(num_peaks, domain_width)

    # Return the best individual found
    if len(hof) > 0:
        return list(hof[0])
    else:
        return create_individual(num_peaks, domain_width)

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses evolutionary optimization of Gaussian peaks.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Multi-start approach with different peak counts
    best_c2 = 0.0
    best_function = []

    # Test different numbers of peaks
    peak_counts = [3, 5, 7, 10, 15]

    for num_peaks in peak_counts:
        try:
            # Optimize with evolutionary algorithm
            peak_params = optimize_with_evolutionary_algorithm(
                num_peaks=num_peaks,
                domain_width=0.5,
                population_size=50,
                generations=50
            )

            # Generate actual function from peak parameters
            domain_points = np.linspace(-0.25, 0.25, 500)
            func_values = gaussian_peak_function(domain_points, peak_params)

            # Convert to step function (take every 10th point to reduce size)
            step_values = func_values[::10].tolist()

            # Evaluate C2
            c2_val = compute_c2(step_values)

            if c2_val > best_c2:
                best_c2 = c2_val
                best_function = step_values

        except Exception as e:
            continue

    # If we didn't find anything good, fallback to a simple approach
    if len(best_function) == 0:
        # Use a basic approach with fewer peaks
        try:
            peak_params = optimize_with_evolutionary_algorithm(
                num_peaks=5,
                domain_width=0.5,
                population_size=30,
                generations=30
            )

            domain_points = np.linspace(-0.25, 0.25, 500)
            func_values = gaussian_peak_function(domain_points, peak_params)
            best_function = func_values[::10].tolist()
            best_c2 = compute_c2(best_function)
        except:
            # Final fallback
            best_function = [10.0] * 200  # Simple uniform function

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")