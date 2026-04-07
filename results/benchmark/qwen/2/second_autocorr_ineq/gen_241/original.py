# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from deap import base, creator, tools, algorithms
import random
from scipy import signal
import math

def compute_autoconvolution(f_values: List[float]) -> np.ndarray:
    """Compute the autoconvolution g = f * f of step function f."""
    n = len(f_values)
    if n == 0:
        return np.array([])

    f_array = np.array(f_values)

    # Compute convolution using numpy's convolve (full mode)
    g = np.convolve(f_array, f_array, mode='full')

    # Trim to appropriate size (should be 2*n-1 elements)
    g = g[n-1:-(n-1)] if n > 1 else g

    return g

def compute_norms(g_values: np.ndarray) -> tuple:
    """Compute the three required norms for C2 calculation."""
    if len(g_values) == 0:
        return 0.0, 0.0, 0.0

    # ||g||₂² using trapezoidal-like piecewise linear integration
    # For adjacent points y1, y2 with width h, contribution is (h/3)(y1² + y1*y2 + y2²)
    if len(g_values) <= 1:
        norm_2_sq = g_values[0]**2 if len(g_values) > 0 else 0.0
    else:
        norm_2_sq = 0.0
        for i in range(len(g_values)-1):
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

def enforce_peak_spacing(peak_params: List[float], domain_width: float = 0.5,
                        min_distance_ratio: float = 0.1) -> None:
    """Enforce minimum distance between Gaussian peaks to prevent narrow autoconvolution."""
    if len(peak_params) < 3:
        return

    # Group peaks by their parameters [amp, center, width]
    peaks = []
    for i in range(0, len(peak_params), 3):
        peaks.append([peak_params[i], peak_params[i+1], peak_params[i+2]])

    # Sort by center position
    peaks.sort(key=lambda x: x[1])

    # Ensure minimum spacing
    min_distance = min_distance_ratio * domain_width
    for i in range(1, len(peaks)):
        prev_center = peaks[i-1][1]
        curr_center = peaks[i][1]
        distance = abs(curr_center - prev_center)

        if distance < min_distance:
            # Adjust position of current peak
            # Move it away from the previous one
            offset = min_distance - distance
            if curr_center > prev_center:
                peaks[i][1] += offset
            else:
                peaks[i][1] -= offset

    # Put them back into flat list
    for i, (amp, center, width) in enumerate(peaks):
        peak_params[i*3] = amp
        peak_params[i*3 + 1] = center
        peak_params[i*3 + 2] = width

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
        # Enforce spacing before evaluation to prevent numerical issues
        enforce_peak_spacing(individual)

        # Generate function from peak parameters
        func_values = gaussian_peak_function(domain_points, individual)
        # Convert to step function values (take discrete samples)
        step_values = func_values.tolist()
        # Compute C2
        c2_value = compute_c2(step_values)
        return (c2_value,)
    except Exception:
        return (0.0,)

def selective_mutation(individual: List[float], mu: float = 0, sigma: float = 0.1) -> List[float]:
    """Perform targeted mutation on peak parameters to improve efficiency"""
    mutated_individual = list(individual)

    # For a peak [amplitude, center, width], we apply different strategies:
    # - amplitude: multiplicative mutation (log-uniform)
    # - center: additive mutation (uniform)
    # - width: multiplicative mutation (log-uniform)
    for i in range(len(mutated_individual)):
        if np.random.random() < 0.3:  # Only mutate 30% of parameters
            if i % 3 == 0:  # amplitude (multiplicative)
                # Multiplicative mutation with log-uniform distribution
                factor = np.exp(np.random.normal(mu, sigma))
                mutated_individual[i] *= factor
                mutated_individual[i] = max(0.1, mutated_individual[i])
            elif i % 3 == 1:  # center (additive)
                # Additive mutation with normal distribution
                delta = np.random.normal(mu, sigma * 0.5)
                mutated_individual[i] += delta
                # Keep within bounds
                mutated_individual[i] = max(-0.25, min(0.25, mutated_individual[i]))
            else:  # width (multiplicative)
                # Multiplicative mutation with log-uniform distribution
                factor = np.exp(np.random.normal(mu, sigma * 0.5))
                mutated_individual[i] *= factor
                mutated_individual[i] = max(0.001, mutated_individual[i])

    # After mutation, enforce spacing to maintain numerical stability
    enforce_peak_spacing(mutated_individual)
    return mutated_individual

def optimized_evolutionary_algorithm(num_peaks: int, domain_width: float = 0.5,
                                   population_size: int = 50, generations: int = 50) -> List[float]:
    """Enhanced evolutionary algorithm with targeted mutation and peak spacing enforcement"""

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
    toolbox.register("mutate", selective_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population and run evolution
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3,
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

def fast_peak_optimization(num_peaks: int) -> List[float]:
    """Quick optimization that uses a greedy approach to find good initial peak parameters"""
    # Start with a simple uniform distribution
    best_params = create_individual(num_peaks)

    # Try several random configurations and pick the best
    best_c2 = 0.0
    best_individual = list(best_params)

    for _ in range(20):  # 20 trials
        individual = create_individual(num_peaks)
        # Enforce spacing immediately to avoid numerical problems
        enforce_peak_spacing(individual)

        # Evaluate with smaller domain for speed
        domain_points = np.linspace(-0.25, 0.25, 200)
        c2_value = evaluate_individual(individual, domain_points)[0]

        if c2_value > best_c2:
            best_c2 = c2_value
            best_individual = list(individual)

    # Fine-tune the best individual using the full domain
    domain_points = np.linspace(-0.25, 0.25, 500)
    return list(best_individual)

def adaptive_refinement(f_values: List[float], max_iterations: int = 500,
                       initial_step_size: float = 0.1, min_improvement: float = 1e-6) -> List[float]:
    """
    Apply adaptive refinement to improve step function based on C2 value.
    Uses a hill-climbing approach with adaptive step size.
    """
    current_f = list(f_values)
    current_c2 = compute_c2(current_f)

    # Track progress for adaptive step size adjustment
    prev_c2 = current_c2
    improvement_count = 0
    step_size = initial_step_size

    iteration = 0
    while iteration < max_iterations:
        # Create a slightly modified version of the function
        modified_f = list(current_f)

        # Choose random index to modify
        idx = np.random.randint(len(modified_f))

        # Slightly perturb the value
        delta = np.random.uniform(-step_size, step_size)
        modified_f[idx] = max(0.0, modified_f[idx] + delta)  # Clamp to non-negative

        # Try both positive and negative perturbations
        test_f = list(modified_f)
        test_c2 = compute_c2(test_f)

        # If improvement, accept it
        if test_c2 > current_c2:
            current_f = test_f
            current_c2 = test_c2
            improvement_count += 1

            # Reset counter if significant improvement
            if test_c2 - prev_c2 > min_improvement * 10:
                improvement_count = 0
        else:
            # Only increment if small improvement
            if abs(test_c2 - current_c2) < min_improvement:
                improvement_count += 1

        # Adjust step size based on recent performance
        if improvement_count > 5:
            step_size *= 0.9  # Reduce step size if stuck
            improvement_count = 0
        elif improvement_count == 0:
            step_size = min(initial_step_size, step_size * 1.1)  # Increase if making progress

        prev_c2 = current_c2
        iteration += 1

        # Early stopping: if no meaningful improvement in several iterations
        if improvement_count > 20:
            break

    return current_f

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Combines evolutionary optimization with adaptive refinement and targeted enhancements.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Multi-start approach with different peak counts using enhanced evolutionary algorithm
    best_c2 = 0.0
    best_function = []

    # Test different numbers of peaks with increased efficiency
    peak_counts = [3, 5, 7, 10, 15]

    for num_peaks in peak_counts:
        try:
            # Use fast optimization approach first for quick results
            peak_params = fast_peak_optimization(num_peaks)

            # Then use the enhanced evolutionary algorithm for refinement
            if num_peaks > 5:  # Only use enhanced EA for larger peak counts
                peak_params = optimized_evolutionary_algorithm(
                    num_peaks=num_peaks,
                    domain_width=0.5,
                    population_size=30,
                    generations=30
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

    # If we didn't find anything good with evolutionary approach,
    # try adaptive refinement on Gaussian-based starting point
    if len(best_function) == 0:
        try:
            # Start with Gaussian-based function
            n_steps = np.random.randint(200, 1000)
            base_f = gaussian_step_function(n_steps)

            # Apply adaptive refinement
            refined_f = adaptive_refinement(base_f, max_iterations=1000)

            # Final evaluation
            final_c2 = compute_c2(refined_f)

            if final_c2 > best_c2:
                best_c2 = final_c2
                best_function = refined_f

        except Exception as e:
            pass

    # If we still haven't found a good solution, fallback to simple approach
    if len(best_function) == 0:
        try:
            # Use a basic approach with fewer peaks and enhanced EA
            peak_params = optimized_evolutionary_algorithm(
                num_peaks=5,
                domain_width=0.5,
                population_size=30,
                generations=30
            )

            domain_points = np.linspace(-0.25, 0.25, 500)
            func_values = gaussian_peak_function(domain_points, peak_params)
            best_function = func_values[::10].tolist()
            best_c2 = compute_c2(best_function)
        except Exception as e:
            # Final fallback to uniform distribution
            best_function = [10.0] * 200  # Simple uniform function

    return best_function

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")

def gaussian_step_function(n: int, sigma: float = 0.1) -> List[float]:
    """Generate a Gaussian-based step function with specified number of steps."""
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    # Create a Gaussian shaped function
    y = np.exp(-0.5 * (x/sigma)**2)
    # Normalize to ensure reasonable values
    y = y / np.max(y) * 20
    # Return as integer-valued steps (but they don't have to be)
    return [float(val) for val in y]

def adaptive_refinement(f_values: List[float], max_iterations: int = 500,
                       initial_step_size: float = 0.1, min_improvement: float = 1e-6) -> List[float]:
    """
    Apply adaptive refinement to improve step function based on C2 value.
    Uses a hill-climbing approach with adaptive step size.
    """
    current_f = list(f_values)
    current_c2 = compute_c2(current_f)

    # Track progress for adaptive step size adjustment
    prev_c2 = current_c2
    improvement_count = 0
    step_size = initial_step_size

    iteration = 0
    while iteration < max_iterations:
        # Create a slightly modified version of the function
        modified_f = list(current_f)

        # Choose random index to modify
        idx = np.random.randint(len(modified_f))

        # Slightly perturb the value
        delta = np.random.uniform(-step_size, step_size)
        modified_f[idx] = max(0.0, modified_f[idx] + delta)  # Clamp to non-negative

        # Try both positive and negative perturbations
        test_f = list(modified_f)
        test_c2 = compute_c2(test_f)

        # If improvement, accept it
        if test_c2 > current_c2:
            current_f = test_f
            current_c2 = test_c2
            improvement_count += 1

            # Reset counter if significant improvement
            if test_c2 - prev_c2 > min_improvement * 10:
                improvement_count = 0
        else:
            # Only increment if small improvement
            if abs(test_c2 - current_c2) < min_improvement:
                improvement_count += 1

        # Adjust step size based on recent performance
        if improvement_count > 5:
            step_size *= 0.9  # Reduce step size if stuck
            improvement_count = 0
        elif improvement_count == 0:
            step_size = min(initial_step_size, step_size * 1.1)  # Increase if making progress

        prev_c2 = current_c2
        iteration += 1

        # Early stopping: if no meaningful improvement in several iterations
        if improvement_count > 20:
            break

    return current_f

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Combines evolutionary optimization with adaptive refinement.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Multi-start approach with different peak counts using evolutionary algorithm
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

    # If we didn't find anything good with evolutionary approach,
    # try adaptive refinement on Gaussian-based starting point
    if len(best_function) == 0:
        try:
            # Start with Gaussian-based function
            n_steps = np.random.randint(200, 1000)
            base_f = gaussian_step_function(n_steps)

            # Apply adaptive refinement
            refined_f = adaptive_refinement(base_f, max_iterations=1000)

            # Final evaluation
            final_c2 = compute_c2(refined_f)

            if final_c2 > best_c2:
                best_c2 = final_c2
                best_function = refined_f

        except Exception as e:
            pass

    # If we still haven't found a good solution, fallback to simple approach
    if len(best_function) == 0:
        try:
            # Use a basic approach with fewer peaks
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
        except Exception as e:
            # Final fallback to uniform distribution
            best_function = [10.0] * 200  # Simple uniform function

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")