# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple
import time
import math
from deap import base, creator, tools, algorithms
import warnings

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞) where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Convert to numpy array
    f = np.array(f_values)

    # Compute autoconvolution g = f * f
    g = signal.convolve(f, f, mode='full')

    # Extract central portion (valid autoconvolution)
    half_len = len(f) - 1
    g = g[half_len:]  # Take right half

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)

    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_structured_gaussian_individual(n_steps: int, n_peaks: int = None) -> List[float]:
    """
    Create a structured individual using adaptive Gaussian peak construction
    with controlled spacing and amplitude scaling.
    """
    # Determine number of peaks based on function length
    if n_peaks is None:
        n_peaks = max(3, min(15, n_steps // 100))

    # Create step function with Gaussian peaks
    x = np.linspace(-0.25, 0.25, n_steps)
    f_vals = np.zeros(n_steps)

    # Generate peak parameters with improved spacing strategy
    peak_positions = []
    peak_widths = []
    peak_heights = []

    # Improved peak positioning strategy:
    # 1. Use a quasi-random distribution that avoids regular patterns
    # 2. Ensure minimum spacing to prevent overlapping autoconvolution effects
    # 3. Space peaks to encourage constructive interference in autoconvolution

    if n_peaks > 1:
        # Space peaks more strategically to avoid destructive interference
        # We'll use a combination of geometric progression with adjustment
        # and ensure proper minimum spacing

        # Start with a more evenly distributed approach
        positions = []

        # For multiple peaks, use a pattern that places them to promote
        # constructive autoconvolution
        base_positions = np.linspace(0.1, 0.9, n_peaks)

        # Apply a nonlinear transformation to get better distribution
        # This helps avoid clustering at edges
        transformed_positions = 0.25 * (1 + np.tanh(2 * (base_positions - 0.5)))

        # Convert to actual domain coordinates in [-0.25, 0.25]
        for i, pos in enumerate(transformed_positions):
            actual_pos = -0.25 + pos * 0.5
            # Add slight jitter to break symmetry
            jitter = np.random.uniform(-0.02, 0.02) if i > 0 and i < n_peaks-1 else 0
            positions.append(actual_pos + jitter)

        # Ensure proper bounds
        positions = [np.clip(pos, -0.24, 0.24) for pos in positions]

        # Sort positions to make processing easier
        positions.sort()

        # Apply minimum spacing enforcement
        min_spacing = 0.02  # Minimum distance between peaks (about 4% of domain)
        final_positions = []
        last_pos = -100  # Initialize to a very small value

        for pos in positions:
            # Only place peak if it's far enough from previous one
            if abs(pos - last_pos) < min_spacing and len(final_positions) > 0:
                # Adjust position to respect minimum spacing by moving away from neighbors
                if len(final_positions) > 0:
                    # Move towards middle of gap
                    gap_center = (final_positions[-1] + pos) / 2
                    # Place at minimum distance from previous peak
                    new_pos = final_positions[-1] + min_spacing
                    if abs(new_pos - pos) > min_spacing:  # If we can move appropriately
                        pos = new_pos
            final_positions.append(pos)
            last_pos = pos

        peak_positions = final_positions
    else:
        # For single peak, center it
        peak_positions.append(0.0)

    # Generate peak parameters with improved distribution
    for i in range(n_peaks):
        # Width: use a mix of small and medium widths to create variety
        # This helps create interesting convolution behavior
        width_base = np.random.uniform(0.005, 0.025)
        # Adjust width slightly to avoid identical peaks
        width = width_base * (0.8 + np.random.random() * 0.4)
        peak_widths.append(width)

        # Height: vary more systematically to create better balance
        # Prefer taller peaks in the middle to encourage strong autoconvolution
        if n_peaks > 1:
            # Calculate relative position to center (normalized to 0-1)
            rel_pos = (peak_positions[i] + 0.25) / 0.5
            # Center peaks get higher amplitude
            height_base = 1.0 + 0.5 * np.cos(rel_pos * np.pi)  # Peaks at 0 and 1 get more height
            height = height_base * np.random.uniform(0.8, 1.5)  # Add some randomness
        else:
            height = np.random.uniform(1.0, 2.0)  # Single peak gets maximum height
        peak_heights.append(height)

    # Create Gaussian curves for each peak
    for center, width, height in zip(peak_positions, peak_widths, peak_heights):
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Apply mathematical principled smoothing with Gaussian kernel instead of Savitzky-Golay
    # This provides better numerical stability and preserves function characteristics
    if n_steps > 50:
        from scipy.ndimage import gaussian_filter1d
        f_vals = gaussian_filter1d(f_vals, sigma=0.8)

    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0)

    # Normalize to reasonable range but preserve peak structure
    if np.max(f_vals) > 0:
        # Scale to approximately unit max but allow some headroom
        f_vals = f_vals / np.max(f_vals) * 1.5

    return f_vals.tolist()

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value using evolutionary approach."""

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    start_time = time.time()

    # Configuration parameters
    max_steps = 10000  # Maximum allowed steps due to time constraints
    min_steps = 100
    max_evaluations = 5000  # Maximum evaluations for evolution
    population_size = 50
    generations = 200

    # Determine the number of steps to use
    n_steps = min(max_steps, max(min_steps, 1000 + int(np.random.randint(0, 300) * 5)))

    # Create fitness function and individual representation
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene range (step heights between 0 and 2.0)
    def create_individual():
        # Use structured Gaussian initialization instead of purely random
        return create_structured_gaussian_individual(n_steps)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function with enhanced error handling
    def evaluate(individual):
        # Ensure non-negative values
        individual = [max(0, val) for val in individual]
        try:
            c2_value = compute_c2(individual)
            # Penalize very low C2 values to avoid numerical issues
            if c2_value < 0.01:
                c2_value = 0.0
            return (c2_value,)
        except Exception:
            return (0.0,)

    toolbox.register("evaluate", evaluate)

    # Genetic operators - enhanced with more aggressive mutation for exploration
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)  # Increased sigma for more exploration
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the evolutionary algorithm
    try:
        # Create initial population with better structured individuals
        pop = toolbox.population(n=population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolution
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                           ngen=generations, stats=stats, verbose=False)

        # Get best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_function = [max(0, val) for val in best_ind]

    except Exception:
        # Fallback to simpler approach if evolution fails
        best_function = []
        for i in range(n_steps):
            # Create a basic bell-curve shaped function
            x = (i / (n_steps - 1)) * 2 - 1  # Map to [-1, 1]
            base_val = max(0, 0.5 * np.exp(-x**2 / 0.5))
            # Add small noise
            noise = random.uniform(-0.1, 0.1)
            val = base_val + noise
            best_function.append(max(0, val))

    # Final refinement with local optimization - enhanced version
    try:
        # Check if we have a function that's worth refining
        initial_c2 = compute_c2(best_function)

        # Validate peak quality and make improvements if necessary
        def validate_and_improve_peak_structure(f_vals, max_iterations=10):
            """Improve peak structure by checking autoconvolution properties"""
            improved_func = f_vals.copy()
            current_c2 = compute_c2(improved_func)

            # Simple iterative improvement: look for areas that could benefit from adjustment
            for iteration in range(max_iterations):
                # Compute autoconvolution to understand its structure
                try:
                    g = signal.convolve(improved_func, improved_func, mode='full')
                    half_len = len(improved_func) - 1
                    g = g[half_len:]

                    # Analyze the shape of g
                    # Look for too many sharp peaks that might hurt C2
                    if len(g) > 10:
                        # Check if we have excessively high peaks in autoconvolution
                        g_max = np.max(np.abs(g))
                        g_mean = np.mean(np.abs(g))

                        # If autoconvolution has very high peaks relative to average,
                        # try to reduce them by adjusting function peaks
                        if g_max > 3 * g_mean and len(improved_func) > 50:
                            # Reduce peak heights slightly
                            improved_func = np.array(improved_func)
                            adjusted_func = improved_func * 0.95  # Reduce all values slightly
                            adjusted_func = np.maximum(adjusted_func, 0)

                            new_c2 = compute_c2(adjusted_func.tolist())
                            if new_c2 > current_c2:
                                improved_func = adjusted_func.tolist()
                                current_c2 = new_c2

                except Exception:
                    pass

                # Occasionally make small random adjustments to break local minima
                if iteration % 3 == 0 and random.random() < 0.3:
                    # Apply small random perturbations to peak locations
                    func_array = np.array(improved_func)
                    for i in range(0, len(func_array), max(1, len(func_array)//20)):
                        if random.random() < 0.5:
                            # Slight perturbation
                            perturbation = random.uniform(-0.05, 0.05) * func_array[i]
                            func_array[i] = max(0, func_array[i] + perturbation)
                    improved_func = func_array.tolist()
                    new_c2 = compute_c2(improved_func)
                    if new_c2 > current_c2:
                        current_c2 = new_c2

            return improved_func, current_c2

        # Apply validation and improvement
        improved_func, final_c2 = validate_and_improve_peak_structure(best_function)

        # If the improvement was beneficial, use it
        if final_c2 > initial_c2:
            best_function = improved_func

    except Exception as e:
        pass

    # Ensure we have the right number of steps
    if len(best_function) != n_steps:
        # Pad or truncate to match exactly
        if len(best_function) < n_steps:
            best_function.extend([0.0] * (n_steps - len(best_function)))
        else:
            best_function = best_function[:n_steps]

    # Final validation to ensure robustness - improved version
    try:
        c2_score = compute_c2(best_function)
        if c2_score < 0.1:
            # If score is very poor, reinitialize with better distribution
            best_function = create_structured_gaussian_individual(n_steps)
    except Exception:
        pass

    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")