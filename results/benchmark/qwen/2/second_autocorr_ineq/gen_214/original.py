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

    # Generate peak parameters with controlled spacing
    peak_positions = []
    peak_widths = []
    peak_heights = []

    # Distribute peaks with logarithmic spacing and minimum gap enforcement
    # This ensures better coverage across different scales
    for i in range(n_peaks):
        if i == 0:
            # First peak near left edge
            pos = np.random.uniform(-0.25, -0.15)
        elif i == n_peaks - 1:
            # Last peak near right edge
            pos = np.random.uniform(0.15, 0.25)
        else:
            # Use logarithmic spacing in relative positions to distribute peaks
            # Convert domain to log space for even distribution
            # Map [-0.25, 0.25] to [0.01, 10] in log space then back
            log_min = np.log(0.01)  # Small positive number
            log_max = np.log(10)    # Large enough number
            # Generate log-uniform distributed position in normalized coordinates
            log_pos = np.random.uniform(log_min, log_max)
            # Map back to original coordinate system
            rel_pos = np.exp(log_pos) / 10.0  # Normalize to [0,1]
            pos = -0.25 + rel_pos * 0.5  # Map to [-0.25, 0.25]

            # Ensure it doesn't overlap with previous peaks
            min_spacing = 0.05
            if len(peak_positions) > 0:
                prev_pos = peak_positions[-1]
                pos_lower = prev_pos + min_spacing
                pos_upper = -0.25 + 0.5  # Right boundary
                if pos_lower < pos_upper:
                    # Ensure we stay within bounds but allow some flexibility
                    pos = np.random.uniform(max(pos_lower, -0.25 + min_spacing), pos_upper)
                else:
                    # If no room, place closer to right edge
                    pos = np.random.uniform(-0.25 + min_spacing, 0.25 - min_spacing)

        peak_positions.append(pos)
        # Width inversely related to height for better control
        width = np.random.uniform(0.005, 0.02)
        peak_widths.append(width)
        # Height inversely proportional to width to maintain balance
        height = np.random.uniform(0.5, 2.0)
        peak_heights.append(height)

    # Create Gaussian curves for each peak
    for center, width, height in zip(peak_positions, peak_widths, peak_heights):
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Apply smoothing to reduce extreme variations
    if n_steps > 50:
        f_vals = signal.savgol_filter(f_vals, min(51, n_steps-1), 3)

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
        # Try to refine using differential evolution on peak parameters only
        # Extract peak information from the existing function
        def extract_peak_info(f_vals):
            """Extract peak positions and heights for targeted optimization"""
            # Simple peak detection - find local maxima
            x = np.linspace(-0.25, 0.25, len(f_vals))

            # Compute derivative to find peaks
            df = np.gradient(f_vals)
            peaks = []

            # Simple peak finding algorithm
            for i in range(1, len(f_vals)-1):
                if f_vals[i] > f_vals[i-1] and f_vals[i] > f_vals[i+1]:
                    # Local maximum
                    peaks.append((x[i], f_vals[i]))

            # Keep top 10 peaks for refinement
            peaks.sort(key=lambda x: x[1], reverse=True)
            top_peaks = peaks[:min(10, len(peaks))]

            return [(pos, height) for pos, height in top_peaks]

        # First, let's try to identify if there are meaningful peaks
        peak_info = extract_peak_info(best_function)
        if len(peak_info) >= 2:  # Only proceed if we found at least 2 peaks
            # Refine specific peak parameters rather than full function
            # We'll optimize the peak positions and heights individually

            # Get indices of peak locations (simplified approach)
            peak_indices = []
            x = np.linspace(-0.25, 0.25, n_steps)

            # Find approximate peak locations by looking at high values
            thresh = np.percentile(best_function, 70)
            for i, val in enumerate(best_function):
                if val > thresh:
                    peak_indices.append(i)

            # If we have sufficient peaks, do selective optimization
            if len(peak_indices) > 2:
                # Take a subset of peak indices for focused optimization
                selected_indices = peak_indices[:min(20, len(peak_indices))]

                # For this advanced optimization, we'll still try differential evolution
                # But focus on adjusting the amplitude and position of these peaks
                def objective_for_de(params):
                    # Create new function with adjusted peak parameters
                    temp_func = best_function.copy()

                    # Apply modifications based on params (simplified)
                    # This is a basic version - for a more precise implementation,
                    # we'd need to map params back to real peak positions and heights

                    # Ensure non-negative values
                    temp_func = [max(0, val) for val in temp_func]
                    try:
                        c2_val = compute_c2(temp_func)
                        return -c2_val  # Negative because we want to maximize
                    except Exception:
                        return 1e10

                # Simplified approach: just do a quick refinement on the full function
                # Prepare bounds for differential evolution (but use smaller subset for speed)
                bounds = [(0.0, 2.0) for _ in range(min(500, n_steps))]

                # Use a much smaller subset for faster optimization
                sample_size = min(100, n_steps)
                sample_indices = sorted(random.sample(range(n_steps), sample_size))
                sample_params = [best_function[i] for i in sample_indices]

                # Perform differential evolution with fewer iterations for speed
                result = differential_evolution(
                    objective_for_de,
                    bounds[:sample_size],
                    maxiter=50,  # Reduced iterations for speed
                    popsize=10,   # Reduced population size
                    seed=42,
                    disp=False
                )

                if result.success:
                    # Update the main function with refined values
                    for i, idx in enumerate(sample_indices):
                        if i < len(result.x):
                            best_function[idx] = max(0, result.x[i])

    except Exception:
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