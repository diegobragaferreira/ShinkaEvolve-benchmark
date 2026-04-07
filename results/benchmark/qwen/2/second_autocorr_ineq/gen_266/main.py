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

# Global constants for performance tuning
MAX_STEPS = 10000
MIN_STEPS = 100
BASE_POPULATION_SIZE = 50
GENERATIONS = 150
TIME_BUDGET = 85.0  # seconds
MEMORY_SIZE = 10  # Number of best solutions to remember

# Global cache for storing previously computed C2 values
c2_cache = {}

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞) where g = f*f
    """
    if not f_values or len(f_values) == 0:
        return 0.0, 0.0, 0.0

    # Convert to numpy array
    f = np.array(f_values, dtype=np.float64)

    # Early exit for invalid arrays
    if np.isnan(f).any() or np.isinf(f).any():
        return 0.0, 0.0, 0.0

    # Compute autoconvolution g = f * f
    try:
        g = signal.convolve(f, f, mode='full')
    except Exception:
        return 0.0, 0.0, 0.0

    # Extract central portion (valid autoconvolution)
    half_len = len(f) - 1
    if len(g) >= half_len:
        g = g[half_len:]  # Take right half
    else:
        return 0.0, 0.0, 0.0

    # Compute norms with numerical stability checks
    try:
        g_squared = g * g
        norm_2_sq = np.sum(g_squared)

        norm_1 = np.sum(np.abs(g))
        norm_inf = np.max(np.abs(g))

        # Check for numerical stability
        if np.isnan(norm_2_sq) or np.isnan(norm_1) or np.isnan(norm_inf):
            return 0.0, 0.0, 0.0

        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0, 0.0, 0.0

        return float(norm_2_sq), float(norm_1), float(norm_inf)
    except Exception:
        return 0.0, 0.0, 0.0

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero with stricter check
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    try:
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return float(c2) if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def create_structured_gaussian_individual(n_steps: int, n_peaks: int = None) -> List[float]:
    """
    Create a structured individual using adaptive Gaussian peak construction
    with controlled spacing and amplitude scaling.
    """
    # Determine number of peaks based on function length
    if n_peaks is None:
        n_peaks = max(3, min(15, n_steps // 100))

    # Create step function with Gaussian peaks
    x = np.linspace(-0.25, 0.25, n_steps, dtype=np.float64)
    f_vals = np.zeros(n_steps, dtype=np.float64)

    # Generate peak parameters with adaptive logarithmic spacing
    peak_positions = []
    peak_widths = []
    peak_heights = []

    # Use adaptive peak positioning that considers both function size and optimal distribution
    if n_peaks > 1:
        # Create adaptive logarithmic spacing that places peaks strategically
        # For better C2 performance, we want more peaks near the center where autoconvolution
        # interactions are more constructive
        if n_peaks == 2:
            # Two peaks: one at each end with moderate separation
            peak_positions = [-0.15, 0.15]
        elif n_peaks == 3:
            # Three peaks: symmetric placement with central peak
            peak_positions = [-0.18, 0.0, 0.18]
        elif n_peaks == 4:
            # Four peaks: two outer and two inner
            peak_positions = [-0.2, -0.05, 0.05, 0.2]
        else:
            # For more peaks, use adaptive logarithmic spacing with center emphasis
            # Start with a power law distribution that emphasizes center placement
            # This creates a distribution like: [0.01, 0.03, 0.07, 0.15, 0.25, 0.35, 0.45, 0.49]
            # but more carefully distributed with center peaks having more influence

            # Base distribution using power law with center emphasis
            alpha = 1.5  # Shape parameter for power law (higher = more center weight)
            base_positions = np.linspace(0, 1, n_peaks)
            # Apply power law transformation to emphasize center
            transformed = np.power(base_positions, alpha)
            # Normalize and map to [-0.25, 0.25] with proper offsets
            normalized = transformed / np.max(transformed)
            center_offset = 0.0  # Slight center bias
            peak_positions = (normalized - 0.5) * 0.5 + center_offset
            # Apply reflection and clipping
            peak_positions = np.clip(peak_positions, -0.24, 0.24)

            # Ensure symmetrical distribution for odd numbers of peaks
            if n_peaks % 2 == 1:
                peak_positions[n_peaks // 2] = 0.0  # Force center peak to be exactly at center
    else:
        # For single peak, center it
        peak_positions.append(0.0)

    # Add small random perturbations to break perfect symmetry but maintain structure
    for i in range(len(peak_positions)):
        if n_peaks > 1:
            # Perturbations scaled by peak distance to preserve distribution
            perturbation = np.random.uniform(-0.005, 0.005) * (0.25 if i == 0 else 0.1)
            peak_positions[i] = float(np.clip(peak_positions[i] + perturbation, -0.24, 0.24))

    # Generate peak parameters with better statistical distribution
    for i in range(n_peaks):
        # Width inversely related to height for better control
        # Use a more refined distribution based on peak position
        base_width = 0.015
        # Peaks closer to center may benefit from slightly narrower widths for sharper interactions
        width_factor = 1.0 + 0.2 * np.abs(peak_positions[i])  # Wider for edge peaks
        width = base_width * width_factor
        # Clamp width to reasonable values
        width = np.clip(width, 0.008, 0.03)
        peak_widths.append(float(width))

        # Height inversely proportional to width to maintain balance
        # But adjust based on peak position - central peaks can be taller
        base_height = 1.5
        height_factor = 1.0 + 0.4 * (1 - np.abs(peak_positions[i]) / 0.25)  # Taller at center
        height = base_height * height_factor
        # Clamp height to reasonable values
        height = np.clip(height, 0.8, 2.5)
        peak_heights.append(float(height))

    # Create Gaussian curves for each peak
    for center, width, height in zip(peak_positions, peak_widths, peak_heights):
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Apply mathematically principled Gaussian smoothing instead of Savitzky-Golay
    if n_steps > 50:
        from scipy.ndimage import gaussian_filter1d
        try:
            # Use adaptive sigma based on function size for better smoothing
            sigma = max(0.5, n_steps / 200.0)
            f_vals = gaussian_filter1d(f_vals, sigma=sigma, mode='constant', cval=0.0)
        except:
            pass

    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0.0)

    # Normalize to reasonable range but preserve peak structure
    if np.max(f_vals) > 0:
        # Scale to approximately unit max but allow some headroom
        f_vals = f_vals / np.max(f_vals) * 1.4

    return f_vals.tolist()

def create_balanced_distribution(n_steps: int) -> List[float]:
    """Create a balanced distribution that typically performs well"""
    x = np.linspace(-0.25, 0.25, n_steps)

    # Create a base shape with multiple peaks
    base_shape = (0.5 * np.exp(-x**2 / 0.02) +
                 0.3 * np.exp(-((x - 0.1)**2) / 0.01) +
                 0.2 * np.exp(-((x + 0.1)**2) / 0.01))

    # Normalize and add some randomness
    base_shape = base_shape / np.max(base_shape) * 0.8

    # Add small random variations
    noise = np.random.normal(0, 0.02, n_steps)
    final_shape = np.maximum(base_shape + noise, 0)

    return final_shape.tolist()

def create_bell_curve_distribution(n_steps: int) -> List[float]:
    """Create a simple bell curve distribution"""
    x = np.linspace(-1, 1, n_steps)
    gaussian_shape = np.exp(-x**2 / 2)
    # Normalize and scale to [0.2, 0.8] range
    gaussian_shape = 0.6 * (gaussian_shape / np.max(gaussian_shape)) + 0.2

    # Add some structured noise that preserves good properties
    f_values = []
    for i in range(n_steps):
        base_val = gaussian_shape[i]
        # Add structured noise that preserves good properties
        noise = 0.05 * np.sin(i * 0.1) + 0.02 * np.random.randn()
        val = max(0, base_val + noise)
        f_values.append(val)

    return f_values

def create_hybrid_initialization(n_steps: int) -> List[float]:
    """Create an initial population member using hybrid approach"""
    # Randomly select one of several initialization strategies
    strategy = np.random.choice(['gaussian', 'balanced', 'bell'])

    if strategy == 'gaussian':
        return create_structured_gaussian_individual(n_steps)
    elif strategy == 'balanced':
        return create_balanced_distribution(n_steps)
    else:  # bell
        return create_bell_curve_distribution(n_steps)

def optimize_peak_parameters(best_function: List[float], n_steps: int) -> List[float]:
    """Use selective optimization on peak parameters for final refinement"""
    try:
        # Identify peak locations by finding local maxima
        x = np.linspace(-0.25, 0.25, n_steps)
        f_vals = np.array(best_function)

        # Detect peaks using gradient-based approach
        df = np.gradient(f_vals)
        peaks = []

        for i in range(1, len(f_vals)-1):
            if f_vals[i] > f_vals[i-1] and f_vals[i] > f_vals[i+1]:
                peaks.append((x[i], f_vals[i], i))

        # Sort by height to get strongest peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:min(10, len(peaks))]

        if len(top_peaks) < 2:
            # Not enough peaks to optimize - return original
            return best_function

        # Use differential evolution to fine-tune peak parameters
        def objective_function(params):
            # Create modified function
            temp_func = np.array(best_function)

            # Apply parameter adjustments (simplified approach for speed)
            # Just test a few key parameter combinations for quick improvement
            try:
                # Try several small adjustments
                adjusted_func = temp_func.copy()
                for i in range(min(len(params), len(adjusted_func))):
                    if i < len(adjusted_func):
                        # Adaptive scaling factor based on param magnitude
                        scale_factor = 0.1 + 0.05 * abs(params[i])  # Base 0.1 with adaptive component
                        adjusted_func[i] = max(0.0, adjusted_func[i] * (1.0 + params[i] * scale_factor))

                c2_val = compute_c2(adjusted_func.tolist())
                return -c2_val  # Negative for maximization
            except:
                return 1e10

        # Use small subset of parameters for faster optimization
        sample_size = min(20, n_steps)
        bounds = [(-0.5, 0.5) for _ in range(sample_size)]

        # Reduced iterations for speed
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=30,
            popsize=8,
            seed=42,
            disp=False
        )

        if result.success:
            # Apply adjustments if they improve C2
            temp_func = np.array(best_function)
            for i in range(min(len(result.x), len(temp_func))):
                if i < len(temp_func):
                    # Adaptive scaling factor based on param magnitude
                    scale_factor = 0.1 + 0.05 * abs(result.x[i])  # Base 0.1 with adaptive component
                    temp_func[i] = max(0.0, temp_func[i] * (1.0 + result.x[i] * scale_factor))

            return temp_func.tolist()

    except Exception:
        pass

    # Fall back to original if optimization fails
    return best_function

def adaptive_evolutionary_strategy(toolbox, pop, stats, start_time):
    """Enhanced evolutionary strategy with adaptive parameters"""
    global c2_cache

    # Track best solutions for memory-based selection
    best_solutions = []  # List of (fitness, individual) tuples

    remaining_time = TIME_BUDGET - (time.time() - start_time)
    if remaining_time < 5:
        return pop

    # Adaptive population size based on time available and problem complexity
    adaptive_pop_size = max(BASE_POPULATION_SIZE,
                          min(BASE_POPULATION_SIZE * 2, int(remaining_time / 2)))

    # If we have time, run more generations
    generations_to_run = min(GENERATIONS, int(remaining_time / 3))

    for gen in range(generations_to_run):
        if time.time() - start_time > TIME_BUDGET - 2:
            break

        # Run evolution step
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                           ngen=1, stats=stats, verbose=False)

        # Memory-based selection: preserve best solutions
        current_best = tools.selBest(pop, 1)[0]
        current_fitness = current_best.fitness.values[0]

        # Add to memory if it's among top solutions
        if len(best_solutions) < MEMORY_SIZE:
            best_solutions.append((current_fitness, current_best))
            best_solutions.sort(key=lambda x: x[0], reverse=True)
        elif current_fitness > best_solutions[-1][0]:
            # Replace worst in memory
            best_solutions[-1] = (current_fitness, current_best)
            best_solutions.sort(key=lambda x: x[0], reverse=True)

    return pop

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value using evolutionary approach."""

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    start_time = time.time()

    # Configuration parameters
    # Determine the number of steps to use with time consideration
    n_steps = min(MAX_STEPS, max(MIN_STEPS, 1000 + int(np.random.randint(0, 300) * 5)))

    # Ensure we don't exceed our time budget with creation overhead
    if time.time() - start_time > TIME_BUDGET - 5:
        return [0.5] * n_steps

    # Create fitness function and individual representation
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene range (step heights between 0 and 2.0)
    def create_individual():
        # Use hybrid initialization instead of purely structured Gaussian
        return create_hybrid_initialization(n_steps)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function with enhanced error handling
    def evaluate(individual):
        # Ensure non-negative values
        individual = [max(0.0, val) for val in individual]
        try:
            c2_value = compute_c2(individual)
            # Penalize very low C2 values to avoid numerical issues
            if c2_value < 0.01:
                c2_value = 0.0
            return (float(c2_value),)
        except Exception:
            return (0.0,)

    toolbox.register("evaluate", evaluate)

    # Genetic operators - enhanced with more aggressive mutation for exploration
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)  # Increased sigma for more exploration
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the evolutionary algorithm
    try:
        # Create initial population with hybrid initialization
        pop = toolbox.population(n=BASE_POPULATION_SIZE)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run enhanced evolutionary strategy
        pop = adaptive_evolutionary_strategy(toolbox, pop, stats, start_time)

        # Get best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_function = [max(0.0, val) for val in best_ind]

    except Exception:
        # Fallback to simpler approach if evolution fails
        best_function = []
        for i in range(n_steps):
            # Create a basic bell-curve shaped function
            x = (i / (n_steps - 1)) * 2 - 1  # Map to [-1, 1]
            base_val = max(0.0, 0.5 * np.exp(-x**2 / 0.5))
            # Add small noise
            noise = random.uniform(-0.1, 0.1)
            val = base_val + noise
            best_function.append(max(0.0, val))

    # Final refinement with local optimization - enhanced version
    try:
        # Apply selective optimization on the best function
        refined_function = optimize_peak_parameters(best_function, n_steps)

        # Validate and potentially use the refined result
        c2_original = compute_c2(best_function)
        c2_refined = compute_c2(refined_function)

        if c2_refined > c2_original:
            best_function = refined_function

    except Exception:
        pass

    # Additional final heuristic improvement
    try:
        # Try a secondary refinement using the best known approaches
        c2_before = compute_c2(best_function)

        # Try creating a few more candidate functions and compare
        candidates = []
        for _ in range(3):
            candidate = create_structured_gaussian_individual(n_steps)
            candidates.append(candidate)

        # Also try the balanced distribution approach
        balanced_candidate = create_balanced_distribution(n_steps)
        candidates.append(balanced_candidate)

        # Evaluate and choose the best
        best_candidate = best_function
        best_c2 = c2_before

        for candidate in candidates:
            c2 = compute_c2(candidate)
            if c2 > best_c2:
                best_c2 = c2
                best_candidate = candidate

        if best_c2 > c2_before:
            best_function = best_candidate

    except Exception:
        pass

    # Ensure we have the right number of steps
    if len(best_function) != n_steps:
        # Pad or truncate to match exactly
        if len(best_function) < n_steps:
            best_function.extend([0.0] * (n_steps - len(best_function)))
        else:
            best_function = best_function[:n_steps]

    # Final validation to ensure robustness
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