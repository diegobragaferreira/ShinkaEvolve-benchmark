# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import random
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution efficiently using numba"""
    n = len(f_vals)
    # Create output array for autoconvolution
    g = np.zeros(2*n - 1, dtype=np.float64)

    # Compute convolution manually with numba optimization
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@numba.jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with numba"""
    n = len(g_vals)

    # L2 norm squared (using trapezoidal-like scheme)
    l2_sq = 0.0
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val

    return l2_sq, l1, linf

def evaluate_individual(individual):
    """Evaluate fitness of an individual (step function)"""
    try:
        # Convert to numpy array and ensure non-negative
        f_vals = np.array(individual, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)

        # Skip if all zeros
        if np.sum(f_vals) == 0:
            return (0.0,)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return (0.0,)

        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return (c2,)
    except:
        return (0.0,)

def adaptive_gaussian_construction(n_steps=None):
    """Create a structured step function with Gaussian peaks for better C2"""
    if n_steps is None:
        n_steps = random.randint(200, 1000)

    # Start with a base structure of Gaussian peaks
    f_vals = np.zeros(n_steps, dtype=np.float64)

    # Determine number of peaks based on function length
    n_peaks = max(2, min(10, n_steps // 100))

    # Place peaks strategically with minimum gap enforcement
    peak_positions = []
    peak_widths = []
    peak_heights = []

    # Generate peak parameters
    for i in range(n_peaks):
        # Ensure minimum spacing between peaks
        if i == 0:
            # First peak at beginning
            center = random.uniform(0.1 * n_steps, 0.3 * n_steps)
        elif i == n_peaks - 1:
            # Last peak at end
            center = random.uniform(0.7 * n_steps, 0.9 * n_steps)
        else:
            # Middle peaks with spacing consideration
            if len(peak_positions) > 0:
                prev_center = peak_positions[-1]
                min_gap = max(20, n_steps // 20)
                center = random.uniform(prev_center + min_gap, n_steps - min_gap)
            else:
                center = random.uniform(0.3 * n_steps, 0.7 * n_steps)

        peak_positions.append(center)
        # Width inversely related to height for better control
        width = random.uniform(10, 30)
        peak_widths.append(width)
        # Height inversely proportional to width to maintain balance
        height = random.uniform(0.5, 2.0)
        peak_heights.append(height)

    # Create Gaussian curves for each peak
    for center, width, height in zip(peak_positions, peak_widths, peak_heights):
        x = np.arange(n_steps, dtype=np.float64)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Apply smoothing to reduce extreme variations
    if n_steps > 50:
        # Use Savitzky-Golay filter for better preservation of shape
        try:
            f_vals = signal.savgol_filter(f_vals, min(51, n_steps-1), 3)
        except:
            pass

    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0)

    # Normalize to reasonable range
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 2.0

    # Apply constraint-aware normalization to prevent extreme autoconvolution spikes
    # This helps avoid numerical instability in later processing
    max_allowed = np.percentile(f_vals, 90) if len(f_vals) > 10 else 1.0
    if max_allowed > 0:
        f_vals = np.minimum(f_vals, max_allowed * 2.0)

    return f_vals.tolist()

def create_individual(size):
    """Create a structured individual using Gaussian construction"""
    return adaptive_gaussian_construction(size)

def mutate_individual(individual):
    """Enhanced mutation with adaptive scaling"""
    for i in range(len(individual)):
        if random.random() < 0.1:  # 10% mutation rate
            # Use adaptive mutation based on current value
            current_value = individual[i]
            if current_value > 0:
                # Scale mutation based on value magnitude
                mutation_scale = 0.1 * current_value
                individual[i] = max(0, individual[i] + random.gauss(0, mutation_scale))
            else:
                individual[i] = max(0, individual[i] + random.gauss(0, 0.1))
    return individual

def evolve_step_function():
    """Main evolutionary algorithm to evolve step function with improved initialization"""
    # Parameters
    pop_size = 50
    n_generations = 100
    min_size = 100
    max_size = 1000

    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, size=random.randint(min_size, max_size))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with more structured individuals
    population = toolbox.population(n=pop_size)

    # Evolve
    best_individual = None
    best_fitness = 0

    for generation in range(n_generations):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Track best
        for ind in population:
            if ind.fitness.values[0] > best_fitness and len(ind) > 0:
                best_fitness = ind.fitness.values[0]
                best_individual = list(ind)

        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace old population
        population[:] = offspring

    return best_individual if best_individual is not None else []

def optimize_peak_parameters(best_function, n_steps):
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
            try:
                # Try several small adjustments
                adjusted_func = temp_func.copy()
                for i in range(min(len(params), len(adjusted_func))):
                    if i < len(adjusted_func):
                        adjusted_func[i] = max(0.0, adjusted_func[i] * (1.0 + params[i] * 0.1))
                
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
                    temp_func[i] = max(0.0, temp_func[i] * (1.0 + result.x[i] * 0.1))
            
            return temp_func.tolist()
            
    except Exception:
        pass
    
    # Fall back to original if optimization fails
    return best_function

def compute_c2(f_values):
    """Compute C2 value for given function"""
    if not f_values:
        return 0.0

    # Convert to numpy array
    f = np.array(f_values, dtype=np.float64)
    
    # Early exit for invalid arrays
    if np.isnan(f).any() or np.isinf(f).any():
        return 0.0

    # Compute autoconvolution g = f * f
    try:
        g = signal.convolve(f, f, mode='full')
    except Exception:
        return 0.0

    # Extract central portion (valid autoconvolution)
    half_len = len(f) - 1
    if len(g) >= half_len:
        g = g[half_len:]  # Take right half
    else:
        return 0.0

    # Compute norms with numerical stability checks
    try:
        g_squared = g * g
        norm_2_sq = np.sum(g_squared)
        
        norm_1 = np.sum(np.abs(g))
        norm_inf = np.max(np.abs(g))
        
        # Check for numerical stability
        if np.isnan(norm_2_sq) or np.isnan(norm_1) or np.isnan(norm_inf):
            return 0.0
            
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return float(c2) if not np.isnan(c2) and not np.isinf(c2) else 0.0
    except Exception:
        return 0.0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Try multiple approaches to find good solution
    best_result = []
    best_c2 = 0

    # Approach 1: Evolutionary algorithm
    try:
        evolved_result = evolve_step_function()
        if evolved_result:
            # Evaluate evolved result
            f_vals = np.array(evolved_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)

                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = evolved_result
    except Exception as e:
        pass

    # If no good result from evolution, fallback to a more informed approach
    if len(best_result) == 0:
        # Use a heuristic approach with more structured sampling
        n_steps = 500  # Fixed size for consistency
        # Create a step function that balances peaks and flat regions
        # This is a simplified version but more principled than pure random
        f_values = np.random.gamma(2, 2, n_steps)  # Gamma distribution gives positive values
        f_values = f_values / np.max(f_values) * 2  # Scale to reasonable range
        f_values = np.maximum(f_values, 0)

        # Apply some smoothing to reduce extreme variations
        f_values = signal.savgol_filter(f_values, min(51, len(f_values)-1), 3) if len(f_values) > 50 else f_values
        f_values = np.maximum(f_values, 0)

        best_result = f_values.tolist()

    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")