# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time
from sklearn.preprocessing import StandardScaler
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution efficiently using numba"""
    n = len(f_vals)
    # Create output array for autoconvolution
    g = np.zeros(2*n - 1)

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

def create_structured_initial_population(pop_size, min_size, max_size):
    """Create structured initial population with improved initialization strategies"""
    population = []
    for _ in range(pop_size):
        size = random.randint(min_size, max_size)
        # Use step function construction with geometric properties
        f_vals = construct_geometric_step_function(size)
        f_vals = np.maximum(f_vals, 0)
        # Apply mild smoothing to reduce extremes
        if len(f_vals) > 50:
            f_vals = signal.savgol_filter(f_vals, min(51, len(f_vals)-1), 3)
        f_vals = np.maximum(f_vals, 0)
        population.append(f_vals.tolist())
    return population

def construct_geometric_step_function(n_steps):
    """Construct a step function using optimized Gaussian peak placement for better C2"""
    # Create a more structured approach with properly spaced Gaussian peaks
    # This aims to avoid destructive interference while creating favorable autoconvolution profiles

    # Determine number of peaks based on function length
    n_peaks = max(2, min(12, n_steps // 100))

    # Create evenly spaced base positions but add some randomness for diversity
    base_positions = np.linspace(0.1 * n_steps, 0.9 * n_steps, n_peaks)

    # Add small random perturbations to positions
    for i in range(len(base_positions)):
        if i > 0 and i < len(base_positions) - 1:
            # Keep interior peaks somewhat spaced
            max_perturbation = max(10, n_steps // 50)
            base_positions[i] += random.uniform(-max_perturbation, max_perturbation)
        elif i == 0:
            # First peak near left boundary with some variation
            base_positions[i] += random.uniform(0, n_steps // 20)
        else:
            # Last peak near right boundary with some variation
            base_positions[i] += random.uniform(-n_steps // 20, 0)

    # Ensure positions are within bounds
    base_positions = np.clip(base_positions, 0, n_steps - 1)

    # Create the function with Gaussian peaks
    f_vals = np.zeros(n_steps)

    # For each peak, generate different parameters to create variety
    for i, center in enumerate(base_positions):
        # Width inversely related to peak importance for balance
        width = max(5, min(n_steps // 10, random.randint(n_steps // 50, n_steps // 20)))

        # Height with controlled variation
        height = random.uniform(0.8, 2.5)

        # Generate Gaussian curve
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian

    # Apply smoothing to reduce extreme variations and prevent numerical issues
    if n_steps > 50:
        # Use Savitzky-Golay filter with appropriate parameters
        window_length = min(51, n_steps - 1)
        if window_length % 2 == 0:
            window_length -= 1  # Must be odd
        if window_length > 1:
            f_vals = signal.savgol_filter(f_vals, window_length, 3)

    # Ensure non-negativity
    f_vals = np.maximum(f_vals, 0)

    # Add some additional structure to make the function more interesting for optimization
    # Apply a gentle envelope to create a more balanced profile
    envelope = 0.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, n_steps))
    f_vals = f_vals * envelope

    # Normalize to reasonable range
    if np.max(f_vals) > 0:
        f_vals = f_vals / np.max(f_vals) * 2.0

    return f_vals

def adaptive_evolution_phase(initial_pop, pop_size, n_generations):
    """Perform evolutionary optimization with adaptive parameters and enhanced operators"""
    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define operators
    def create_individual():
        size = random.randint(200, 800)  # Adaptive size
        f_vals = construct_geometric_step_function(size)
        f_vals = np.maximum(f_vals, 0)
        if len(f_vals) > 50:
            f_vals = signal.savgol_filter(f_vals, min(51, len(f_vals)-1), 3)
        f_vals = np.maximum(f_vals, 0)
        return f_vals.tolist()

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)

    # Enhanced crossover operator that preserves some structure
    def enhanced_crossover(ind1, ind2):
        size = min(len(ind1), len(ind2))
        point1 = random.randint(1, size//2)
        point2 = random.randint(size//2, size-1)
        if point1 > point2:
            point1, point2 = point2, point1

        # Mix the parts
        child1 = ind1[:point1] + ind2[point1:point2] + ind1[point2:]
        child2 = ind2[:point1] + ind1[point1:point2] + ind2[point2:]

        # Ensure lengths match
        child1 = child1[:len(ind1)]
        child2 = child2[:len(ind2)]

        # Pad shorter ones
        while len(child1) < len(ind1):
            child1.append(random.uniform(0, 1))
        while len(child2) < len(ind2):
            child2.append(random.uniform(0, 1))

        return child1, child2

    toolbox.register("mate", enhanced_crossover)

    # Enhanced mutation operator
    def enhanced_mutation(individual):
        for i in range(len(individual)):
            if random.random() < 0.1:  # 10% mutation rate
                # Use adaptive mutation strength
                if random.random() < 0.7:
                    # Small perturbation
                    individual[i] = max(0, individual[i] + random.gauss(0, 0.05 * max(1, individual[i])))
                else:
                    # Larger adjustment
                    individual[i] = max(0, individual[i] * random.uniform(0.8, 1.2))
        return individual

    toolbox.register("mutate", enhanced_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with diverse individuals
    population = initial_pop if initial_pop else toolbox.population(n=pop_size)

    # Evolve
    best_individual = None
    best_fitness = 0
    best_generation = 0

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
                best_generation = generation

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

        # Early termination if no improvement for several generations
        if generation - best_generation > 20:
            break

    return best_individual if best_individual is not None else []

def local_refinement(individual, max_iterations=50):
    """Refine solution using gradient-free optimization with multiple strategies"""
    if not individual:
        return individual

    # Strategy 1: Optuna-based refinement
    try:
        def objective(trial):
            # Create a slightly modified version of the individual
            modified = individual.copy()
            for i in range(len(modified)):
                # Apply small adjustments to improve C2
                adjustment_factor = trial.suggest_float(f'multiplier_{i}', 0.95, 1.05)
                modified[i] = max(0, modified[i] * adjustment_factor)

            # Ensure non-negative values
            modified = [max(0, x) for x in modified]
            return evaluate_individual(modified)[0]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(20, max_iterations))
        optuna_best = study.best_params

        # Apply the best found modifications
        refined = individual.copy()
        for i in range(len(refined)):
            if f'multiplier_{i}' in optuna_best:
                refined[i] = max(0, refined[i] * optuna_best[f'multiplier_{i}'])

        if evaluate_individual(refined)[0] > evaluate_individual(individual)[0]:
            return refined

    except:
        pass

    # Strategy 2: Direct optimization of a few selected parameters
    try:
        # Sample a subset of indices to optimize
        sample_size = min(20, len(individual) // 5)
        indices_to_try = random.sample(range(len(individual)), sample_size)

        best_individual = individual.copy()
        best_score = evaluate_individual(best_individual)[0]

        for _ in range(10):  # Limited iterations
            # Try changing a few values
            test_individual = individual.copy()
            for idx in indices_to_try:
                if random.random() < 0.5:
                    # Random adjustment
                    scale_factor = random.uniform(0.8, 1.2)
                    test_individual[idx] = max(0, test_individual[idx] * scale_factor)

            score = evaluate_individual(test_individual)[0]
            if score > best_score:
                best_score = score
                best_individual = test_individual

        if best_score > evaluate_individual(individual)[0]:
            return best_individual
    except:
        pass

    return individual

def enhanced_construct_function() -> list[float]:
    """Enhanced function to construct step-function with high C2 value."""
    start_time = time.time()

    # Phase 1: Fast initial sampling with improved structured approach
    best_result = []
    best_c2 = 0

    # Create improved structured initial population
    initial_pop = create_structured_initial_population(20, 150, 1000)

    # Phase 2: Evolutionary optimization with enhanced operators
    try:
        evolved_result = adaptive_evolution_phase(initial_pop, 30, 50)
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

    # Phase 3: Advanced local refinement with enhanced strategies
    if best_result and time.time() - start_time < 70:  # Leave time for refinement
        refined_result = local_refinement(best_result)
        # Re-evaluate to see if refinement helped
        f_vals = np.array(refined_result, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)
        if np.sum(f_vals) > 0:
            g_vals = compute_autoconvolution_numba(f_vals)
            l2_sq, l1, linf = compute_norms_numba(g_vals)
            if l1 > 1e-15 and linf > 1e-15:
                c2 = l2_sq / (l1 * linf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = refined_result

    # Phase 4: Final optimization of top results
    if best_result and time.time() - start_time < 80:
        # Apply a special optimization step for the best result
        try:
            # Try to improve by making small, systematic changes
            best_candidate = best_result.copy()
            current_score = evaluate_individual(best_candidate)[0]

            # Try to enhance the result with small adjustments
            for iteration in range(30):
                test_candidate = best_candidate.copy()
                # Select some indices to modify
                indices_to_modify = random.sample(range(len(test_candidate)),
                                                min(10, len(test_candidate)//5))

                for idx in indices_to_modify:
                    # Make small adjustments
                    if random.random() < 0.7:
                        # Slight increase/decrease
                        factor = random.uniform(0.95, 1.05)
                        test_candidate[idx] = max(0, test_candidate[idx] * factor)
                    else:
                        # Try a more significant change
                        factor = random.uniform(0.9, 1.1)
                        test_candidate[idx] = max(0, test_candidate[idx] * factor)

                new_score = evaluate_individual(test_candidate)[0]
                if new_score > current_score:
                    best_candidate = test_candidate
                    current_score = new_score

            if evaluate_individual(best_candidate)[0] > evaluate_individual(best_result)[0]:
                best_result = best_candidate
        except:
            pass

    # Phase 5: Fallback to well-structured approach if nothing worked
    if len(best_result) == 0 or best_c2 < 0.5:
        # Use an improved heuristic approach
        n_steps = random.randint(300, 800)  # Variable size with more emphasis on larger
        # Create a more sophisticated step function
        f_values = construct_geometric_step_function(n_steps)
        f_values = np.maximum(f_values, 0)

        # Apply smoothing
        if len(f_values) > 50:
            f_values = signal.savgol_filter(f_values, min(51, len(f_values)-1), 3)
        f_values = np.maximum(f_values, 0)

        best_result = f_values.tolist()

    # Final evaluation and time management
    if best_result:
        try:
            f_vals = np.array(best_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                if l1 > 1e-15 and linf > 1e-15:
                    final_c2 = l2_sq / (l1 * linf)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
        except:
            pass

    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed

    return best_result

# Alias the enhanced function to the expected name
construct_function = enhanced_construct_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")