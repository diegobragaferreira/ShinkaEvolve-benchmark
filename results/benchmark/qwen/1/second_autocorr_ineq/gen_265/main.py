# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, njit
import random
import time
from scipy.spatial.distance import pdist, squareform

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.8
CROSSOVER_PROB = 0.7
NUM_STARTS = 5
MAX_EVALUATIONS = 10000
MIN_POP_SIZE = 30
MAX_POP_SIZE = 150

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the three norms of the autoconvolution g = f*f
    Returns ||g||₂², ||g||₁, ||g||∞
    """
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution g = f*f
    # Using discrete convolution with proper indexing
    g = np.zeros(2*n - 1)

    # For each element in output convolved array
    for i in range(2*n - 1):
        # Sum over valid indices
        start_idx = max(0, i - n + 1)
        end_idx = min(i, n - 1) + 1
        sum_val = 0.0
        for j in range(start_idx, end_idx):
            k = i - j
            if 0 <= k < n:
                sum_val += f_vals[j] * f_vals[k]
        g[i] = sum_val

    # Compute norms
    g_squared = g * g
    norm_l2_sq = np.sum(g_squared)

    norm_l1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_l2_sq, norm_l1, norm_inf

@njit
def compute_c2_score(f_vals):
    """
    Compute the C2 score: ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_l2_sq, norm_l1, norm_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_l1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0

    return norm_l2_sq / (norm_l1 * norm_inf)

def multi_scale_initialization(n):
    """Create diverse initial step functions with multi-scale patterns"""
    # Strategy 1: Multi-peak pattern with varying amplitudes
    pattern1 = np.zeros(n)
    n_peaks = 3 + np.random.randint(0, 3)
    for _ in range(n_peaks):
        center = np.random.randint(0, n)
        width = max(1, n // 10 + np.random.randint(-2, 3))
        height = 0.5 + np.random.random() * 1.0
        start = max(0, center - width // 2)
        end = min(n, center + width // 2)
        pattern1[start:end] = np.maximum(pattern1[start:end], height)

    # Strategy 2: Alternating band pattern with Gaussian smoothing
    pattern2 = np.zeros(n)
    segment_size = max(1, n // 8 + np.random.randint(-2, 3))
    for i in range(0, n, segment_size):
        end_idx = min(i + segment_size, n)
        if (i // segment_size) % 2 == 0:
            pattern2[i:end_idx] = 1.0 + np.random.random(end_idx - i) * 0.5
        else:
            pattern2[i:end_idx] = 0.2 + np.random.random(end_idx - i) * 0.3

    # Strategy 3: Sine wave modulated pattern
    pattern3 = []
    freq = 5 + np.random.randint(0, 5)
    for i in range(n):
        pos = i / (n - 1) if n > 1 else 0.5
        val = 0.7 + 0.3 * np.sin(freq * np.pi * pos) + 0.2 * np.random.random()
        pattern3.append(max(0.0, val))

    # Combine patterns with different weights
    weights = [0.4, 0.35, 0.25]  # Different weights for combination
    combined_pattern = weights[0] * pattern1 + weights[1] * pattern2 + weights[2] * np.array(pattern3)

    # Normalize and ensure non-negativity
    combined_pattern = np.clip(combined_pattern, 0, None)
    if np.sum(combined_pattern) > 0:
        combined_pattern = combined_pattern / np.sum(combined_pattern) * n

    return combined_pattern.tolist()

def adaptive_tournament_selection(population, fitness_scores, tournament_size=3):
    """Perform tournament selection with adaptive tournament size"""
    selected = []
    for _ in range(len(population)):
        # Adaptive tournament size based on population diversity
        if len(population) > 50:
            actual_tournament = min(tournament_size, len(population) // 10)
        else:
            actual_tournament = tournament_size

        # Select random individuals for tournament
        tournament_indices = np.random.choice(len(population), actual_tournament, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Select the best from tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx])

    return selected

def calculate_diversity(population):
    """Calculate population diversity using pairwise distances"""
    if len(population) < 2:
        return 0.0

    # Convert to array for distance calculation
    pop_array = np.array(population)
    if len(pop_array.shape) == 1:
        pop_array = pop_array.reshape(-1, 1)

    # Calculate pairwise Euclidean distances
    distances = pdist(pop_array, metric='euclidean')
    if len(distances) > 0:
        return np.mean(distances)
    else:
        return 0.0

def evolutionary_optimization():
    """Enhanced evolutionary optimization with adaptive parameters and multiple refinement strategies"""
    best_c2 = 0.0
    best_f = None

    # Track convergence
    prev_best_c2 = -np.inf
    no_improvement_count = 0
    max_no_improvement = 10

    # Try multiple random starts with adaptive population sizes
    for start_num in range(NUM_STARTS):
        # Adaptive population size based on iteration
        pop_size = MIN_POP_SIZE + (MAX_POP_SIZE - MIN_POP_SIZE) * (start_num / NUM_STARTS)
        pop_size = int(max(MIN_POP_SIZE, min(MAX_POP_SIZE, pop_size)))

        # Initialize population with multi-scale patterns
        population = []
        for i in range(pop_size):
            n = random.randint(500, 2000)  # Larger range for better resolution
            individual = multi_scale_initialization(n)
            # Ensure non-negative values
            individual = [max(0.0, x) for x in individual]
            population.append(individual)

        # Track fitness scores
        fitness_scores = []
        for individual in population:
            try:
                c2 = compute_c2_score(individual)
                fitness_scores.append(c2)
            except:
                fitness_scores.append(0.0)

        # Evolution loop with adaptive parameters
        current_pop_size = pop_size
        current_generations = GENERATIONS

        for generation in range(current_generations):
            # Adaptive parameter adjustments
            if no_improvement_count > max_no_improvement:
                # Reduce population size or increase mutation rate
                if current_pop_size > MIN_POP_SIZE:
                    current_pop_size = max(MIN_POP_SIZE, current_pop_size - 10)
                if MUTATION_RATE < 1.0:
                    MUTATION_RATE = min(1.0, MUTATION_RATE + 0.05)
                no_improvement_count = 0

            # Selection
            selected_population = adaptive_tournament_selection(population, fitness_scores)

            # Create offspring through crossover and mutation
            offspring = []
            while len(offspring) < current_pop_size:
                # Tournament selection for parents
                parent1_idx = np.random.randint(0, len(selected_population))
                parent2_idx = np.random.randint(0, len(selected_population))

                parent1 = selected_population[parent1_idx]
                parent2 = selected_population[parent2_idx]

                # Crossover (uniform)
                child = []
                for j in range(len(parent1)):
                    if random.random() < CROSSOVER_PROB:
                        child.append(parent1[j])
                    else:
                        child.append(parent2[j])

                # Mutation with adaptive rate
                mutation_rate = 0.1 + 0.05 * (1.0 - generation / current_generations)
                for j in range(len(child)):
                    if random.random() < mutation_rate:
                        # Gaussian mutation with adaptive intensity
                        intensity = 0.1 * (1.0 - generation / current_generations)
                        delta = np.random.normal(0, intensity)
                        child[j] = max(0.0, child[j] + delta)

                offspring.append(child)

            # Trim to correct population size
            population = offspring[:current_pop_size]

            # Evaluate new population
            fitness_scores = []
            for individual in population:
                try:
                    c2 = compute_c2_score(individual)
                    fitness_scores.append(c2)
                except:
                    fitness_scores.append(0.0)

            # Check for improvement
            current_best = max(fitness_scores)
            if current_best > prev_best_c2:
                prev_best_c2 = current_best
                no_improvement_count = 0

                # Find best solution in this generation
                best_idx = np.argmax(fitness_scores)
                current_best_solution = population[best_idx]

                if current_best > best_c2:
                    best_c2 = current_best
                    best_f = current_best_solution.copy()
            else:
                no_improvement_count += 1

            # Early stopping if convergence
            if no_improvement_count > max_no_improvement * 2:
                break

        # Early stopping if we have a good solution
        if best_c2 > 0.95:  # Good enough to stop early
            break

    # Apply advanced refinement strategies if we found a solution
    if best_f is not None and best_c2 > 0:
        # Strategy 1: Local refinement with multiple methods
        refined_solution = best_f.copy()
        refined_c2 = best_c2

        # Local refinement with L-BFGS-B
        try:
            bounds = [(0.0, 10.0) for _ in range(len(best_f))]

            def local_objective(x):
                # Ensure non-negativity
                x = [max(0.0, xi) for xi in x]
                return -compute_c2_score(x)

            local_result = minimize(
                local_objective,
                best_f,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200}
            )

            if local_result.success:
                refined_candidate = [max(0.0, float(x)) for x in local_result.x]
                new_c2 = compute_c2_score(refined_candidate)
                if new_c2 > refined_c2:
                    refined_c2 = new_c2
                    refined_solution = refined_candidate

        except Exception as e:
            pass

        # Strategy 2: Add small random perturbations to escape local optima
        try:
            perturbed_solution = refined_solution.copy()
            noise_std = 0.01 * np.std(refined_solution)
            for i in range(len(perturbed_solution)):
                if random.random() < 0.3:  # 30% chance to perturb
                    noise = np.random.normal(0, noise_std)
                    perturbed_solution[i] = max(0.0, perturbed_solution[i] + noise)

            # Normalize
            if np.sum(perturbed_solution) > 0:
                perturbed_solution = [x / np.sum(perturbed_solution) * len(perturbed_solution)
                                    for x in perturbed_solution]

            new_c2 = compute_c2_score(perturbed_solution)
            if new_c2 > refined_c2:
                refined_c2 = new_c2
                refined_solution = perturbed_solution

        except Exception as e:
            pass

        best_f = refined_solution
        best_c2 = refined_c2

    # Final fallback
    if best_f is None:
        best_f = multi_scale_initialization(1000)

    return best_f

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run the optimization
    start_time = time.time()
    try:
        f_values = evolutionary_optimization()
        elapsed_time = time.time() - start_time
        # Add safety check to ensure valid output
        if f_values is None or len(f_values) == 0:
            # Fallback to simple uniform distribution
            f_values = [0.5] * 200
    except:
        # Final fallback
        f_values = [0.5] * 200

    # Ensure we return a reasonable-sized list
    if len(f_values) < 50:
        f_values = f_values + [0.5] * (50 - len(f_values))
    elif len(f_values) > 10000:
        f_values = f_values[:10000]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")