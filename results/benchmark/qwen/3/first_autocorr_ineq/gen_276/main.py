# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import random
import time
from scipy.fft import fft, ifft
from collections import defaultdict, OrderedDict
import heapq
from scipy.optimize import linprog

def compute_autocorrelation_constant(sequence):
    """
    Compute C₁ for a given sequence using FFT for efficiency.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    """
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Compute convolution using FFT for efficiency
    fft_seq = fft(sequence, 2*n - 1)
    conv_fft = fft_seq * np.conj(fft_seq)
    conv = ifft(conv_fft).real[:2*n-1]
    max_conv = np.max(conv)

    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def solve_convolution_lp_advanced(f_sequence, rhs, n):
    """
    Advanced LP solver with regularization and stability enhancements.
    """
    try:
        # Create constraint matrix efficiently
        a_ub = np.zeros((2 * n - 1, n))
        b_ub = np.full(2 * n - 1, rhs)

        # Vectorized constraint generation
        for k in range(2 * n - 1):
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    a_ub[k, j] = f_sequence[i]

        # Add regularization to improve numerical stability
        reg_factor = 1e-8
        a_ub_reg = a_ub + reg_factor * np.random.randn(*a_ub.shape) * 1e-10
        a_ub = a_ub_reg

        # Non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Objective: maximize sum of variables (equivalent to minimizing -sum)
        c = -np.ones(n)

        # Solve with multiple strategies for robustness
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                        options={'presolve': True, 'time_limit': 3})

        if result.success:
            return result.x
        else:
            # Attempt fallback with simplex method
            try:
                result = linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex')
                if result.success:
                    return result.x
            except:
                pass
            return None

    except Exception:
        return None

def evaluate_objective(sequence):
    """
    Evaluate the objective function: -1/C₁ (we minimize this to maximize 1/C₁)
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return float('inf')  # Invalid solution
    return -1.0 / c1  # Negative because we want to maximize 1/C₁

def evaluate_sequence_with_cache(sequence, cache):
    """
    Evaluate sequence with caching to avoid redundant computations.
    """
    key = tuple(sequence)
    if key in cache:
        return cache[key]

    result = evaluate_objective(sequence)
    cache[key] = result
    return result

def get_good_direction_curvature_aware(sequence, iteration=0):
    """
    Compute a curvature-aware direction for movement using second-order information.
    """
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None

    # Normalize sequence
    normalized_sequence = np.array(sequence) / sum_sequence

    try:
        # Compute convolution
        conv = fftconvolve(normalized_sequence, normalized_sequence, mode='full')[:2*n-1]
        rhs = np.max(conv)

        # Solve LP optimization
        g_fun = solve_convolution_lp_advanced(normalized_sequence, rhs, n)

        if g_fun is None or np.any(np.isnan(g_fun)):
            return None

        # Normalize the resulting sequence
        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            return None

        normalized_g_fun = g_fun / sum_g

        # Curvature-aware adjustment using finite differences
        eps = 1e-4
        curvature_correction = np.zeros_like(normalized_g_fun)

        # Estimate curvature by finite differences
        for i in range(n):
            perturbed_seq = normalized_sequence.copy()
            perturbed_seq[i] += eps

            # Recompute convolution for perturbed sequence
            conv_perturbed = fftconvolve(perturbed_seq, perturbed_seq, mode='full')[:2*n-1]
            max_conv_perturbed = np.max(conv_perturbed)

            # Estimate second derivative
            second_deriv = (max_conv_perturbed - np.max(conv)) / (eps ** 2)
            curvature_correction[i] = max(0, second_deriv)

        # Apply curvature correction to direction
        curvature_adjusted = normalized_g_fun + 0.1 * curvature_correction
        curvature_adjusted = np.maximum(curvature_adjusted, 0)

        # Adaptive step size
        base_t = 0.05
        dynamic_t = base_t * (1 - min(iteration / 100, 0.9))

        new_sequence = [
            (1 - dynamic_t) * x + dynamic_t * y
            for x, y in zip(sequence, curvature_adjusted)
        ]

        return new_sequence

    except Exception:
        return None

def generate_initial_sequence():
    """
    Generate a good initial random sequence with more structure.
    Uses inverse C1 weighting to guide initialization towards promising regions.
    """
    n = random.randint(100, 1000)

    # Initialize with a balanced approach
    # Try to create sequences that are likely to give good 1/C1 values
    if random.random() < 0.4:
        # Gaussian-like distribution centered around typical values
        center = n // 2
        sequence = [np.exp(-0.5 * ((i - center)**2) / (n/10)**2) for i in range(n)]
        sequence = [x * 100.0 for x in sequence]  # Scale up
    elif random.random() < 0.6:
        # Exponential decay
        sequence = [0.9 ** i for i in range(n)]
        sequence = [x * 100.0 for x in sequence]
    else:
        # Random with inverse C1 weighting influence
        sequence = []
        for i in range(n):
            # Use a distribution biased towards values that might decrease C1
            if random.random() < 0.5:
                val = random.uniform(0.1, 50.0)
            else:
                val = random.uniform(50.0, 1000.0)
            sequence.append(val)

    # Normalize to ensure sum is reasonable
    sum_val = np.sum(sequence)
    if sum_val < 0.01:
        sequence = [x + 0.1 for x in sequence]

    return sequence

def generate_population(size, min_size=100, max_size=1000):
    """Generate a population of sequences."""
    population = []
    for _ in range(size):
        n = random.randint(min_size, max_size)
        sequence = generate_initial_sequence()
        population.append(sequence)
    return population

def quadratic_optimization_step(current_seq, max_iter=100):
    """
    Perform a quadratic optimization step to improve the sequence.
    """
    n = len(current_seq)
    # Define bounds: all elements must be in [0, 1000]
    bounds = [(0.0, 1000.0) for _ in range(n)]

    # Define constraints
    def sum_constraint(x):
        return np.sum(x) - 0.01  # Require sum >= 0.01

    constraints = [{'type': 'ineq', 'fun': sum_constraint}]

    # Objective function to minimize
    def objective(x):
        return evaluate_objective(x)

    # Try multiple optimization methods
    methods_to_try = ['SLSQP', 'L-BFGS-B']

    for method in methods_to_try:
        try:
            # Use smaller tolerance for faster convergence
            result = minimize(objective, current_seq, method=method, bounds=bounds,
                            constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
            if result.success:
                return result.x.tolist()
        except:
            continue

    # If all methods fail, return the original sequence slightly perturbed
    perturbed = [max(0.0, x + random.gauss(0, 0.05)) for x in current_seq]
    if np.sum(perturbed) < 0.01:
        perturbed[0] = max(0.0, perturbed[0] + 0.01)
    return perturbed

def mutate_sequence(sequence, mutation_rate=0.1):
    """Mutate a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    # Adaptive mutation rate: lower for longer sequences
    if len(sequence) > 500:
        mutation_rate *= 0.5
    elif len(sequence) < 200:
        mutation_rate *= 1.5

    # Calculate standard deviation for mutation scaling
    std_dev = np.std(sequence) if len(sequence) > 0 else 1.0
    mutation_scale = max(0.1, std_dev * 0.1)  # Scale mutation by sequence variability

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate with Gaussian noise scaled by sequence variability
            mutated[i] = max(0.0, mutated[i] + random.gauss(0, mutation_scale))
    return mutated

def crossover_sequences(parent1, parent2):
    """Perform crossover between two sequences with adaptive mixing."""
    # Use a blend crossover that considers characteristics of parents
    n1, n2 = len(parent1), len(parent2)
    min_len = min(n1, n2)

    # Create offspring with blended elements
    offspring = []

    # Determine if we're doing crossover or just taking one parent
    if random.random() < 0.7:  # 70% chance of crossover
        # Blend elements with weight based on parent characteristics
        for i in range(max(n1, n2)):
            if i < min_len:
                # Blend based on similarity of elements
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
            elif i < n1:
                # Extending beyond shorter parent
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
    else:
        # Just take one parent with some variation
        parent = parent1 if random.random() < 0.5 else parent2
        offspring = parent.copy()

    # Ensure all elements are non-negative
    offspring = [max(0.0, x) for x in offspring]
    return offspring

def diversify_population(population, diversity_threshold=0.05):
    """
    Introduce diversity by adding random sequences when population becomes too homogeneous.
    Uses entropy-based measure for better diversity detection.
    """
    if len(population) < 2:
        return population

    # Calculate entropy-based diversity metric
    avg_values = [np.mean(seq) for seq in population]
    std_devs = [np.std(seq) for seq in population]

    # Combined diversity measure (entropy + variability)
    diversity_metrics = [np.std(avg_values) / (np.mean(avg_values) + 1e-8) +
                        np.mean(std_devs) / (np.mean(avg_values) + 1e-8)]

    avg_diversity = np.mean(diversity_metrics) if diversity_metrics else 0

    if avg_diversity < diversity_threshold:
        # Add some random sequences to maintain diversity
        num_new = max(1, len(population) // 4)  # Add 25% random sequences
        for _ in range(num_new):
            new_seq = generate_initial_sequence()
            population.append(new_seq)

    return population

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence using an enhanced evolutionary approach.
    """
    start_time = time.time()
    population_size = 30
    generations = 60
    keep_top = 8
    elite_preservation = 2
    diversity_check_interval = 10

    # Cache for storing previously computed evaluations
    evaluation_cache = {}

    # Generate initial population
    population = generate_population(population_size)

    # Evaluate initial population
    fitness_scores = []
    for seq in population:
        fitness = evaluate_sequence_with_cache(seq, evaluation_cache)
        fitness_scores.append((seq, fitness))

    # Sort population by fitness (lower is better)
    fitness_scores.sort(key=lambda x: x[1])

    # Track best solution globally
    global_best = fitness_scores[0][0]
    global_best_fitness = fitness_scores[0][1]

    # Main evolution loop
    for gen in range(generations):
        if time.time() - start_time > 170:  # Leave 10 seconds for finalization
            break

        # Periodic diversity check
        if gen % diversity_check_interval == 0:
            population = diversify_population(population)

        # Keep top performers (elite)
        top_performers = [seq for seq, _ in fitness_scores[:keep_top]]

        # Create new population
        new_population = top_performers[:]

        # Preserve elites
        if elite_preservation > 0:
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[:elite_preservation]
            elites = [fitness_scores[i][0] for i in elite_indices]
            new_population.extend(elites)

        # Add mutated versions of top performers
        for i in range(population_size - len(new_population)):
            if random.random() < 0.7:  # 70% chance of mutation
                parent = random.choice(top_performers)
                child = mutate_sequence(parent)
            else:  # 30% chance of crossover
                p1, p2 = random.sample(top_performers, 2)
                child = crossover_sequences(p1, p2)

            new_population.append(child)

        # Apply enhanced local optimization to some individuals
        for i in range(0, len(new_population), 2):
            if random.random() < 0.6:  # 60% chance of local optimization
                # Use curvature-aware optimization for better convergence
                new_population[i] = get_good_direction_curvature_aware(new_population[i], iteration=gen)

        # Evaluate new population
        fitness_scores = []
        for seq in new_population:
            fitness = evaluate_sequence_with_cache(seq, evaluation_cache)
            fitness_scores.append((seq, fitness))

        # Sort population by fitness
        fitness_scores.sort(key=lambda x: x[1])

        # Update global best
        if fitness_scores[0][1] < global_best_fitness:
            global_best = fitness_scores[0][0]
            global_best_fitness = fitness_scores[0][1]

        # Occasionally introduce completely new sequences to avoid local optima
        if gen % 15 == 0 and gen > 0:
            # Replace worst performers with new random sequences
            worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[-population_size//4:]
            for idx in worst_indices:
                new_seq = generate_initial_sequence()
                fitness_scores[idx] = (new_seq, evaluate_objective(new_seq))

    # Final optimization of the best sequence
    final_best = quadratic_optimization_step(global_best, max_iter=200)

    # Return the best sequence found
    return final_best

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")