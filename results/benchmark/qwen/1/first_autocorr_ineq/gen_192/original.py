# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from collections import deque
import warnings

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better efficiency"""
    n = len(seq)
    if n == 0:
        return np.array([])
    # Pad to size 2*n-1 for full convolution
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    # Compute FFT-based convolution
    fft_seq = fft(padded_seq)
    conv_result = ifft(fft_seq * np.conj(fft_seq)).real[:2*n-1]
    return conv_result

def compute_autocorrelation_constant(sequence):
    """Compute the C1 constant for a given sequence"""
    if len(sequence) == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    # Use FFT for efficient convolution
    conv_result = compute_convolution_fft(sequence)
    max_conv = np.max(conv_result)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_a ** 2)

    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing its inverse C1"""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequence
    return 1.0 / c1  # Return 1/C1 as the objective

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build constraint matrix efficiently
    # Each constraint corresponds to a convolution element
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Add non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    # Add bounds to avoid numerical issues
    bounds = [(0, 1000) for _ in range(n)]  # Clips heights to [0, 1000]

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        return None

def get_gradient_estimate(sequence, epsilon=1e-4):
    """Estimate gradient using finite differences"""
    n = len(sequence)
    if n == 0:
        return None

    grad = []
    for i in range(n):
        # Create perturbed sequences
        seq_plus = sequence.copy()
        seq_minus = sequence.copy()

        seq_plus[i] += epsilon
        seq_minus[i] -= epsilon

        # Evaluate both and estimate derivative
        val_plus = evaluate_sequence(seq_plus)
        val_minus = evaluate_sequence(seq_minus)

        grad_i = (val_plus - val_minus) / (2 * epsilon)
        grad.append(grad_i)

    return np.array(grad)

def adaptive_sequence_length():
    """Adaptively choose sequence length based on performance considerations"""
    # Sample from a log-uniform distribution to explore various sizes
    base_length = 500
    return int(np.random.lognormal(np.log(base_length), 0.5))

def get_better_direction(sequence):
    """Generate a better sequence direction using a hybrid approach."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Normalize the sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution and RHS for LP
    conv_result = compute_convolution_fft(normalized_sequence)
    rhs = np.max(conv_result)

    # Solve LP to get improved direction
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    # Normalize the resulting sequence
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Use weighted average with gradient-based step
    grad = get_gradient_estimate(sequence)
    if grad is not None:
        # Combine gradient information with LP solution
        t = 0.1  # Blend factor
        new_sequence = [
            (1-t)*x + t*y + 0.01*grad[i] for i, (x, y) in enumerate(zip(sequence, normalized_g_fun))
        ]
    else:
        new_sequence = [
            (1-0.05)*x + 0.05*y for x, y in zip(sequence, normalized_g_fun)
        ]

    # Ensure non-negativity and clipping
    new_sequence = [max(0, min(x, 1000)) for x in new_sequence]

    return new_sequence

def local_search_improvement(initial_seq, max_iter=50, use_cached=False):
    """Perform local search improvements around a sequence."""
    current_seq = initial_seq.copy()
    current_score = evaluate_sequence(current_seq)

    if use_cached:
        # Use cached evaluations to avoid recomputation
        cached_evaluations = {}
        def cached_evaluate(seq):
            seq_tuple = tuple(seq)
            if seq_tuple in cached_evaluations:
                return cached_evaluations[seq_tuple]
            else:
                val = evaluate_sequence(seq)
                cached_evaluations[seq_tuple] = val
                return val
    else:
        cached_evaluate = evaluate_sequence

    for _ in range(max_iter):
        # Get a better direction
        better_dir = get_better_direction(current_seq)
        if better_dir is None:
            break

        # Try multiple steps
        for step_size in [0.05, 0.1, 0.2]:
            candidate_seq = [
                max(0, min(x + step_size * (y-x), 1000))
                for x, y in zip(current_seq, better_dir)
            ]

            candidate_score = cached_evaluate(candidate_seq)
            if candidate_score > current_score:
                current_seq = candidate_seq
                current_score = candidate_score
                break  # Accept the improvement

    return current_seq

def generate_structured_sequence(n, pattern_type="sine"):
    """Generate a structured sequence to improve convergence."""
    if pattern_type == "sine":
        # Sine wave pattern
        base_seq = [np.sin(i * np.pi / n) for i in range(n)]
    elif pattern_type == "gaussian":
        # Gaussian-like pattern
        center = n // 2
        std_dev = n / 6
        base_seq = [np.exp(-((i - center)**2) / (2 * std_dev**2)) for i in range(n)]
    else:
        # Default to linear decay
        base_seq = [max(0.001, 1.0 - i/n) for i in range(n)]

    # Normalize to ensure positive values and reasonable scale
    base_seq = [max(0.001, x + 1.0) for x in base_seq]
    # Scale to have reasonable magnitude
    total = sum(base_seq)
    if total > 0:
        base_seq = [x * 100 / total for x in base_seq]
    return base_seq

def evolve_sequence(population_size, generations, stagnation_threshold,
                   best_sequences_history, use_elite=True):
    """Evolve a population of sequences to find optimal ones."""
    # Initial population
    population = []
    elite_individual = None
    elite_fitness = 0

    for _ in range(population_size):
        n = adaptive_sequence_length()
        # Use structured initialization for some individuals
        if random.random() < 0.4 and len(best_sequences_history) > 0:
            # Use a learned structure from history
            seq = generate_structured_sequence(n, random.choice(["sine", "gaussian"]))
        else:
            seq = [np.random.random() * 100 for _ in range(n)]
        population.append(seq)

    # Track best individual
    best_fitness = 0
    best_individual = None
    stagnation_count = 0

    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append(fitness)

        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Check for stagnation
        if stagnation_count > stagnation_threshold:
            break

        # Create new population
        new_population = []

        # Keep top performers if needed
        if use_elite:
            new_population.append(best_individual)

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = population[random.choice(range(min(len(population), population_size//2)))]
            parent2 = population[random.choice(range(min(len(population), population_size//2)))]

            # Crossover (simple averaging)
            child = [(a+b)/2 for a,b in zip(parent1, parent2)]

            # Mutation
            if random.random() < 0.7:
                # Add slight mutation
                for i in range(len(child)):
                    if random.random() < 0.1:
                        child[i] *= random.uniform(0.9, 1.1)

            # Ensure non-negative values
            child = [max(0, x) for x in child]

            # Clip to reasonable values
            child = [min(x, 1000) for x in child]

            # Make sure it's valid
            if sum(child) > 0.01:
                new_population.append(child)

        population = new_population[:population_size]

    return best_individual, best_fitness

def adaptive_evolution_strategy(max_time=180):
    """Adaptive evolutionary strategy with dynamic parameters."""
    start_time = time.time()

    # History to store recent best scores for early stopping
    recent_scores = deque(maxlen=10)
    best_sequences_history = deque(maxlen=5)  # Store top sequences from evolution

    best_sequence = None
    best_score = 0.0
    best_c1 = float('inf')

    iteration = 0
    max_iterations = 5000

    # Phase 1: Multi-start with varying sequence lengths and structured initializations
    while iteration < max_iterations and time.time() - start_time < max_time:
        # Adaptive sequence length sampling
        n = adaptive_sequence_length()

        # Choose initialization type based on iteration
        init_types = ["uniform", "sine", "gaussian"]
        init_type = random.choice(init_types)
        sequence = generate_structured_sequence(n, init_type)

        # Local improvement (using cached evaluations)
        improved_seq = local_search_improvement(sequence, max_iter=30, use_cached=True)

        score = evaluate_sequence(improved_seq)
        c1 = compute_autocorrelation_constant(improved_seq)

        if score > best_score:
            best_score = score
            best_sequence = improved_seq.copy()
            best_c1 = c1
            print(f"Iteration {iteration}: New best score = {score:.6f}, C1 = {c1:.6f}")

            # Check benchmark
            benchmark_ratio = 1.5031 / c1
            if benchmark_ratio > 1.0:
                print(f"BEAT BENCHMARK at iteration {iteration}! Ratio = {benchmark_ratio:.6f}")
                break

        # Store recent scores for convergence detection
        recent_scores.append(score)

        # Store good sequences for history
        if len(best_sequences_history) < best_sequences_history.maxlen or score > min(best_sequences_history, key=lambda x: x[1])[1]:
            if len(best_sequences_history) == best_sequences_history.maxlen:
                # Remove worst
                worst_entry = min(best_sequences_history, key=lambda x: x[1])
                best_sequences_history.remove(worst_entry)
            best_sequences_history.append((improved_seq.copy(), score))

        # Early stopping if scores are not improving
        if len(recent_scores) == recent_scores.maxlen:
            if abs(max(recent_scores) - min(recent_scores)) < 1e-6:
                print(f"Early stopping at iteration {iteration}")
                break

        iteration += 1

    # Phase 2: Evolutionary search to refine further with dynamic population size
    if best_sequence is not None and time.time() - start_time < max_time - 10:
        try:
            # Dynamic population size based on iteration
            pop_size = min(100, 30 + iteration // 100)
            gens = min(20, 10 + iteration // 100)

            current_sequence, current_fitness = evolve_sequence(
                population_size=pop_size,
                generations=gens,
                stagnation_threshold=5,
                best_sequences_history=best_sequences_history,
                use_elite=True
            )

            if current_fitness > best_score:
                best_score = current_fitness
                best_sequence = current_sequence
                best_c1 = compute_autocorrelation_constant(best_sequence)
                print(f"Evolution phase improved score to {best_score:.6f}, C1 = {best_c1:.6f}")

                # Check benchmark
                benchmark_ratio = 1.5031 / best_c1
                if benchmark_ratio > 1.0:
                    print(f"BEAT BENCHMARK at evolution stage! Ratio = {benchmark_ratio:.6f}")

        except Exception as e:
            pass  # Ignore errors in evolution step

    # Final validation
    if best_sequence is not None:
        final_score = evaluate_sequence(best_sequence)
        final_c1 = compute_autocorrelation_constant(best_sequence)
        benchmark_ratio = 1.5031 / final_c1

        print(f"Final result:")
        print(f"  Score: {final_score:.6f}")
        print(f"  C1: {final_c1:.6f}")
        print(f"  Benchmark ratio: {benchmark_ratio:.6f}")
        print(f"  Iterations: {iteration}")
    else:
        # Fallback if no good sequence found
        n = adaptive_sequence_length()
        best_sequence = [np.random.random() * 10 for _ in range(n)]
        final_score = evaluate_sequence(best_sequence)
        final_c1 = compute_autocorrelation_constant(best_sequence)
        benchmark_ratio = 1.5031 / final_c1

    return best_sequence

def search_for_best_sequence(max_time=180) -> list[float]:
    """Main function to search for the best coefficient sequence."""
    return adaptive_evolution_strategy(max_time)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")