# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better efficiency"""
    n = len(seq)
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

def get_good_direction_to_move_into(sequence):
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Normalize the sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute the target value for the LP constraint
    conv_result = compute_convolution_fft(normalized_sequence)
    rhs = np.max(conv_result)

    # Solve the LP problem
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    # Normalize the resulting sequence
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Apply gradient descent-like update with momentum
    t = 0.05  # Increased learning rate for faster convergence
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    # Ensure non-negativity and clipping
    new_sequence = [max(0, min(x, 1000)) for x in new_sequence]

    return new_sequence

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
            print(f'LP optimization failed with status: {result.status}')
            return None
    except Exception as e:
        print(f'LP optimization error: {e}')
        return None

def adaptive_sequence_length():
    """Adaptively choose sequence length based on performance considerations"""
    # Start with a reasonable base length and adapt based on results
    base_length = 500
    return np.random.randint(base_length // 2, base_length * 2)

def generate_structured_sequence(n):
    """Generate a structured sequence to improve convergence."""
    # Try to create sequences that are likely to perform well
    # Using sine wave pattern for initial structure
    base_seq = [np.sin(i * np.pi / n) for i in range(n)] 
    # Normalize to ensure positive values and reasonable scale
    base_seq = [max(0.001, x + 1.0) for x in base_seq]
    # Scale to have reasonable magnitude
    total = sum(base_seq)
    if total > 0:
        base_seq = [x * 100 / total for x in base_seq]
    return base_seq

def evolve_sequence(population_size=50, generations=20, stagnation_threshold=5):
    """Evolve a population of sequences to find optimal ones."""
    # Initial population
    population = []
    for _ in range(population_size):
        n = adaptive_sequence_length()
        # Use structured initialization for some individuals
        if random.random() < 0.3:
            seq = generate_structured_sequence(n)
        else:
            seq = [np.random.random() * 100 for _ in range(n)]
        population.append(seq)
    
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
        
        # Keep top performers
        sorted_indices = np.argsort(fitness_scores)[::-1][:population_size//2]
        for idx in sorted_indices:
            new_population.append(population[idx])
            
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = population[random.choice(sorted_indices)]
            parent2 = population[random.choice(sorted_indices)]
            
            # Crossover (simple averaging)
            child = [(a+b)/2 for a,b in zip(parent1, parent2)]
            
            # Mutation
            if random.random() < 0.8:
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

def search_for_best_sequence(max_time=180) -> list[float]:
    """Function to search for the best coefficient sequence with improved logic."""
    start_time = time.time()

    # Try evolutionary approach first for several seconds
    best_sequence = None
    best_fitness = 0

    # Run evolution multiple times with different parameters
    for attempt in range(3):
        try:
            current_sequence, current_fitness = evolve_sequence(
                population_size=30 + attempt*10,
                generations=10 + attempt*5
            )
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = current_sequence
        except Exception as e:
            print(f"Evolution attempt {attempt} failed with error: {e}")
            continue
            
        if time.time() - start_time > max_time - 10:  # Leave 10 seconds for cleanup
            break

    # If no good sequence found from evolution, use gradient-based search
    if best_sequence is None:
        # Initialize with a structured sequence
        n = adaptive_sequence_length()
        best_sequence = generate_structured_sequence(n)

    best_score = evaluate_sequence(best_sequence)
    best_c1 = compute_autocorrelation_constant(best_sequence)

    iterations = 0
    max_iterations = 5000  # Prevent infinite loop

    while iterations < max_iterations and time.time() - start_time < max_time:
        # Try to improve the current sequence
        improved_sequence = get_good_direction_to_move_into(best_sequence)

        if improved_sequence is not None:
            # Check if this improvement is beneficial
            new_score = evaluate_sequence(improved_sequence)
            new_c1 = compute_autocorrelation_constant(improved_sequence)

            if new_score > best_score:
                best_sequence = improved_sequence
                best_score = new_score
                best_c1 = new_c1

                # Print progress every 100 iterations
                if iterations % 100 == 0:
                    print(f"Iteration {iterations}: Score = {best_score:.6f}, C1 = {best_c1:.6f}")

                # Check if we beat the benchmark
                benchmark_ratio = 1.5031 / best_c1
                if benchmark_ratio > 1.0:
                    print(f"BEAT BENCHMARK at iteration {iterations}! Ratio = {benchmark_ratio:.6f}")
                    break
        else:
            # If we couldn't improve, try random restart
            n = adaptive_sequence_length()
            best_sequence = [np.random.random() * 10 for _ in range(n)]

        iterations += 1

    # Final validation
    final_score = evaluate_sequence(best_sequence)
    final_c1 = compute_autocorrelation_constant(best_sequence)
    benchmark_ratio = 1.5031 / final_c1

    print(f"Final result:")
    print(f"  Score: {final_score:.6f}")
    print(f"  C1: {final_c1:.6f}")
    print(f"  Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"  Iterations: {iterations}")

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")