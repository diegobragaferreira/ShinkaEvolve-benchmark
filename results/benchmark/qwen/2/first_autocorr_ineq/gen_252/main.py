# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import random
import time
from collections import deque

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_step_function(n_steps, heights=None, max_height=1000):
    """Generate a step function with n_steps and optional specific heights."""
    if heights is None:
        heights = [random.uniform(0, max_height) for _ in range(n_steps)]
    else:
        heights = heights[:n_steps] + [0] * (n_steps - len(heights))
        heights = heights[:n_steps]

    # Create step function with equal-length steps
    total_length = 1000  # Fixed length for step function
    step_sizes = [total_length // n_steps] * n_steps
    remainder = total_length % n_steps
    for i in range(remainder):
        step_sizes[i] += 1

    # Build the sequence
    sequence = []
    for i, (height, size) in enumerate(zip(heights, step_sizes)):
        sequence.extend([height] * size)

    # Trim or pad to exact length
    if len(sequence) > total_length:
        sequence = sequence[:total_length]
    elif len(sequence) < total_length:
        sequence.extend([0.0] * (total_length - len(sequence)))

    return sequence

def generate_power_law_sequence(n, alpha=0.95):
    """Generates a power-law decay sequence which often performs well."""
    sequence = [alpha ** i for i in range(n)]
    # Normalize to ensure sum is reasonable
    total = sum(sequence)
    if total > 0:
        sequence = [x / total * 1000 for x in sequence]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence by randomly modifying elements."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            new_sequence[i] *= random.uniform(1 - max_mutation, 1 + max_mutation)
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    return new_sequence

def crossover_sequences(seq1, seq2):
    """Perform uniform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))
    new_seq = []
    for i in range(max_len):
        if random.random() < 0.5:
            new_seq.append(padded_seq1[i])
        else:
            new_seq.append(padded_seq2[i])
    return new_seq

def quadratic_optimization_approach(initial_sequence, max_iter=1000):
    """Use quadratic programming to directly optimize the 1/C1 objective."""
    n = len(initial_sequence)
    x0 = np.array(initial_sequence)

    def objective(x):
        x = np.maximum(x, 0)
        conv = fftconvolve(x, x, mode='full')[:len(x)*2-1]
        max_conv = np.max(conv)
        sum_x = np.sum(x)
        if sum_x < 0.01:
            return 1e6
        c1 = (2 * len(x) * max_conv) / (sum_x ** 2)
        return -1.0 / c1  # Negative because we want to maximize 1/C1

    bounds = [(0, 1000) for _ in range(n)]
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-9})
        return result.x.tolist() if result.success else initial_sequence
    except:
        return initial_sequence

def simulated_annealing_search(initial_sequence, max_iter=200):
    """Improved simulated annealing with adaptive cooling schedule."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    temp = 100.0
    cooling_rate = 0.995
    min_temp = 0.1

    for iteration in range(max_iter):
        mutated = mutate_sequence(current_sequence, mutation_rate=0.3, max_mutation=0.3)
        mutated_fitness = evaluate_sequence(mutated)
        
        # Accept or reject based on Metropolis criterion
        if mutated_fitness > current_fitness or random.random() < np.exp((mutated_fitness - current_fitness) / (temp + 1e-10)):
            current_sequence = mutated
            current_fitness = mutated_fitness
        
        if current_fitness > best_fitness:
            best_sequence = current_sequence.copy()
            best_fitness = current_fitness
        
        temp = max(temp * cooling_rate, min_temp)
    
    return best_sequence, best_fitness

def hybrid_search_strategy(max_time_seconds=180):
    """Combines multiple strategies in a smart order to maximize exploration."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0

    # Strategy 1: Try various step counts and height patterns
    step_counts = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    height_patterns = [
        lambda n: [1000 * (0.9 ** i) for i in range(n)],  # Exponential decay
        lambda n: [1000 * (1.0 / (i + 1)) for i in range(n)],  # Harmonic decay
        lambda n: [1000] * n,  # Constant
        lambda n: [1000 * (i / (n - 1)) if n > 1 else 1000 for i in range(n)]  # Linear increase
    ]

    # Try various combinations of step counts and height patterns
    for attempt in range(50):  # More attempts to ensure thorough search
        if time.time() - start_time > max_time_seconds:
            break

        n_steps = random.choice(step_counts)
        pattern_fn = random.choice(height_patterns)

        # Generate step function using a pattern
        heights = pattern_fn(n_steps)
        sequence = generate_step_function(n_steps, heights=heights)

        # Optimize using quadratic programming
        optimized_sequence = quadratic_optimization_approach(sequence, 150)

        # Evaluate the optimized sequence
        inv_c1 = evaluate_sequence(optimized_sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized_sequence

    # Strategy 2: Refinement using simulated annealing
    if best_sequence is not None:
        refined_seq, refined_fitness = simulated_annealing_search(best_sequence, 100)
        if refined_fitness > best_inv_c1:
            best_inv_c1 = refined_fitness
            best_sequence = refined_seq

    # Strategy 3: Fallback to generic optimization if needed
    if best_sequence is None or time.time() - start_time > 170:
        initial_sequence = generate_power_law_sequence(random.randint(10, 100))
        optimized_seq = quadratic_optimization_approach(initial_sequence)
        optimized_fitness = evaluate_sequence(optimized_seq)
        if optimized_fitness > best_inv_c1:
            best_inv_c1 = optimized_fitness
            best_sequence = optimized_seq

    # Final check with simulated annealing
    if best_sequence is not None:
        final_seq, final_fitness = simulated_annealing_search(best_sequence, 100)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq

    return best_sequence if best_sequence is not None else generate_power_law_sequence(50)

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    return hybrid_search_strategy(170)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")