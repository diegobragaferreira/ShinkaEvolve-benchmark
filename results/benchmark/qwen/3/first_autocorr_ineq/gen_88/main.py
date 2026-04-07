# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def solve_convolution_lp(f_sequence: List[float], rhs: float) -> Optional[List[float]]:
    """Solves the convolution LP for a given sequence and RHS with adaptive constraint handling."""
    try:
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build constraint matrix for convolution constraints
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

        # Solve the linear program with multiple methods as fallback
        # Try with presolve enabled first for better performance
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True})
        except:
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            except:
                # Try revised simplex if highs fails
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex')

        if result.success:
            return result.x.tolist()
        else:
            return None

    except Exception as e:
        # Handle any unexpected errors gracefully
        return None

def get_good_direction_to_move_into(
    sequence: List[float],
    iteration: int = 0,
    max_iterations: int = 10
) -> Optional[List[float]]:
    """Improve the sequence using evolutionary strategy and LP optimization."""
    n = len(sequence)

    # Normalize the sequence
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution constraints
    if n > 100:
        b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
    else:
        b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

    rhs = np.max(b)

    # Adaptive constraint tightening to improve numerical stability
    # Reduce the RHS slightly to make LP more feasible
    rhs_factor = 1.0 + 0.01 * iteration  # Gradually tighten constraints
    rhs = rhs * rhs_factor

    # Try multiple times to solve LP
    g_fun = None
    for _ in range(max_iterations):
        g_fun = solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is not None:
            break
        else:
            # If LP fails, slightly modify constraints and retry
            rhs *= 1.01

    if g_fun is None:
        return None

    # Normalize the solution from LP
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Adaptive learning rate that decreases with iterations for better convergence
    base_learning_rate = 0.05
    adaptive_lr = base_learning_rate / (1 + 0.1 * iteration)
    adaptive_lr = max(0.005, adaptive_lr)  # Minimum learning rate
    
    new_sequence = [
        (1 - adaptive_lr) * x + adaptive_lr * y for x, y in zip(sequence, normalized_g_fun)
    ]

    # Ensure non-negativity and reasonable bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1, iteration: int = 0) -> List[float]:
    """Apply random mutation to a sequence with adaptive rate."""
    mutated = sequence.copy()
    # Decrease mutation rate with iterations to stabilize convergence
    adaptive_mutation_rate = mutation_rate / (1 + 0.05 * iteration)
    adaptive_mutation_rate = max(0.01, adaptive_mutation_rate)
    
    for i in range(len(mutated)):
        if random.random() < adaptive_mutation_rate:
            mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.1 * mutated[i]))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform uniform crossover between two sequences."""
    child = []
    for i in range(min(len(seq1), len(seq2))):
        if random.random() < 0.5:
            child.append(seq1[i])
        else:
            child.append(seq2[i])
    return child

def hybrid_local_search(individual: List[float], max_iter: int = 50, restarts: int = 3) -> List[float]:
    """Apply a hybrid local search to further improve a sequence with multiple restarts."""
    current = individual.copy()
    best_sequence = current.copy()
    _, best_inv_c1 = compute_c1_constant(current)
    
    for restart in range(restarts):
        current = individual.copy()
        for _ in range(max_iter):
            improved = get_good_direction_to_move_into(current, iteration=_)
            if improved is None:
                break
            _, inv_c1_new = compute_c1_constant(improved)
            _, inv_c1_old = compute_c1_constant(current)
            if inv_c1_new > inv_c1_old:
                current = improved
                if inv_c1_new > best_inv_c1:
                    best_inv_c1 = inv_c1_new
                    best_sequence = current.copy()
            else:
                # If no improvement, try a perturbation to escape local minima
                current = [max(0, x + np.random.normal(0, 0.01 * x)) for x in current]
                
        # Perturb for next restart
        individual = [max(0, x + np.random.normal(0, 0.05 * x)) for x in individual]
        
    return best_sequence

def adaptive_search_for_best_sequence(
    max_time_seconds: int = 180,
    initial_length_range: Tuple[int, int] = (100, 1000),
    population_size: int = 30,
    generations: int = 100,
    elite_fraction: float = 0.2,
    local_search_iters: int = 50,
    restarts_per_local_search: int = 3
) -> List[float]:
    """Main function to search for the best coefficient sequence using improved evolutionary approach."""
    start_time = time.time()

    # Initialize population with diverse sequences
    population = []
    for _ in range(population_size):
        n = random.randint(*initial_length_range)
        # Use exponential distribution for more diverse initialization
        individual = [np.random.exponential(1.0) * 100 for _ in range(n)]
        population.append(individual)

    best_sequence = None
    best_inv_c1 = 0.0

    for generation in range(generations):
        if time.time() - start_time > max_time_seconds:
            break

        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_c1_constant(individual)
            fitness_scores.append((individual, inv_c1))

        # Sort by fitness (descending order)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Update best solution
        current_best, current_best_inv_c1 = fitness_scores[0]
        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = current_best.copy()

        # Select elite individuals
        elite_count = int(elite_fraction * population_size)
        elite_individuals = [ind for ind, _ in fitness_scores[:elite_count]]

        # Generate new population
        new_population = elite_individuals.copy()

        # Create offspring through crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(elite_individuals)
            parent2 = random.choice(elite_individuals)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation
            child = mutate_sequence(child)

            # Random initialization for diversity
            if random.random() < 0.1:
                n = random.randint(*initial_length_range)
                child = [np.random.exponential(1.0) * 100 for _ in range(n)]

            new_population.append(child)

        population = new_population

    # Final optimization of best solution with hybrid local search
    if best_sequence is not None:
        # Apply local search to the best solution found
        best_sequence = hybrid_local_search(best_sequence, local_search_iters, restarts_per_local_search)

    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    return adaptive_search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")