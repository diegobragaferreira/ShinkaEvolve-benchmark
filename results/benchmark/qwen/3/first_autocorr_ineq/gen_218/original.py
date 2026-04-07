# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

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

def solve_convolution_lp(f_sequence: List[float], rhs: float, n: int) -> Optional[List[float]]:
    """Solves the convolution LP for a given sequence and RHS with robustness."""
    try:
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

        # Solve the linear program with multiple fallbacks
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                                options={'presolve': True, 'maxiter': 1000})

        if not result.success:
            # Try simpler method if highs fails
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex')

        if result.success:
            g_sequence = result.x
            # Validate solution
            if np.any(np.isnan(g_sequence)) or np.any(np.isinf(g_sequence)):
                return None
            # Ensure non-negativity and reasonable values
            g_sequence = np.maximum(g_sequence, 0)
            if np.sum(g_sequence) < 1e-10:
                return None
            return g_sequence.tolist()
        else:
            return None

    except Exception as e:
        return None

def get_good_direction_to_move_into(sequence: List[float]) -> Optional[List[float]]:
    """Improve the sequence using enhanced local optimization strategies."""
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

    # Try solving LP multiple times with adjusted RHS if necessary
    g_fun = None
    for attempt in range(5):
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)
        if g_fun is not None:
            break
        rhs *= 1.05

    if g_fun is None:
        # Fallback to simple gradient ascent with adaptive step size
        t = min(0.1, 0.05 + 0.02 * np.log(n + 1))
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize the solution from LP
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Apply adaptive perturbation with multiple strategies
    t = min(0.1, 0.05 + 0.02 * np.log(n + 1))

    # Hybrid approach: combine LP solution with gradient descent refinement
    # First try direct combination
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

    # Add a bit of randomness for exploration
    if n > 100:
        # For larger sequences, use more sophisticated exploration
        random_perturbation = np.random.normal(0, 0.01 * t, n)
        new_sequence = [max(0, x + 0.5 * p * x) for x, p in zip(new_sequence, random_perturbation)]
    else:
        # For smaller sequences, add simple noise
        new_sequence = [max(0, x + np.random.normal(0, 0.02 * x)) for x in new_sequence]

    # Ensure non-negativity and reasonable bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply random mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
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

def adaptive_search_for_best_sequence(
    max_time_seconds: int = 180,
    initial_length_range: Tuple[int, int] = (100, 500),
    population_size: int = 20,
    generations: int = 50,
    elite_fraction: float = 0.2
) -> List[float]:
    """Main function to search for the best coefficient sequence using improved evolutionary approach."""
    start_time = time.time()

    # Initialize population with diverse sequences
    population = []
    for _ in range(population_size):
        n = random.randint(*initial_length_range)
        # Use different initialization strategies
        init_type = random.randint(0, 2)
        if init_type == 0:
            individual = [random.random() * 100 for _ in range(n)]
        elif init_type == 1:
            individual = [0.0] * n
            individual[random.randint(0, n-1)] = random.random() * 100
        else:
            individual = [abs(random.gauss(0, 1)) * 10 for _ in range(n)]
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
                # Use different initialization strategies
                init_type = random.randint(0, 2)
                if init_type == 0:
                    child = [random.random() * 100 for _ in range(n)]
                elif init_type == 1:
                    child = [0.0] * n
                    child[random.randint(0, n-1)] = random.random() * 100
                else:
                    child = [abs(random.gauss(0, 1)) * 10 for _ in range(n)]

            new_population.append(child)

        population = new_population

    # Final optimization of best solution
    if best_sequence is not None:
        for _ in range(20):  # Additional fine-tuning iterations
            improved = get_good_direction_to_move_into(best_sequence)
            if improved is None:
                break
            _, inv_c1_new = compute_c1_constant(improved)
            _, inv_c1_old = compute_c1_constant(best_sequence)
            if inv_c1_new > inv_c1_old:
                best_sequence = improved
            else:
                break

    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    return adaptive_search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")