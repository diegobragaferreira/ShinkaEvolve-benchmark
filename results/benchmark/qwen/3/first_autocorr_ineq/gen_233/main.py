# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Optional, Tuple
import time

# Library of historically successful sequences
HISTORICAL_SEQUENCES = [
    [1.0] * 100,  # Uniform sequence
    [1.0, 0.5] * 50,  # Alternating sequence
    [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # Increasing sequence
    [0.5 ** i for i in range(100)],  # Geometric decay
    [1.0 if i % 2 == 0 else 0.1 for i in range(100)],  # Sparse sequence
]

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

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
    """Solves the convolution LP for a given sequence and RHS with robust fallback handling."""
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

        # Solve the linear program with multiple methods as fallback
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 1000})

        if not result.success:
            # Try different method if highs fails
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex', options={'maxiter': 1000})

        if result.success:
            return result.x.tolist()
        else:
            return None

    except Exception:
        return None

def get_curvature_corrected_direction(
    sequence: List[float],
    normalized_sequence: List[float],
    max_iterations: int = 10
) -> Optional[List[float]]:
    """Compute a curvature-corrected direction using finite differences."""
    n = len(sequence)

    if n <= 10:
        return None  # Not enough elements for meaningful curvature estimation

    # Normalize the sequence
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Compute base convolution
    if n > 100:
        b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
    else:
        b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

    rhs = np.max(b)

    # Solve LP for base direction
    g_fun = None
    for _ in range(max_iterations):
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)
        if g_fun is not None:
            break
        else:
            rhs *= 1.01

    if g_fun is None:
        return None

    # Normalize solution
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Apply curvature correction using finite differences
    epsilon = 1e-4
    hessian_approx = np.zeros((n, n))

    for i in range(n):
        # Perturb sequence in dimension i
        perturbed_seq = normalized_sequence.copy()
        perturbed_seq[i] += epsilon

        # Recompute convolution with perturbation
        if n > 100:
            b_perturbed = convolve_fft(np.array(perturbed_seq), np.array(perturbed_seq))
        else:
            b_perturbed = convolve_direct(np.array(perturbed_seq), np.array(perturbed_seq))

        second_derivative = (np.max(b_perturbed) - np.max(b)) / (epsilon ** 2)
        hessian_approx[i, i] = max(0, second_derivative)

    # Apply curvature correction to direction
    curvature_correction = np.dot(hessian_approx, g_fun)
    curvature_correction = curvature_correction / (1.0 + np.linalg.norm(curvature_correction))

    corrected_direction = np.array(g_fun) + 0.1 * curvature_correction
    corrected_direction = np.maximum(corrected_direction, 0)

    # Normalize again after correction
    sum_corrected = np.sum(corrected_direction)
    if sum_corrected > 0:
        corrected_direction = corrected_direction * (np.sum(g_fun) / sum_corrected)

    return corrected_direction.tolist()

def get_good_direction_to_move_into(
    sequence: List[float],
    max_iterations: int = 10
) -> Optional[List[float]]:
    """Improve the sequence using curvature-aware evolutionary strategy."""
    n = len(sequence)

    # Normalize the sequence
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Use curvature-aware correction for better convergence
    corrected_direction = get_curvature_corrected_direction(sequence, normalized_sequence, max_iterations)

    if corrected_direction is None:
        # Fallback to direct computation
        if n > 100:
            b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
        else:
            b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

        rhs = np.max(b)
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)

        if g_fun is None:
            return None

        sum_g = np.sum(g_fun)
        if sum_g < 0.01:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    else:
        normalized_g_fun = corrected_direction

    # Apply adaptive step-size
    current_c1, _ = compute_c1_constant(sequence)
    t = min(0.1, max(0.01, 0.05 * (1.0 - min(1.0, current_c1 / 1.5))))

    # Add diversity with Gaussian noise
    noise = [np.random.normal(0, 0.01) for _ in range(n)]
    new_sequence = [
        (1 - t) * x + t * y + noise[i] for i, (x, y) in enumerate(zip(sequence, normalized_g_fun))
    ]

    # Ensure non-negativity and reasonable bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply random mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use larger variance for better exploration
            mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.5 * mutated[i]))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform uniform crossover between two sequences."""
    # Ensure both sequences are of same length
    min_len = min(len(seq1), len(seq2))
    child = []
    for i in range(min_len):
        if random.random() < 0.5:
            child.append(seq1[i])
        else:
            child.append(seq2[i])

    # Handle length differences
    if len(seq1) > len(seq2):
        child.extend(seq1[min_len:])
    elif len(seq2) > len(seq1):
        child.extend(seq2[min_len:])

    return child

def initialize_sequence_with_history():
    """Initialize a sequence using historical sampling combined with randomness."""
    # 70% chance to use historical sequence, 30% chance for random
    if random.random() < 0.7 and HISTORICAL_SEQUENCES:
        # Sample from historical sequences with slight perturbation
        historical = random.choice(HISTORICAL_SEQUENCES)
        # Apply small random perturbation
        perturbed = [max(0.0, x + random.gauss(0, 0.1 * x)) for x in historical]
        return perturbed
    else:
        # Purely random initialization
        n = random.randint(100, 500)
        return [random.random() * 100 for _ in range(n)]

def hierarchical_evolution_strategy(
    max_time_seconds: int = 180,
    initial_length_range: Tuple[int, int] = (100, 500),
    population_size: int = 30,
    generations_per_scale: int = 40,
    scales: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0],
    elite_fraction: float = 0.3,
    patience_limit: int = 20
) -> List[float]:
    """Hierarchical evolutionary strategy to optimize the sequence."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0
    patience_counter = 0
    last_improvement = 0

    # Process different scales hierarchically
    for scale in scales:
        if time.time() - start_time > max_time_seconds:
            break

        # Initialize population for current scale
        population = []
        n = max(50, int(200 * scale))
        for _ in range(population_size):
            individual = initialize_sequence_with_history()
            population.append(individual)

        for generation in range(generations_per_scale):
            if time.time() - start_time > max_time_seconds:
                break

            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                _, inv_c1 = compute_c1_constant(individual)
                fitness_scores.append((individual, inv_c1))

            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Update best solution
            current_best, current_best_inv_c1 = fitness_scores[0]
            if current_best_inv_c1 > best_inv_c1:
                best_inv_c1 = current_best_inv_c1
                best_sequence = current_best.copy()
                patience_counter = 0
                last_improvement = generation
            else:
                patience_counter += 1

            if patience_counter > patience_limit:
                break

            # Select elite
            elite_count = int(elite_fraction * population_size)
            elite_individuals = [ind for ind, _ in fitness_scores[:elite_count]]

            # Generate new population
            new_population = elite_individuals.copy()

            # Create offspring
            while len(new_population) < population_size:
                parent1 = random.choice(elite_individuals)
                parent2 = random.choice(elite_individuals)

                child = crossover_sequences(parent1, parent2)
                child = mutate_sequence(child)

                # Random initialization for diversity
                if random.random() < 0.1:
                    n = random.randint(*initial_length_range)
                    child = [random.uniform(0.1, 1.0) for _ in range(n)]

                new_population.append(child)

            population = new_population

    # Final tuning
    if best_sequence is not None:
        for _ in range(30):
            if time.time() - start_time > max_time_seconds:
                break
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
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    return hierarchical_evolution_strategy()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")