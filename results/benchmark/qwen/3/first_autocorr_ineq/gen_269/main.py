# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Historical sequences for seeding the algorithm
HISTORICAL_SEQUENCES = [
    [1.0] * 100,
    [1.0] * 50 + [0.0] * 50,
    [1.0, 0.0] * 50,
    [0.0] * 25 + [1.0] * 50 + [0.0] * 25,
    [1.0, 1.0, 0.0, 0.0] * 25,
    [1.0, 0.5, 0.25, 0.125] * 25,
    [1.0, 0.8, 0.6, 0.4, 0.2] * 20,
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 10,
]

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

def assess_numerical_stability(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Assess whether FFT convolution would be numerically stable for given inputs.
    Returns True if using FFT is likely to be stable, False otherwise.
    """
    n = len(a)

    # For very small sequences, always prefer direct convolution
    if n < 50:
        return False

    # Check for sequences with very low variance or flat signals
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-6 or std_b < 1e-6:
        return False

    # Check if sequences have extreme dynamic ranges
    if np.max(a) / (np.min(a) + 1e-10) > 1000 or np.max(b) / (np.min(b) + 1e-10) > 1000:
        return False

    # Compute a sample of the convolution and examine its properties
    try:
        direct_conv = convolve_direct(a, b)
        fft_conv = convolve_fft(a, b)

        # Compare magnitudes and ensure they are close
        direct_max = np.max(np.abs(direct_conv))
        fft_max = np.max(np.abs(fft_conv))

        if direct_max > 0 and abs(direct_max - fft_max) / direct_max > 0.1:
            return False

        return True
    except:
        return False

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Prefer FFT for large sequences or ones that are deemed stable
    use_fft = n > 100 and assess_numerical_stability(a, a)

    if use_fft:
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

        # Efficiently build constraint matrix for convolution constraints
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

        # Solve with multiple fallbacks for stability
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                                options={'presolve': True, 'maxiter': 1000})

        if not result.success:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex',
                                    options={'maxiter': 1000})

        if result.success:
            g_sequence = result.x
            # Validate solution
            if np.any(np.isnan(g_sequence)) or np.any(np.isinf(g_sequence)):
                return None
            g_sequence = np.maximum(g_sequence, 0)
            if np.sum(g_sequence) < 1e-10:
                return None
            return g_sequence.tolist()
        else:
            return None

    except Exception as e:
        return None

def get_good_direction_to_move_into(
    sequence: List[float],
    max_iterations: int = 10
) -> Optional[List[float]]:
    """Improve the sequence using enhanced local optimization strategies with curvature awareness."""
    n = len(sequence)

    # Normalize the sequence
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution constraints using adaptive method
    if n > 100:
        b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
    else:
        b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

    rhs = np.max(b)

    # Try solving LP multiple times with adjusted RHS if necessary
    g_fun = None
    for attempt in range(max_iterations):
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)
        if g_fun is not None:
            break
        rhs *= 1.05

    if g_fun is None:
        # Fallback strategy with gradient ascent
        t = min(0.1, 0.05 + 0.02 * np.log(n + 1))
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize the solution from LP
    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Apply curvature-aware directional bias correction
    if n > 10:
        try:
            # Use a more numerically stable approach to curvature estimation
            epsilon = 1e-4

            grad_vector = np.array(normalized_g_fun)
            # Perturb along this direction to estimate curvature
            perturbed_seq1 = [x + epsilon * y for x, y in zip(normalized_sequence, grad_vector)]
            perturbed_seq2 = [x - epsilon * y for x, y in zip(normalized_sequence, grad_vector)]

            # Recompute convolution with perturbed sequences
            if n > 100:
                b_forward = convolve_fft(np.array(perturbed_seq1), np.array(perturbed_seq1))
                b_backward = convolve_fft(np.array(perturbed_seq2), np.array(perturbed_seq2))
            else:
                b_forward = convolve_direct(np.array(perturbed_seq1), np.array(perturbed_seq1))
                b_backward = convolve_direct(np.array(perturbed_seq2), np.array(perturbed_seq2))

            # Estimate curvature from max convolution values
            curv_forward = np.max(b_forward)
            curv_backward = np.max(b_backward)
            curv_center = np.max(b)

            # More accurate second derivative approximation
            curvature = (curv_forward + curv_backward - 2 * curv_center) / (epsilon ** 2)

            # Apply curvature correction if meaningful
            if curvature > 0:
                curvature_correction = curvature * 0.01 * grad_vector
                curvature_correction = curvature_correction / (1.0 + np.linalg.norm(curvature_correction) * 0.1)
                corrected_direction = np.array(normalized_g_fun) + curvature_correction
                normalized_g_fun = corrected_direction.tolist()
        except Exception:
            pass

    # Apply adaptive perturbation for exploration
    t = min(0.1, 0.05 + 0.02 * np.log(n + 1))
    new_sequence = [(1 - t*0.8) * x + t*0.8 * y for x, y in zip(sequence, normalized_g_fun)]

    # Add more sophisticated randomness for better exploration
    for i in range(n):
        if random.random() < 0.1:
            new_sequence[i] = max(0, new_sequence[i] * np.random.normal(1, 0.2))
        else:
            new_sequence[i] = max(0, new_sequence[i] + np.random.normal(0, 0.02 * new_sequence[i]))

    # Ensure non-negativity and bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply random mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.5 * mutated[i]))
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

def initialize_population_with_historical_samples(
    population_size: int,
    initial_length_range: Tuple[int, int],
    historical_samples: int = 3
) -> List[List[float]]:
    """Initialize population by sampling from historical sequences."""
    population = []

    # Add some historical sequences
    for _ in range(historical_samples):
        idx = random.randrange(len(HISTORICAL_SEQUENCES))
        seq = HISTORICAL_SEQUENCES[idx].copy()
        # Adjust length to fit range
        n = random.randint(*initial_length_range)
        if n < len(seq):
            seq = seq[:n]
        elif n > len(seq):
            seq = seq + [0.0] * (n - len(seq))
        population.append(seq)

    # Fill the rest with random sequences
    for _ in range(population_size - historical_samples):
        n = random.randint(*initial_length_range)
        individual = [random.random() * 100 for _ in range(n)]
        population.append(individual)

    return population

def adaptive_search_for_best_sequence(
    max_time_seconds: int = 180,
    initial_length_range: Tuple[int, int] = (100, 500),
    population_size: int = 20,
    generations: int = 50,
    elite_fraction: float = 0.2,
    patience_limit: int = 10
) -> List[float]:
    """Main function to search for the best coefficient sequence using improved evolutionary approach."""
    start_time = time.time()

    # Initialize population with historical sequences for better starting points
    population = initialize_population_with_historical_samples(
        population_size, initial_length_range
    )

    best_sequence = None
    best_inv_c1 = 0.0
    patience_counter = 0

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
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if no improvement for too long
        if patience_counter > patience_limit:
            break

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
                child = [random.random() * 100 for _ in range(n)]

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