# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional
import random
import time
from joblib import Parallel, delayed

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence: List[float]) -> tuple[float, float]:
    """Computes the autocorrelation constant C1 and its reciprocal 1/C1."""
    if not sequence or sum(sequence) < 0.01:
        return (float('inf'), 0.0)

    n = len(sequence)
    # Use FFT-based convolution for efficiency O(n log n)
    conv = np.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    sum_seq = sum(sequence)

    if sum_seq == 0:
        return (float('inf'), 0.0)

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return (c1, inv_c1)

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Use FFT for faster convolution
    try:
        # Pad sequence to ensure proper convolution length
        padded_len = 2 * n - 1
        padded_sequence = np.pad(normalized_sequence, (0, padded_len - n), 'constant')
        # Compute FFT and convolution
        fft_seq = fft(padded_sequence)
        conv_result = ifft(fft_seq * np.conj(fft_seq))
        # Extract valid convolution results
        conv_result = np.real(conv_result[:padded_len])
        rhs = np.max(conv_result)
    except Exception as e:
        print(f"Error during FFT convolution: {e}")
        return None

    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Precompute the convolution constraint matrix using explicit loop
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            print('LP optimization failed:', result.message)
            return None
    except Exception as e:
        print(f'LP optimization error: {e}')
        return None

def adaptive_tournament_selection(population: List[List[float]],
                                fitness_scores: List[float],
                                generation: int,
                                population_size: int) -> List[float]:
    """Perform adaptive tournament selection based on diversity and generation."""

    # Determine tournament size based on generation and population diversity
    if generation <= 20:  # Early generations
        tournament_size = min(9, max(5, population_size // 4))
    elif generation >= 50:  # Later generations
        tournament_size = min(4, max(3, population_size // 8))
    else:  # Middle generations
        tournament_size = 5

    # Perform tournament selection
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]

    return population[winner_idx].copy()

def generate_step_sequence(length: int, num_steps: int) -> List[float]:
    """Generate a step function sequence with specified number of steps."""
    if num_steps <= 0 or length <= 0:
        return [1.0] * length

    step_positions = sorted(random.sample(range(length), min(num_steps, length)))
    step_heights = [random.uniform(10.0, 100.0) for _ in range(len(step_positions))]

    sequence = [0.0] * length
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        sequence[pos:end_pos] = [height] * (end_pos - pos)
    return sequence

def generate_structured_sequence(length: int) -> List[float]:
    """Generate a more structured sequence that potentially performs better."""
    # Create a sequence with some inherent structure
    sequence = []

    # Mix of exponential decay and step-like patterns
    for i in range(length):
        # Exponential decay component
        exp_component = 100 * np.exp(-i * 0.01)
        # Add some periodic variations
        period_component = 10 * np.sin(i * 0.2) * np.cos(i * 0.05)
        # Combine components
        val = max(0.01, exp_component + period_component)
        sequence.append(val)

    return sequence

def generate_diverse_population(population_size: int, length_range=(100, 1000)) -> List[List[float]]:
    """Generate a diverse initial population with various patterns."""
    population = []

    # Generate sequences using different methods
    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        population.append(generate_structured_sequence(n))

    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        num_steps = max(2, min(20, n // 10))
        population.append(generate_step_sequence(n, num_steps))

    # Fill remaining with standard random sequences
    while len(population) < population_size:
        n = random.randint(*length_range)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)

    return population

def mutate_sequence(sequence: List[float], generation: int, population_size: int) -> List[float]:
    """Apply adaptive mutation to a sequence."""
    mutated = sequence.copy()
    # Dynamic mutation rate
    mutation_rate = max(0.05, 0.3 * (1 - generation / (population_size * 2)))
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Global perturbation - larger changes early, smaller later
            if generation < population_size // 4:
                noise = random.gauss(0, 5.0)  # Large change
            else:
                noise = random.gauss(0, 1.0)  # Small change
            mutated[i] = max(0.01, mutated[i] + noise)
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover
    crossover_point = random.randint(1, min_len - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]
    return child

def fitness_sharing(fitness_scores: List[float], population: List[List[float]], sigma: float = 0.5) -> List[float]:
    """Apply fitness sharing to maintain diversity."""
    shared_fitness = []
    for i in range(len(population)):
        sharing_value = 0
        for j in range(len(population)):
            if i != j:
                # Euclidean distance between sequences (normalized)
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j])) / len(population[i])
                if dist < sigma:
                    sharing_value += 1 - dist / sigma
        shared_fitness.append(fitness_scores[i] / (1 + sharing_value))
    return shared_fitness

def parallel_evaluate_fitness(population: List[List[float]]) -> List[float]:
    """Evaluate fitness (1/C1) of each individual in parallel."""
    results = Parallel(n_jobs=-1)(delayed(compute_autocorrelation_constant)(seq) for seq in population)
    return [inv_c1 for _, inv_c1 in results]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    max_time_seconds = 170

    # Initialize population with diverse strategies
    population_size = 30
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 30

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual in parallel
        fitness_scores = parallel_evaluate_fitness(population)

        # Apply fitness sharing for diversity
        shared_fitness_scores = fitness_sharing(fitness_scores, population)

        # Track best solution
        current_best_idx = np.argmax(shared_fitness_scores)
        current_best_inv_c1 = shared_fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Selection with fitness proportionate selection and tournament backup
        selected_parents = []

        # Fitness proportionate selection
        total_fitness = sum(shared_fitness_scores)
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in shared_fitness_scores]
            selected_parents.extend(random.choices(population, probabilities, k=population_size - 1))
        else:
            # Fallback to tournament selection
            tournament_size = 5
            for _ in range(population_size - 1):
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [shared_fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_parents.append(population[winner_idx].copy())

        # Elitism: keep the best individual
        selected_parents.insert(0, best_sequence.copy())

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate_sequence(child, generation, population_size)

            new_population.append(child)

        population = new_population[:population_size]

        # Early termination if no significant improvement
        if stagnation_count > 10 and best_inv_c1 > 0.6:
            break

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_step_sequence(100, 5)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")