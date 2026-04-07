# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
from typing import List, Tuple
import time

# Constants for the optimization
MAX_GENERATIONS = 1000
POPULATION_SIZE = 50
MUTATION_RATE_START = 0.1
MUTATION_RATE_END = 0.01
ELITISM_COUNT = 5
STAGNATION_THRESHOLD = 50

def compute_autocorrelation_constant(sequence: List[float]) -> float:
    """Computes the autocorrelation constant C1 for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Compute convolution using FFT for efficiency
    convolved = fft_convolve(sequence, sequence, mode='full')
    # Take the second half (the actual convolution)
    convolved = convolved[len(sequence)-1:]

    max_conv = np.max(convolved)
    sum_seq = np.sum(sequence)

    if sum_seq < 0.01:
        return float('inf')  # Reject invalid sequences

    # C1 = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def compute_inverse_c1(sequence: List[float]) -> float:
    """Computes 1/C1 for maximization."""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0  # Invalid sequences get very low score
    return 1.0 / c1 if c1 > 0 else 0.0

def generate_structured_sequence(length: int, seq_type: str = 'random') -> List[float]:
    """Generate sequences with specific structures for better initialization."""
    if seq_type == 'sine':
        # Generate sine wave pattern
        return [abs(np.sin(i * np.pi / length)) for i in range(length)]
    elif seq_type == 'gaussian':
        # Generate truncated normal distribution
        return [max(0, np.random.normal(1, 0.5)) for _ in range(length)]
    elif seq_type == 'step':
        # Generate step function
        return [1.0 if i < length//2 else 0.5 for i in range(length)]
    else:
        # Random uniform
        return [random.uniform(0, 1) for _ in range(length)]

def mutate_sequence(sequence: List[float], mutation_rate: float) -> List[float]:
    """Apply mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    if len(seq1) != len(seq2):
        # If lengths differ, make them same size
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

    point = random.randint(1, len(seq1) - 1)
    child = seq1[:point] + seq2[point:]
    return child

def local_refinement(sequence: List[float], max_iterations: int = 10) -> List[float]:
    """Simple local refinement heuristic."""
    refined = sequence.copy()
    for _ in range(max_iterations):
        # Try small adjustments to see if we can improve
        test_seq = refined.copy()
        idx = random.randint(0, len(test_seq) - 1)
        test_seq[idx] = max(0, test_seq[idx] + random.gauss(0, 0.01))

        if compute_inverse_c1(test_seq) > compute_inverse_c1(refined):
            refined = test_seq

    return refined

def search_for_best_sequence() -> List[float]:
    """Enhanced search for the best coefficient sequence using evolutionary algorithm."""
    start_time = time.time()

    # Initialize population with diverse strategies
    population = []
    for i in range(POPULATION_SIZE):
        length = random.randint(50, 500)  # Vary sequence length
        init_strategy = random.choice(['random', 'sine', 'gaussian', 'step'])
        individual = generate_structured_sequence(length, init_strategy)
        population.append(individual)

    best_sequence = None
    best_fitness = 0.0
    stagnation_count = 0

    for generation in range(MAX_GENERATIONS):
        # Check time limit
        if time.time() - start_time > 170:  # Leave some buffer
            break

        # Evaluate fitness for all individuals
        fitness_scores = []
        for ind in population:
            fitness = compute_inverse_c1(ind)
            fitness_scores.append((fitness, ind))

        # Sort by fitness
        fitness_scores.sort(reverse=True)
        current_best_fitness = fitness_scores[0][0]
        current_best_sequence = fitness_scores[0][1]

        # Update global best
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_sequence = current_best_sequence.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Early stopping if no improvement
        if stagnation_count > STAGNATION_THRESHOLD:
            break

        # Select top individuals (elitism)
        elite = [ind for _, ind in fitness_scores[:ELITISM_COUNT]]

        # Adaptive mutation rate
        mutation_rate = MUTATION_RATE_START + (
            (MUTATION_RATE_END - MUTATION_RATE_START) *
            generation / MAX_GENERATIONS
        )

        # Generate new population
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1 = random.choice(fitness_scores[:POPULATION_SIZE//2])[1]
            parent2 = random.choice(fitness_scores[:POPULATION_SIZE//2])[1]

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation
            child = mutate_sequence(child, mutation_rate)

            # Local refinement
            child = local_refinement(child, 5)

            new_population.append(child)

        population = new_population

    # Final refinement of best sequence
    if best_sequence is not None:
        best_sequence = local_refinement(best_sequence, 20)

    return best_sequence if best_sequence is not None else []

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")