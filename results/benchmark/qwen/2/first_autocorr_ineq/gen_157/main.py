# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal, optimize
from scipy.fft import fft, ifft
import random
from typing import List, Tuple, Optional
import time
import copy
from functools import lru_cache

# Constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
FFT_THRESHOLD = 100  # Use FFT for sequences longer than this
POPULATION_SIZE = 50
GENERATIONS = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
ELITE_SIZE = 10  # Number of top sequences to preserve

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

@lru_cache(maxsize=1000)
def cached_autocorrelation_constant(sequence_tuple: tuple) -> float:
    """
    Cached version of autocorrelation_constant to speed up repeated evaluations.
    """
    sequence = list(sequence_tuple)
    n = len(sequence)
    if n == 0:
        return 0.0

    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0

    # Compute autoconvolution using FFT for efficiency
    if n > FFT_THRESHOLD:
        # Use FFT for fast convolution
        padded_len = 2 * n - 1
        seq_fft = fft(sequence, padded_len)
        conv_fft = seq_fft * seq_fft.conj()  # Element-wise multiplication
        autoconv = ifft(conv_fft).real
        max_conv = max(autoconv)
    else:
        # Direct convolution for small sequences
        autoconv = signal.convolve(sequence, sequence, mode='full')
        max_conv = max(autoconv)

    # Calculate C₁
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def autocorrelation_constant(sequence: List[float]) -> float:
    """
    Calculates C₁ = 2n * max(b) / (sum(a))^2 where b = a * a (autoconvolution).
    Returns the inverse 1/C₁ which we want to maximize.
    """
    return cached_autocorrelation_constant(tuple(sequence))

def create_individual(length: int) -> List[float]:
    """Create a random individual with given length."""
    return [random.uniform(0.1, 1.0) for _ in range(length)]

def create_population(size: int, min_length: int = MIN_SEQ_LENGTH, max_length: int = MAX_SEQ_LENGTH) -> List[List[float]]:
    """Create an initial population."""
    return [create_individual(random.randint(min_length, max_length)) for _ in range(size)]

def fitness(individual: List[float]) -> float:
    """Evaluate fitness of an individual (inverse of C₁)."""
    return autocorrelation_constant(individual)

def tournament_selection(population: List[List[float]], fitnesses: List[float], tournament_size: int) -> List[float]:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
    """Specialized crossover that respects sequence structure."""
    # If parents have different lengths, pad the shorter one
    max_len = max(len(parent1), len(parent2))
    p1 = parent1 + [0.0] * (max_len - len(parent1))
    p2 = parent2 + [0.0] * (max_len - len(parent2))
    
    # Use uniform crossover
    child1 = []
    child2 = []
    
    for i in range(max_len):
        if random.random() < 0.5:
            child1.append(p1[i])
            child2.append(p2[i])
        else:
            child1.append(p2[i])
            child2.append(p1[i])
    
    # Trim to original lengths (with some variance)
    len1 = random.randint(MIN_SEQ_LENGTH, max_len)
    len2 = random.randint(MIN_SEQ_LENGTH, max_len)
    
    child1 = child1[:len1]
    child2 = child2[:len2]
    
    # Ensure minimum length
    if len(child1) < MIN_SEQ_LENGTH:
        child1.extend([0.0] * (MIN_SEQ_LENGTH - len(child1)))
    if len(child2) < MIN_SEQ_LENGTH:
        child2.extend([0.0] * (MIN_SEQ_LENGTH - len(child2)))
    
    return child1, child2

def mutate(individual: List[float], mutation_rate: float = MUTATION_RATE) -> List[float]:
    """Mutate an individual with careful handling around boundary conditions."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Gaussian mutation for continuous values
            mutated[i] += random.gauss(0, 0.1 * mutated[i])
            mutated[i] = max(0.01, mutated[i])  # Ensure non-negativity
    
    return mutated

def evolve_generation(population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
    """Evolve one generation of the population."""
    new_population = []
    
    # Elitism: keep top individuals
    elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
    elite = [population[i] for i in elite_indices]
    new_population.extend(elite)
    
    # Generate offspring
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
        
        child1, child2 = crossover(parent1, parent2)
        
        child1 = mutate(child1)
        child2 = mutate(child2)
        
        new_population.extend([child1, child2])
    
    # Trim to exact population size
    return new_population[:POPULATION_SIZE]

def search_for_best_sequence() -> List[float]:
    """Main search function using genetic algorithm."""
    global start_time
    start_time = time.time()
    
    # Initialize population
    population = create_population(POPULATION_SIZE)
    
    best_sequence = None
    best_fitness = 0.0
    
    for generation in range(GENERATIONS):
        if time.time() - start_time > MAX_TIME_SECONDS - 2:
            break
            
        # Evaluate fitness
        fitnesses = [fitness(individual) for individual in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_sequence = copy.deepcopy(population[max_fitness_idx])
        
        # Evolve to next generation
        population = evolve_generation(population, fitnesses)
    
    # Final refinement with local search
    if best_sequence is not None:
        try:
            def objective_func(seq_array):
                return -autocorrelation_constant(seq_array.tolist())
            
            x0 = np.array(best_sequence, dtype=float)
            result = optimize.minimize(
                objective_func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 50, 'adaptive': True}
            )
            
            if result.success:
                refined_seq = np.maximum(result.x, 0)
                if np.sum(refined_seq) < 0.01:
                    refined_seq[0] = 0.1
                refined_fitness = autocorrelation_constant(refined_seq.tolist())
                if refined_fitness > best_fitness:
                    best_sequence = refined_seq.tolist()
        except:
            pass
    
    return best_sequence if best_sequence is not None else [0.1] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")