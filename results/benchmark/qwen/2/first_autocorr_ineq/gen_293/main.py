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
INITIAL_MUTATION_RATE = 0.1
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

def mutate(individual: List[float], mutation_rate: float) -> List[float]:
    """Mutate an individual with careful handling around boundary conditions."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Gaussian mutation for continuous values
            mutated[i] += random.gauss(0, 0.1 * mutated[i])
            mutated[i] = max(0.01, mutated[i])  # Ensure non-negativity
    
    return mutated

def compute_gradient_approximation(sequence: List[float], epsilon_base: float = 1e-4) -> List[float]:
    """
    Approximate gradient using symmetric finite differences with adaptive epsilon.
    """
    n = len(sequence)
    grad = []
    for i in range(n):
        # Determine adaptive epsilon based on element magnitude
        elem_mag = abs(sequence[i])
        if elem_mag < 1e-6:
            epsilon = epsilon_base
        else:
            epsilon = epsilon_base * elem_mag

        # Perturb dimension i symmetrically
        perturbed_plus = sequence[:]
        perturbed_minus = sequence[:]
        perturbed_plus[i] += epsilon
        perturbed_minus[i] -= epsilon

        # Ensure non-negativity
        perturbed_plus[i] = max(0, perturbed_plus[i])
        perturbed_minus[i] = max(0, perturbed_minus[i])

        # Evaluate function
        f_plus = autocorrelation_constant(perturbed_plus)
        f_minus = autocorrelation_constant(perturbed_minus)

        grad_i = (f_plus - f_minus) / (2 * epsilon)
        grad.append(grad_i)

    return grad

def solve_convolution_lp(f_sequence: list[float], rhs: float) -> list[float] | None:
    """
    Solves the convolution LP for a given sequence and RHS with enhanced numerical stability.
    """
    try:
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []
        
        # Efficiently sample key convolution constraints
        # Use fewer constraints to reduce memory and improve performance
        constraint_indices = np.random.choice(np.arange(2 * n - 1), size=min(1000, 2 * n - 1), replace=False)
        for k in sorted(constraint_indices):
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

        # Solve with more robust options
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': False})
        
        if result.success:
            g_sequence = result.x
            return g_sequence.tolist()
        else:
            return None
    except Exception:
        return None

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """
    Returns the direction to move into the sequence using gradient and LP optimization.
    """
    try:
        n = len(sequence)
        if n == 0:
            return None

        # Normalize the input sequence
        sum_sequence = sum(sequence)
        if sum_sequence < 0.01:
            sum_sequence = 0.01
        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Compute the right-hand side for convolution constraints
        max_conv_value = max(signal.convolve(normalized_sequence, normalized_sequence, mode='full')[n-1:])
        rhs = max_conv_value

        # Solve the LP to get the direction vector g
        g_fun = solve_convolution_lp(normalized_sequence, rhs)
        
        if g_fun is None:
            # Fallback to gradient ascent if LP fails
            try:
                grad = compute_gradient_approximation(sequence)
                step_size = 0.01
                new_sequence = [max(0, sequence[i] + step_size * grad[i]) for i in range(len(sequence))]
                return new_sequence
            except:
                return None
            return None

        # Normalize g_fun similarly
        sum_g_fun = sum(g_fun)
        if sum_g_fun < 0.01:
            sum_g_fun = 0.01
        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

        # Update sequence using a fixed step size (t=0.01)
        t = 0.01
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        return new_sequence
    except Exception:
        return None

def refine_with_local_search(individual: List[float], max_iter: int = 50) -> List[float]:
    """
    Refine individual using local search techniques.
    """
    current = individual.copy()
    current_fitness = fitness(current)
    
    for _ in range(max_iter):
        # Try gradient-based update
        direction = get_good_direction_to_move_into(current)
        if direction is not None:
            new_fitness = fitness(direction)
            if new_fitness > current_fitness:
                current = direction
                current_fitness = new_fitness
            else:
                # Try another approach if gradient update didn't help
                try:
                    # Simple perturbation
                    perturbed = [max(0.01, x + random.gauss(0, 0.01)) for x in current]
                    perturbed_fitness = fitness(perturbed)
                    if perturbed_fitness > current_fitness:
                        current = perturbed
                        current_fitness = perturbed_fitness
                except:
                    pass
        else:
            # Try simple perturbation if direction finding fails
            try:
                perturbed = [max(0.01, x + random.gauss(0, 0.01)) for x in current]
                perturbed_fitness = fitness(perturbed)
                if perturbed_fitness > current_fitness:
                    current = perturbed
                    current_fitness = perturbed_fitness
            except:
                pass
    
    return current

def evolve_generation(population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
    """Evolve one generation of the population with enhanced strategies."""
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
        
        # Adaptive mutation rate
        mutation_rate = INITIAL_MUTATION_RATE * (1 - len(new_population) / POPULATION_SIZE)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        new_population.extend([child1, child2])
    
    # Trim to exact population size
    return new_population[:POPULATION_SIZE]

def search_for_best_sequence() -> List[float]:
    """Main search function using genetic algorithm with enhanced local search."""
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
        
        # Apply local search to top individuals in later generations
        if generation >= GENERATIONS // 2:
            for i in range(0, min(5, len(population)), 2):
                population[i] = refine_with_local_search(population[i])
    
    # Final refinement with local search
    if best_sequence is not None:
        try:
            refined_seq = refine_with_local_search(best_sequence, 100)
            refined_fitness = fitness(refined_seq)
            if refined_fitness > best_fitness:
                best_sequence = refined_seq
        except:
            pass
    
    return best_sequence if best_sequence is not None else [0.1] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")