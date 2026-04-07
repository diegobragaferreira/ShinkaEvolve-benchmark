# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit
from joblib import Parallel, delayed
import warnings
from collections import OrderedDict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class FastAutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""
    
    def __init__(self, cache_size: int = 1024):
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = cache_size

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, sequence: List[float]) -> tuple:
        """Generate a hashable cache key."""
        # Round floats to avoid floating point precision issues in caching
        rounded_seq = tuple(round(x, 8) for x in sequence)
        return rounded_seq

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Generate cache key
        key = self._get_cache_key(sequence)
        
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return self._cache[key]

        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (first item in OrderedDict)
            self._cache.popitem(last=False)

        self._cache_misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache[key] = result
            return result

        n = len(sequence)
        # Use FFT-based convolution for efficiency O(n log n)
        conv = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache[key] = result
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache[key] = result
        return result

# Global evaluator instance
_evaluator = FastAutocorrelationEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def compute_fitness_parallel(seqs: List[List[float]]) -> List[float]:
    """Compute fitness scores for a batch of sequences in parallel."""
    # Limit the number of jobs to prevent oversubscription
    n_jobs = min(len(seqs), 8)  # Use at most 8 threads
    return Parallel(n_jobs=n_jobs, backend='threading')(delayed(lambda s: compute_autocorrelation_constant(s)[1])(seq) for seq in seqs)

def generate_step_sequence(length: int, num_steps: int = None) -> List[float]:
    """Generate a step function sequence with random heights."""
    if num_steps is None:
        num_steps = max(2, min(20, length // 10))
    step_positions = sorted(random.sample(range(length), num_steps))
    step_heights = [random.uniform(10.0, 100.0) for _ in range(num_steps)]
    
    sequence = [0.0] * length
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        sequence[pos:end_pos] = [height] * (end_pos - pos)
    return sequence

def generate_gaussian_sequence(length: int) -> List[float]:
    """Generate a Gaussian-like distribution."""
    sequence = [random.gauss(50.0, 20.0) for _ in range(length)]
    return [max(0.01, x) for x in sequence]

def generate_uniform_sequence(length: int) -> List[float]:
    """Generate a uniform random sequence."""
    return [random.uniform(0.1, 100.0) for _ in range(length)]

def generate_pattern_aware_sequence(length: int) -> List[float]:
    """Generate a sequence with known good patterns such as exponential decay."""
    sequence = [0.0] * length
    num_peaks = max(2, min(15, length // 50))
    
    # Place peaks with exponentially decaying heights
    for i in range(num_peaks):
        pos = random.randint(0, length - 1)
        height = random.uniform(50.0, 150.0) * (0.8 ** i)
        sequence[pos] = max(0.01, height)
    
    # Smooth the sequence using moving average
    smoothed = sequence.copy()
    window_size = max(3, length // 100)
    for i in range(len(sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence), i + window_size // 2 + 1)
        smoothed[i] = np.mean(sequence[start:end])
    
    # Ensure all values are positive
    sequence = [max(0.01, x) for x in smoothed]
    return sequence

def generate_diverse_population(population_size: int, length_range=(100, 1000)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []
    
    # Add pattern-aware examples
    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        population.append(generate_pattern_aware_sequence(n))

    # Add step-function examples to encourage structure finding
    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        population.append(generate_step_sequence(n))

    # Add Gaussian examples
    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        population.append(generate_gaussian_sequence(n))

    # Fill remaining with uniform random
    while len(population) < population_size:
        n = random.randint(*length_range)
        population.append(generate_uniform_sequence(n))

    return population

def mutate_sequence(sequence: List[float], generation: int, population_size: int, 
                   diversity_factor: float = 1.0) -> List[float]:
    """Apply adaptive mutation to a sequence with rate based on generation and diversity."""
    mutated = sequence.copy()
    
    # Adaptive mutation rate that decreases with generation and increases with diversity
    base_mutation_rate = 0.3 * (1 - generation / (population_size * 2))
    mutation_rate = max(0.05, base_mutation_rate * diversity_factor)
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian noise scaled by mutation strength
            noise = random.gauss(0, 0.3 * mutated[i])
            mutated[i] = max(0.01, mutated[i] + noise)
            
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover with bias towards preserving better parts
    crossover_point = random.randint(1, min_len - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]
    return child

def calculate_population_diversity(fitness_scores: List[float]) -> float:
    """Calculate diversity of the population based on fitness scores."""
    if len(fitness_scores) <= 1:
        return 0.0
    std_dev = np.std(fitness_scores)
    mean_score = np.mean(fitness_scores) + 1e-10  # Avoid division by zero
    return std_dev / mean_score

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None
        
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraint matrix for convolution
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        # Fallback if optimization fails
        return None

def gradient_improve_sequence(sequence: list[float], step_size: float = 0.01) -> list[float]:
    """Apply simple gradient-like improvement to sequence."""
    improved = sequence.copy()

    # Simple smoothing: adjust towards local average
    for i in range(len(improved)):
        neighbors = []
        if i > 0:
            neighbors.append(improved[i-1])
        if i < len(improved) - 1:
            neighbors.append(improved[i+1])

        if neighbors:
            avg_neighbor = np.mean(neighbors)
            improved[i] = improved[i] * (1 - step_size) + avg_neighbor * step_size

    return improved

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with enhanced local search."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        # Fallback to a simple gradient-based approach
        return gradient_improve_sequence(sequence)

    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def local_search_improvement(sequence: list[float]) -> list[float]:
    """Enhanced local search combining multiple strategies."""
    # Strategy 1: Gradient-based improvement
    improved1 = gradient_improve_sequence(sequence, step_size=0.01)
    _, inv_c1_1 = compute_autocorrelation_constant(improved1)
    
    # Strategy 2: Simple perturbation with optimization
    perturbed = sequence.copy()
    for i in range(len(perturbed)):
        if random.random() < 0.1:  # Small chance to perturb
            perturbed[i] *= random.uniform(0.9, 1.1)
    perturbed = [max(0.01, x) for x in perturbed]
    
    # Try local optimization on perturbed version
    try:
        def objective(x):
            _, inv_c1 = compute_autocorrelation_constant(x)
            return -inv_c1  # Minimize negative to maximize original

        bounds = [(0.01, 1000.0) for _ in range(len(perturbed))]
        result = optimize.minimize(objective, perturbed, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})
        if result.success:
            improved2 = result.x.tolist()
            _, inv_c1_2 = compute_autocorrelation_constant(improved2)
        else:
            improved2 = perturbed
            _, inv_c1_2 = compute_autocorrelation_constant(improved2)
    except:
        improved2 = perturbed
        _, inv_c1_2 = compute_autocorrelation_constant(improved2)
    
    # Choose the best result
    _, inv_c1_original = compute_autocorrelation_constant(sequence)
    if inv_c1_1 >= inv_c1_original and inv_c1_1 >= inv_c1_2:
        return improved1
    elif inv_c1_2 >= inv_c1_original and inv_c1_2 >= inv_c1_1:
        return improved2
    else:
        return sequence

def evolve_sequences_with_adaptive_selection(population_size: int = 50, generations: int = 200):
    """Evolve sequences using adaptive tournament selection."""
    # Define DEAP structures
    from deap import base, creator, tools
    import random
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene generation and individual creation
    def create_individual():
        return creator.Individual(generate_random_sequence())

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation and operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100)
    
    def adaptive_tournament_select(population, k):
        """Adaptive tournament selection with dynamic tournsize."""
        # Start with smaller tournaments for early generations to promote diversity
        # Increase tournament size as generations progress to favor exploitation
        generation = getattr(adaptive_tournament_select, 'generation', 0)
        adaptive_tournament_select.generation = generation + 1

        # Calculate diversity measure (standard deviation of fitness)
        fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
        diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0.0

        # Dynamic tournsize calculation
        min_tour = 2
        max_tour = 10
        if diversity < 0.01 and generation > 100:  # Low diversity and late in evolution
            tournsize = max_tour
        elif diversity > 0.1 and generation < 50:  # High diversity early on
            tournsize = min_tour
        else:
            # Intermediate case: interpolate between min and max
            tournsize = min_tour + int((max_tour - min_tour) * (generation / 200.0))

        # Ensure tournsize stays within bounds
        tournsize = max(min_tour, min(max_tour, tournsize))

        return tools.selTournament(population, k, tournsize=tournsize)

    toolbox.register("select", adaptive_tournament_select)

    # Initialize population
    population = toolbox.population(population_size)

    # Evolution loop
    for gen in range(generations):
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate fitness for new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    return list(best_individual)

def evaluate_individual(individual):
    """Evaluate an individual (sequence) and return fitness."""
    try:
        fitness = compute_autocorrelation_constant(individual)[1]
        return (fitness,)
    except Exception as e:
        return (0.0,)

def generate_random_sequence(min_length=10, max_length=1000):
    """Generate a random sequence within specified constraints."""
    length = random.randint(min_length, max_length)
    # Generate heights in [0, 1000]
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    return sequence

def local_improvement(sequence, max_iter=50):
    """Apply local improvement using gradient-based method."""
    current_seq = sequence[:]
    best_score = compute_autocorrelation_constant(current_seq)[1]
    best_seq = current_seq[:]

    for _ in range(max_iter):
        improved_seq = get_good_direction_to_move_into(current_seq)
        if improved_seq is None:
            break
        new_score = compute_autocorrelation_constant(improved_seq)[1]
        if new_score > best_score:
            best_score = new_score
            best_seq = improved_seq[:]
            current_seq = improved_seq[:]
        else:
            break
    return best_seq

def get_good_direction_to_move_into_lp(sequence):
    """Use LP-based approach for direction finding."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = np.array(sequence) / sum_sequence
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence, mode='full'))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun == 0:
        return None

    normalized_g_fun = np.array(g_fun) / sum_g_fun
    t = 0.01
    new_sequence = (1 - t) * np.array(sequence) + t * normalized_g_fun * sum_sequence
    new_sequence = np.clip(new_sequence, 0, 1000)
    return new_sequence.tolist()

def initialize_good_sequence(length=None):
    """Initialize a good starting sequence with exponential decay."""
    if length is None:
        length = random.randint(100, 1000)

    # Create a sequence with exponential decay to balance mass and convolution
    # This helps to reduce the peak convolution while maintaining significant total mass
    sequence = [1000 * np.exp(-i/10) for i in range(length)]

    # Normalize to have reasonable total mass
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]

    return sequence

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator.clear_cache()

    # Initialize population with diverse strategies
    population_size = 30
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 30
    diversity_threshold = 0.05  # Minimum diversity to maintain exploration

    # Adjusted population size for early generations to allow for exploration
    adjusted_population_size = population_size

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1
        
        # Adjust population size based on generation to shift from exploration to exploitation
        if generation > 20:
            adjusted_population_size = max(20, population_size // 2)
        else:
            adjusted_population_size = population_size

        # Evaluate fitness (1/C1) of each individual in parallel
        fitness_scores = compute_fitness_parallel(population)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Apply local search to the best sequence
        if best_sequence is not None and current_best_inv_c1 > 0.5:  # Only apply if reasonably good
            local_search_result = local_search_improvement(best_sequence)
            _, local_inv_c1 = compute_autocorrelation_constant(local_search_result)
            if local_inv_c1 > best_inv_c1:
                best_inv_c1 = local_inv_c1
                best_sequence = local_search_result
                stagnation_count = 0

        # Calculate population diversity
        diversity = calculate_population_diversity(fitness_scores)
        
        # Inject new variation if diversity is too low
        if diversity < diversity_threshold and generation > 5:
            num_new = max(1, adjusted_population_size // 6)
            for _ in range(num_new):
                new_seq = generate_pattern_aware_sequence(random.randint(100, 1000))
                population[random.randint(0, len(population)-1)] = new_seq

        # Selection with tournament selection and elitism
        selected_parents = []
        tournament_size = 5  # Larger tournament for more selection pressure

        # Elitism: keep the top performer
        elite_idx = current_best_idx
        selected_parents.append(population[elite_idx].copy())

        # Tournament selection for rest
        for _ in range(adjusted_population_size - 1):  # -1 because we already added elite
            tournament_indices = random.sample(range(adjusted_population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx].copy())

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism: keep best individual

        while len(new_population) < adjusted_population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate_sequence(child, generation, population_size, diversity)

            new_population.append(child)

        # Prune population to adjusted size
        population = new_population[:adjusted_population_size]

        # Early termination if target is reached
        if best_inv_c1 > 0.6653:  # Benchmark value
            break

    # Final cleanup and validation
    if best_sequence is not None and sum(best_sequence) > 0.01:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_uniform_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_uniform_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")