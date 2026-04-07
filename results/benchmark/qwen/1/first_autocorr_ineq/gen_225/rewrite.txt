# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.signal import fftconvolve
from deap import base, creator, tools, algorithms
import time
from scipy import optimize
import functools
import warnings

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationEvaluator:
    """Encapsulates evaluation logic with caching."""
    
    def __init__(self):
        self._cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def clear_cache(self):
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    @functools.lru_cache(maxsize=10000)
    def _compute_with_cache(self, sequence_tuple):
        return self._compute_raw(list(sequence_tuple))
    
    def evaluate(self, sequence):
        try:
            return self._compute_with_cache(tuple(sequence))
        except Exception:
            return 0.0
    
    def _compute_raw(self, sequence):
        if len(sequence) == 0 or np.sum(sequence) < 0.01:
            return 0.0
        
        conv = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(conv[len(sequence)-1:])
        sum_a = np.sum(sequence)
        n = len(sequence)
        
        if sum_a == 0:
            return 0.0
        
        C1 = 2 * n * max_conv / (sum_a ** 2)
        return 1 / C1

# Global evaluator instance
evaluator = AutocorrelationEvaluator()

def evaluate_individual(individual):
    """Evaluate an individual (sequence) and return fitness."""
    try:
        fitness = evaluator.evaluate(individual)
        return (fitness,)
    except Exception as e:
        return (0.0,)

class SequenceGenerator:
    """Generates various types of sequences for initial population."""
    
    @staticmethod
    def random_sequence(min_length=10, max_length=1000):
        length = random.randint(min_length, max_length)
        sequence = [random.uniform(0, 1000) for _ in range(length)]
        return sequence
    
    @staticmethod
    def exponential_decay_sequence(length=None):
        if length is None:
            length = random.randint(100, 1000)
        sequence = [1000 * np.exp(-i/10) for i in range(length)]
        total_mass = sum(sequence)
        if total_mass > 0:
            sequence = [x / total_mass * 100 for x in sequence]
        return sequence
    
    @staticmethod
    def step_sequence(length=None, num_steps=None):
        if length is None:
            length = random.randint(100, 1000)
        if num_steps is None:
            num_steps = max(2, min(20, length // 10))
        step_positions = sorted(random.sample(range(length), num_steps))
        step_heights = [random.uniform(0.1, 100.0) for _ in range(num_steps)]
        
        sequence = [0.0] * length
        for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
            if i < len(step_positions) - 1:
                end_pos = step_positions[i+1]
            else:
                end_pos = length
            sequence[pos:end_pos] = [height] * (end_pos - pos)
        return sequence

class LocalImprover:
    """Implements local search strategies for fine-tuning sequences."""
    
    @staticmethod
    def gradient_based_improvement(sequence, max_iter=50):
        current_seq = sequence[:]
        best_score = evaluator.evaluate(current_seq)
        best_seq = current_seq[:]
        
        for _ in range(max_iter):
            improved_seq = LocalImprover._get_gradient_direction(current_seq)
            if improved_seq is None:
                break
            new_score = evaluator.evaluate(improved_seq)
            if new_score > best_score:
                best_score = new_score
                best_seq = improved_seq[:]
                current_seq = improved_seq[:]
            else:
                break
        return best_seq
    
    @staticmethod
    def _get_gradient_direction(sequence):
        n = len(sequence)
        if n == 0:
            return None

        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        normalized_sequence = np.array(sequence) / sum_sequence
        current_value = evaluator.evaluate(sequence)

        epsilon = 1e-4
        step_direction = np.zeros(n)

        for i in range(n):
            perturbed_sequence = normalized_sequence.copy()
            perturbed_sequence[i] += epsilon
            new_value = evaluator.evaluate((perturbed_sequence * sum_sequence).tolist())
            step_direction[i] = (new_value - current_value) / epsilon

        step_norm = np.linalg.norm(step_direction)
        if step_norm > 0:
            step_direction = step_direction / step_norm

        t = 0.01
        new_sequence = (1 - t) * np.array(sequence) + t * step_direction * sum_sequence
        new_sequence = np.clip(new_sequence, 0, 1000)

        return new_sequence.tolist()

class EvolutionEngine:
    """Handles the genetic algorithm evolution process."""
    
    def __init__(self, pop_size=50, generations=200):
        self.pop_size = pop_size
        self.generations = generations
        self.toolbox = base.Toolbox()
        self._setup_toolbox()
    
    def _setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100)
        self.toolbox.register("select", self._adaptive_selection)
    
    def _create_individual(self):
        return creator.Individual(SequenceGenerator.random_sequence())
    
    def _adaptive_selection(self, population, k):
        # Start with smaller tournaments for early generations to promote diversity
        # Increase tournament size as generations progress to favor exploitation
        generation = getattr(self._adaptive_selection, 'generation', 0)
        self._adaptive_selection.generation = generation + 1

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
    
    def run_evolution(self, initial_population):
        population = initial_population
        stagnation_count = 0
        max_stagnation = 10
        fitness_history = []
        
        for gen in range(self.generations):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            
            # Track stagnation and adaptively adjust
            current_fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
            if len(current_fitnesses) > 0:
                avg_fitness = np.mean(current_fitnesses)
                fitness_history.append(avg_fitness)

                if len(fitness_history) >= 2:
                    recent_change = abs(fitness_history[-1] - fitness_history[-2])
                    if recent_change < 1e-5:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                
                if stagnation_count > max_stagnation:
                    # Increase diversity by adding new individuals
                    new_pop_size = min(100, len(population) + 10)
                    additional_individuals = self.toolbox.population(new_pop_size - len(population))
                    population.extend(additional_individuals)
                    stagnation_count = 0

        return tools.selBest(population, 1)[0]

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

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    """
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except Exception:
        warnings.warn("Linear Programming failed.")
        return None

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        warnings.warn("Linear Programming not successful.")
        return None

def search_for_best_sequence() -> list[float]:
    """Main function to find the best coefficient sequence."""
    start_time = time.time()
    evaluator.clear_cache()

    # Multi-start approach: try several initialization strategies
    best_score = 0
    best_sequence = None
    strategies = [
        SequenceGenerator.random_sequence,
        SequenceGenerator.exponential_decay_sequence,
        SequenceGenerator.step_sequence,
        lambda: [1.0] * random.randint(100, 1000),
    ]

    for strategy in strategies:
        sequence = strategy()
        sequence = LocalImprover.gradient_based_improvement(sequence, max_iter=20)
        lp_improved = get_good_direction_to_move_into_lp(sequence)
        if lp_improved is not None:
            sequence = lp_improved

        try:
            # Initialize evolution with the current sequence
            initial_pop = [sequence]
            for _ in range(49):  # Fill up to pop size
                initial_pop.append(SequenceGenerator.random_sequence())
            
            engine = EvolutionEngine()
            evolved_seq = engine.run_evolution(initial_pop)
            evolved_seq = LocalImprover.gradient_based_improvement(list(evolved_seq), max_iter=10)
            score = evaluator.evaluate(evolved_seq)
            if score > best_score:
                best_score = score
                best_sequence = evolved_seq[:]
        except Exception as e:
            continue  # Continue to next strategy if evolution fails

    # Fallback to a basic approach if nothing worked
    if best_sequence is None:
        best_sequence = [1.0] * 100

    # Final verification and cleanup
    if len(best_sequence) == 0 or np.sum(best_sequence) < 0.01:
        best_sequence = [1.0]

    # Limit size to prevent excessive computation
    if len(best_sequence) > 1000:
        best_sequence = best_sequence[:1000]

    # Clip values to [0, 1000] for practicality
    best_sequence = [max(0, min(1000, x)) for x in best_sequence]

    elapsed = time.time() - start_time
    if elapsed > 170:
        return best_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")