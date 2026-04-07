# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
from collections import OrderedDict
from joblib import Parallel, delayed

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class LRUCache:
    """LRU Cache implementation for storing previous computations."""
    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

class AutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""
    
    def __init__(self):
        self._cache = LRUCache(maxsize=256)
        self.hits = 0
        self.misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache = LRUCache(maxsize=256)
        self.hits = 0
        self.misses = 0

    def compute(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        cached_result = self._cache.get(seq_tuple)
        if cached_result is not None:
            self.hits += 1
            return cached_result

        self.misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache.put(seq_tuple, result)
            return result

        n = len(sequence)
        # Use FFT-based convolution for efficiency O(n log n)
        conv = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache.put(seq_tuple, result)
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache.put(seq_tuple, result)
        return result

# Global evaluator instance
_evaluator = AutocorrelationEvaluator()

def fitness_evaluator(sequence: List[float]) -> float:
    """Compute fitness (1/C1) for a sequence."""
    _, inv_c1 = _evaluator.compute(sequence)
    return inv_c1

def fitness_batch_evaluator(sequences: List[List[float]]) -> List[float]:
    """Evaluate fitness for a batch of sequences in parallel."""
    return Parallel(n_jobs=-1)(
        delayed(lambda s: _evaluator.compute(s)[1])(seq) for seq in sequences
    )

class SequenceInitializer:
    """Factory class for creating diverse initial sequences."""
    
    @staticmethod
    def harmonic_sequence(length: int) -> List[float]:
        """Generate a harmonic-like sequence."""
        sequence = []
        for i in range(length):
            harmonic = 1.0 / (1 + i * 0.1) * np.sin(i * 0.5)
            sequence.append(max(0.01, abs(harmonic) * 100))
        return sequence

    @staticmethod
    def exponential_sequence(length: int) -> List[float]:
        """Generate an exponentially decaying sequence."""
        sequence = []
        for i in range(length):
            val = 100 * np.exp(-i * 0.02)
            sequence.append(max(0.01, val))
        return sequence

    @staticmethod
    def step_sequence(length: int, num_steps: int) -> List[float]:
        """Generate a step function sequence."""
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

    @staticmethod
    def gaussian_sequence(length: int) -> List[float]:
        """Generate a Gaussian-like distribution."""
        sequence = [random.gauss(50.0, 20.0) for _ in range(length)]
        return [max(0.01, x) for x in sequence]

    @staticmethod
    def uniform_sequence(length: int) -> List[float]:
        """Generate a uniform random sequence."""
        return [random.uniform(0.1, 100.0) for _ in range(length)]

    @classmethod
    def generate_diverse_population(cls, 
                                  population_size: int, 
                                  length_range: Tuple[int, int] = (100, 1000)
                                  ) -> List[List[float]]:
        """Generate a diverse initial population with various patterns."""
        population = []
        
        # Generate sequences using different methods
        for _ in range(population_size // 6):
            n = random.randint(*length_range)
            population.append(cls.harmonic_sequence(n))

        for _ in range(population_size // 6):
            n = random.randint(*length_range)
            population.append(cls.exponential_sequence(n))

        for _ in range(population_size // 6):
            n = random.randint(*length_range)
            num_steps = max(2, min(20, n // 10))
            population.append(cls.step_sequence(n, num_steps))

        # Add some step-function examples
        for _ in range(population_size // 6):
            n = random.randint(*length_range)
            num_steps = max(2, min(20, n // 10))
            population.append(cls.step_sequence(n, num_steps))

        # Add some Gaussian examples
        for _ in range(population_size // 6):
            n = random.randint(*length_range)
            population.append(cls.gaussian_sequence(n))

        # Fill remaining with standard random
        while len(population) < population_size:
            n = random.randint(*length_range)
            population.append(cls.uniform_sequence(n))

        return population

class MutationStrategy:
    """Handles evolutionary mutation operations."""
    
    @staticmethod
    def adaptive_mutation(sequence: List[float], 
                         generation: int, 
                         population_size: int, 
                         mutation_strength: float = 0.3) -> List[float]:
        """Apply adaptive mutation to a sequence with decreasing rate over generations."""
        mutated = sequence.copy()
        # Adaptive mutation rate that decreases with generation
        mutation_rate = max(0.05, 0.3 * (1 - generation / (population_size * 2)))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply Gaussian noise scaled by mutation strength
                noise = random.gauss(0, mutation_strength * mutated[i])
                mutated[i] = max(0.01, mutated[i] + noise)
        return mutated

    @staticmethod
    def crossover(seq1: List[float], seq2: List[float]) -> List[float]:
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

class GradientLocalSearch:
    """Handles local search strategies including gradient-based improvements."""
    
    @staticmethod
    def get_good_direction_to_move_into(sequence: List[float]) -> List[float] | None:
        """Returns the direction to move into the sequence using a specialized optimization."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)
        
        # Avoid division by zero
        if sum_sequence < 1e-10:
            return None

        # Use a direct gradient descent approach to improve the sequence
        try:
            # Approximate gradient using finite differences
            epsilon = 1e-4
            base_fitness = 1.0 / _evaluator.compute(sequence)[0] if _evaluator.compute(sequence)[0] != float('inf') else 0.0
            
            # Compute gradient per dimension
            grad = np.zeros(n)
            for i in range(n):
                perturbed = sequence[:]
                perturbed[i] = max(0.01, perturbed[i] + epsilon)
                perturbed_fitness = 1.0 / _evaluator.compute(perturbed)[0] if _evaluator.compute(perturbed)[0] != float('inf') else 0.0
                grad[i] = (perturbed_fitness - base_fitness) / epsilon
            
            # Update in the direction of gradient ascent
            step_size = 0.1
            updated = [max(0.01, sequence[i] + step_size * grad[i]) for i in range(n)]
            
            return updated
        except Exception:
            # Fallback to simple perturbation
            return [max(0.01, x * random.uniform(0.95, 1.05)) for x in sequence]

    @staticmethod
    def multi_start_local_search(sequence: List[float], max_starts: int = 5) -> List[float]:
        """Perform multi-start local search to find better local optima."""
        best_sequence = sequence.copy()
        best_fitness = fitness_evaluator(best_sequence)

        # Try multiple perturbations and local searches
        for _ in range(max_starts):
            # Create a slightly perturbed version of the sequence
            perturbed = [x * random.uniform(0.95, 1.05) for x in sequence]
            # Ensure non-negative values
            perturbed = [max(0.01, x) for x in perturbed]

            # Apply local search to the perturbed sequence
            improved = GradientLocalSearch.get_good_direction_to_move_into(perturbed)
            if improved is not None:
                # Evaluate the improved sequence
                inv_c1 = fitness_evaluator(improved)
                if inv_c1 > best_fitness:
                    best_fitness = inv_c1
                    best_sequence = improved

        return best_sequence

class EvolutionController:
    """Main evolution controller that orchestrates the evolutionary process."""
    
    def __init__(self, population_size: int = 30, max_time_seconds: int = 170):
        self.population_size = population_size
        self.max_time_seconds = max_time_seconds
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.stagnation_count = 0
        self.max_stagnation = 30
        self.generation = 0

    def evolve(self) -> List[float]:
        """Run the evolutionary optimization process."""
        _evaluator.clear_cache()
        
        # Initialize population
        population = SequenceInitializer.generate_diverse_population(
            self.population_size, (100, 1000)
        )

        start_time = time.time()
        
        while time.time() - start_time < self.max_time_seconds and self.stagnation_count < self.max_stagnation:
            self.generation += 1
            
            # Evaluate fitness for all individuals in parallel
            fitness_scores = fitness_batch_evaluator(population)

            # Track best individual
            current_best_idx = np.argmax(fitness_scores)
            current_best_inv_c1 = fitness_scores[current_best_idx]

            if current_best_inv_c1 > self.best_inv_c1:
                self.best_inv_c1 = current_best_inv_c1
                self.best_sequence = population[current_best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Apply enhanced local search to the best sequence
            if self.best_sequence is not None:
                local_search_result = GradientLocalSearch.multi_start_local_search(self.best_sequence)
                if local_search_result is not None:
                    # Evaluate the local search result
                    local_inv_c1 = fitness_evaluator(local_search_result)
                    if local_inv_c1 > self.best_inv_c1:
                        self.best_inv_c1 = local_inv_c1
                        self.best_sequence = local_search_result
                        self.stagnation_count = 0

            # Selection with tournament selection and elitism
            selected_parents = self._tournament_selection(population, fitness_scores)

            # Create new population through crossover and mutation
            new_population = [self.best_sequence.copy()]  # Elitism: keep best individual

            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_parents)
                parent2 = random.choice(selected_parents)

                # Crossover
                child = MutationStrategy.crossover(parent1, parent2)

                # Mutation with adaptive rate
                child = MutationStrategy.adaptive_mutation(
                    child, self.generation, self.population_size, mutation_strength=0.2
                )

                new_population.append(child)

            population = new_population[:self.population_size]

        # Final cleanup and validation
        if self.best_sequence is not None:
            sum_seq = sum(self.best_sequence)
            if sum_seq > 0.01:
                self.best_sequence = [x / sum_seq * 100 for x in self.best_sequence]

        return self.best_sequence if self.best_sequence else SequenceInitializer.uniform_sequence(100)

    def _tournament_selection(self, population: List[List[float]], 
                             fitness_scores: List[float]) -> List[List[float]]:
        """Perform tournament selection."""
        selected = []
        tournament_size = 5  # Larger tournament for more selection pressure

        # Elitism: keep the top performer
        elite_idx = np.argmax(fitness_scores)
        selected.append(population[elite_idx].copy())

        # Tournament selection for rest
        for _ in range(self.population_size - 1):  # -1 because we already added elite
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach
        controller = EvolutionController(population_size=30, max_time_seconds=170)
        best_sequence = controller.evolve()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return SequenceInitializer.uniform_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")