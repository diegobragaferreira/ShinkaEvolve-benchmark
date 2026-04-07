# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time
import random
import math
from functools import lru_cache
from typing import List, Tuple, Deque
from collections import deque
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants with caching."""
    
    def __init__(self, cache_size: int = 512):
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(round(x, 8) for x in sequence)
        
        if seq_tuple in self.cache:
            self.hits += 1
            return self.cache[seq_tuple]
            
        self.misses += 1
        
        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self.cache[seq_tuple] = result
            return result

        n = len(sequence)
        
        # Use FFT-based convolution for efficiency O(n log n)
        conv = fftconvolve(sequence, sequence, mode='full')
        # Extract the valid convolution part
        max_conv = np.max(conv[n-1:2*n-1])
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self.cache[seq_tuple] = result
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (first item in dict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[seq_tuple] = result
        return result

class StepFunctionGenerator:
    """Generates various types of step function sequences."""
    
    @staticmethod
    def generate_step_sequence(length: int, num_steps: int = None) -> List[float]:
        """Generate a step function with random heights."""
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

    @staticmethod
    def generate_gaussian_sequence(length: int) -> List[float]:
        """Generate a Gaussian-like distribution."""
        sequence = [random.gauss(50.0, 20.0) for _ in range(length)]
        return [max(0.01, x) for x in sequence]

    @staticmethod
    def generate_uniform_sequence(length: int) -> List[float]:
        """Generate a uniform random sequence."""
        return [random.uniform(0.1, 100.0) for _ in range(length)]

    @staticmethod
    def generate_exponential_sequence(length: int) -> List[float]:
        """Generate an exponential decay sequence."""
        return [1000 * np.exp(-i/10) for i in range(length)]

    @staticmethod
    def generate_pattern_sequence(length: int) -> List[float]:
        """Generate a sequence with known good patterns."""
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

    @classmethod
    def generate_diverse_sequence(cls, length_range=(100, 1000)) -> List[float]:
        """Generate a diverse sequence using various pattern generators."""
        n = random.randint(*length_range)
        method = random.choice([
            cls.generate_step_sequence,
            cls.generate_gaussian_sequence,
            cls.generate_uniform_sequence,
            cls.generate_exponential_sequence,
            cls.generate_pattern_sequence
        ])
        return method(n)

class EvolutionaryOptimizer:
    """Core evolutionary optimization engine."""
    
    def __init__(self, pop_size: int = 50, max_generations: int = 100):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.evaluator = AutocorrelationEvaluator()
        self.best_sequence = None
        self.best_fitness = 0.0

    def _evaluate_fitness_batch(self, sequences: List[List[float]]) -> List[float]:
        """Evaluate multiple sequences in parallel."""
        return Parallel(n_jobs=-1)(
            delayed(self.evaluator.evaluate)(seq)[1] for seq in sequences
        )

    def _compute_diversity(self, fitness_scores: List[float]) -> float:
        """Calculate population diversity."""
        if len(fitness_scores) <= 1:
            return 0.0
        return np.std(fitness_scores) / (np.mean(fitness_scores) + 1e-10)

    def _adapt_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adapt mutation rate based on generation and population diversity."""
        base_rate = 0.3 * (1 - generation / self.max_generations)
        return max(0.05, base_rate * (1 + diversity))

    def _tournament_selection(self, population: List[List[float]], 
                            fitness_scores: List[float], 
                            tournament_size: int = 5) -> List[List[float]]:
        """Perform tournament selection."""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, seq1: List[float], seq2: List[float]) -> List[float]:
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

    def _mutate(self, sequence: List[float], mutation_rate: float, 
               mutation_strength: float = 0.3) -> List[float]:
        """Apply mutation to a sequence."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply Gaussian noise scaled by mutation strength
                noise = random.gauss(0, mutation_strength * mutated[i])
                mutated[i] = max(0.01, mutated[i] + noise)
        return mutated

    def _gradient_refinement(self, sequence: List[float], iterations: int = 20) -> List[float]:
        """Refine sequence using gradient information."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)
        
        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq

        # Initialize momentum term
        velocity = np.zeros_like(seq)
        momentum = 0.9
        learning_rate = 1e-4

        for step in range(iterations):
            # Compute convolution
            conv = fftconvolve(seq, seq, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = np.max(conv)

            # Compute gradient estimate using central differences
            grad = np.zeros_like(seq)
            epsilon = 1e-6
            for i in range(n):
                # Perturb forward and backward
                seq_forward = seq.copy()
                seq_forward[i] += epsilon
                seq_forward /= np.sum(seq_forward)

                seq_backward = seq.copy()
                seq_backward[i] -= epsilon
                seq_backward /= np.sum(seq_backward)

                # Compute convolution for both perturbed sequences
                conv_forward = fftconvolve(seq_forward, seq_forward, mode='full')
                conv_forward = conv_forward[n-1:2*n-1]
                max_forward = np.max(conv_forward)

                conv_backward = fftconvolve(seq_backward, seq_backward, mode='full')
                conv_backward = conv_backward[n-1:2*n-1]
                max_backward = np.max(conv_backward)

                # Central difference gradient estimate
                grad[i] = (max_forward - max_backward) / (2 * epsilon)

            # Adaptive learning rate with decay
            adaptive_lr = learning_rate * (1.0 - 0.9 * (step / iterations))

            # Momentum update
            velocity = momentum * velocity + adaptive_lr * grad

            # Update using gradient ascent with momentum
            seq += velocity
            seq = np.maximum(seq, 0)  # Ensure non-negative

            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq

        return seq.tolist()

    def optimize(self, max_time_seconds: int = 170) -> List[float]:
        """Run the evolutionary optimization process."""
        start_time = time.time()
        
        # Initialize population
        population = [StepFunctionGenerator.generate_diverse_sequence() 
                     for _ in range(self.pop_size)]
        
        # Evaluate initial population
        fitness_scores = self._evaluate_fitness_batch(population)
        
        # Track best solution
        best_idx = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_idx]
        self.best_sequence = population[best_idx].copy()
        
        print(f"Initial best fitness: {self.best_fitness:.6f}")
        
        # Tracks for convergence detection
        fitness_history = deque([self.best_fitness], maxlen=10)
        stagnation_counter = 0
        max_stagnation = 20
        
        # Run evolution
        for generation in range(self.max_generations):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Compute diversity
            diversity = self._compute_diversity(fitness_scores)
            
            # Selection
            selected = self._tournament_selection(population, fitness_scores)
            
            # Create offspring
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1 = self._crossover(selected[i], selected[i+1])
                    child2 = self._crossover(selected[i+1], selected[i])
                    
                    # Mutation with adaptive rate
                    mut_rate = self._adapt_mutation_rate(generation, diversity)
                    child1 = self._mutate(child1, mut_rate)
                    child2 = self._mutate(child2, mut_rate)
                    
                    offspring.extend([child1, child2])
                else:
                    # Handle odd population size
                    child = self._crossover(selected[i], selected[i])
                    mut_rate = self._adapt_mutation_rate(generation, diversity)
                    child = self._mutate(child, mut_rate)
                    offspring.append(child)

            # Keep offspring size consistent
            offspring = offspring[:self.pop_size]

            # Apply gradient refinement to top individuals
            top_indices = np.argsort(fitness_scores)[-min(10, len(offspring)):][::-1]
            refine_count = max(1, int(0.3 * len(top_indices)))
            for i in range(refine_count):
                idx = top_indices[i]
                offspring[idx] = self._gradient_refinement(offspring[idx])

            # Evaluate offspring
            offspring_fitnesses = self._evaluate_fitness_batch(offspring)

            # Replace population with offspring
            population = offspring
            fitness_scores = offspring_fitnesses

            # Update best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_sequence = population[best_idx].copy()
                stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                stagnation_counter += 1

            # Update history for convergence detection
            fitness_history.append(self.best_fitness)

            # Adaptive stopping conditions
            if generation > 10:
                # Check for fitness plateau
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < 1e-8:
                    print("Fitness plateau detected, stopping early")
                    break
                    
                # Check for stagnation
                if stagnation_counter > max_stagnation:
                    print("Stagnation detected, stopping early")
                    break

            # Print progress
            if generation % 10 == 0:
                print(f"Gen {generation}: Best = {self.best_fitness:.6f}, Diversity = {diversity:.6f}")

        # Final cleanup
        if self.best_sequence is not None:
            # Normalize to ensure meaningful values
            sum_seq = sum(self.best_sequence)
            if sum_seq > 0.01:
                self.best_sequence = [x / sum_seq * 100 for x in self.best_sequence]

        return self.best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        optimizer = EvolutionaryOptimizer(pop_size=50, max_generations=100)
        best_sequence = optimizer.optimize(max_time_seconds=170)
        return best_sequence
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to simple approach
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
