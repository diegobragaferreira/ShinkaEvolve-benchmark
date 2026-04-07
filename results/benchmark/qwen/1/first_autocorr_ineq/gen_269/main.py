# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial
import copy
from typing import List, Tuple, Optional

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    """
    An evolutionary optimizer designed to maximize 1/C₁ for step function sequences.
    """

    def __init__(self, pop_size: int = 50, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.best_score = 0.0
        self.best_sequence = None

    @staticmethod
    def evaluate_sequence(sequence: List[float]) -> Tuple[float, float, float, float]:
        """
        Evaluate a sequence and return its performance metrics.

        Args:
            sequence: List of non-negative real numbers representing step heights

        Returns:
            tuple: (C₁, 1/C₁, max_convolution_value, sum_of_sequence)
        """
        try:
            # Convert to numpy array
            a = np.array(sequence)
            sum_a = np.sum(a)

            # Avoid division by zero or negligible sums
            if sum_a < 1e-10:
                return float('inf'), 0.0, 0.0, sum_a

            # Compute autoconvolution using FFT for efficiency
            b = fftconvolve(a, a, mode='full')
            b = b[len(a)-1:2*len(a)-1]  # Convolution part

            max_b = np.max(b)

            # Compute C₁ = 2n * max(b) / (sum(a))^2
            n = len(a)
            c1 = 2 * n * max_b / (sum_a ** 2)

            # Return inverse for maximization
            inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

            return c1, inv_c1, max_b, sum_a
        except Exception as e:
            return float('inf'), 0.0, 0.0, 0.0

    @staticmethod
    def evaluate_population_parallel(population: List[List[float]],
                                   chunk_size: int = 10) -> List[float]:
        """
        Evaluate a list of sequences in parallel.

        Args:
            population: List of sequences to evaluate
            chunk_size: Number of sequences per worker

        Returns:
            List of performance metrics (1/C₁) for each sequence
        """
        # Split the population into chunks
        chunks = [population[i:i+chunk_size] for i in range(0, len(population), chunk_size)]

        # Use multiprocessing to evaluate chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(AutocorrelationOptimizer._evaluate_chunk, chunks)

        # Flatten results
        flattened_results = [item for sublist in results for item in sublist]
        return flattened_results

    @staticmethod
    def _evaluate_chunk(chunk: List[List[float]]) -> List[float]:
        """Helper function to evaluate a chunk of sequences."""
        return [AutocorrelationOptimizer.evaluate_sequence(seq)[1] for seq in chunk]

    @staticmethod
    def generate_random_valid_sequence() -> List[float]:
        """Generate a random valid sequence with length between 100 and 500."""
        length = random.randint(100, 500)
        sequence = [random.uniform(0, 1000) for _ in range(length)]

        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01

        return sequence

    @staticmethod
    def generate_structured_sequence() -> List[float]:
        """Generate a structured sequence using sinusoidal pattern for better initialization."""
        length = random.randint(100, 500)
        # Create a sequence with a sinusoidal pattern to encourage balanced convolution
        sequence = [abs(np.sin(i * np.pi / length)) * 1000 for i in range(length)]
        # Add some random variation for exploration while maintaining structure
        for i in range(length):
            if random.random() < 0.1:
                sequence[i] += random.uniform(-200, 200)
        # Ensure non-negative values
        sequence = [max(0, x) for x in sequence]
        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01
        return sequence

    @staticmethod
    def generate_exponential_sequence() -> List[float]:
        """Generate an exponentially decaying sequence to balance mass and convolution."""
        length = random.randint(100, 500)
        # Create exponential decay pattern
        sequence = [1000 * np.exp(-i / (length * 0.5)) for i in range(length)]
        # Add some randomness
        for i in range(length):
            if random.random() < 0.1:
                sequence[i] += random.uniform(-100, 100)
        # Ensure non-negative values
        sequence = [max(0, x) for x in sequence]
        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01
        return sequence

    @staticmethod
    def mutate_sequence(sequence: List[float], generation: int = 0,
                       mu: float = 0, sigma: float = 100,
                       indpb: float = 0.1) -> List[float]:
        """Mutate a sequence with adaptive Gaussian noise."""
        mutated = copy.deepcopy(sequence)
        # Reduce mutation rate as generations progress
        adaptive_indpb = indpb * (1.0 - generation / 100.0)
        for i in range(len(mutated)):
            if random.random() < adaptive_indpb:
                # Use larger mutations early, smaller later
                adaptive_sigma = sigma * (1.0 - generation / 100.0) + 1.0
                mutated[i] += random.gauss(mu, adaptive_sigma)
                mutated[i] = max(0, mutated[i])  # Ensure non-negative
        return mutated

    @staticmethod
    def crossover_sequences(seq1: List[float], seq2: List[float],
                          cxpb: float = 0.5) -> Tuple[List[float], List[float]]:
        """Perform uniform crossover between two sequences."""
        child1, child2 = copy.deepcopy(seq1), copy.deepcopy(seq2)
        for i in range(len(child1)):
            if random.random() < cxpb:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def optimize(self) -> List[float]:
        """Run the evolutionary optimization process."""
        start_time = time.time()
        timeout = 170  # Leave 10 seconds for cleanup

        # Initialize population with multiple strategies
        population = []
        for i in range(self.pop_size):
            if i % 3 == 0:
                population.append(self.generate_structured_sequence())
            elif i % 3 == 1:
                population.append(self.generate_exponential_sequence())
            else:
                population.append(self.generate_random_valid_sequence())

        # Evaluate initial population
        fitnesses = self.evaluate_population_parallel(population)

        # Track best individual
        best_idx = np.argmax(fitnesses)
        self.best_score = fitnesses[best_idx]
        self.best_sequence = copy.deepcopy(population[best_idx])

        print(f"Initial best score: {self.best_score:.6f}")

        # Track fitness history for termination criteria
        fitness_history = [self.best_score]

        # Begin evolution
        for gen in range(self.generations):
            if time.time() - start_time > timeout:
                break

            # Adaptive tournament selection based on diversity
            selected = self.adaptive_tournament_selection(population, fitnesses, gen)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover_sequences(selected[i], selected[i+1])
                    child1 = self.mutate_sequence(child1, gen)
                    child2 = self.mutate_sequence(child2, gen)
                    offspring.extend([child1, child2])
                else:
                    # Handle odd population size
                    offspring.append(self.mutate_sequence(selected[i], gen))

            # Keep offspring size consistent
            offspring = offspring[:self.pop_size]

            # Apply local refinement to top individuals
            top_indices = np.argsort(fitnesses)[-min(5, len(offspring)):][::-1]
            for i in range(len(top_indices)):
                idx = top_indices[i]
                # Apply local refinement with gradient-based method
                offspring[idx] = self.local_refinement(offspring[idx])

            # Evaluate offspring
            offspring_fitnesses = self.evaluate_population_parallel(offspring)

            # Replace population with offspring
            population = offspring
            fitnesses = offspring_fitnesses

            # Update best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_sequence = copy.deepcopy(population[best_idx])

            # Update fitness history
            fitness_history.append(self.best_score)
            if len(fitness_history) > 10:
                fitness_history.pop(0)

            print(f"Gen {gen}: Best = {self.best_score:.6f}")

            # Adaptive termination check
            if gen > 10:
                # Check for stagnation in fitness improvement
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < 1e-8:
                    print("Fitness plateau detected, stopping early")
                    break

                # Check for diversity threshold
                if np.std(fitnesses) < 1e-6:
                    print("Population converged, stopping early")
                    break

        return self.best_sequence

    @staticmethod
    def adaptive_tournament_selection(population: List[List[float]],
                                    fitnesses: List[float],
                                    generation: int,
                                    base_tournament_size: int = 3) -> List[List[float]]:
        """Select individuals using adaptive tournament selection based on diversity."""
        selected = []
        # Adjust tournament size based on generation
        diversity = np.std(fitnesses) if len(fitnesses) > 1 else 0.0
        tournament_size = max(2, min(base_tournament_size + int(diversity * 10), 10))

        for _ in range(len(population)):
            # Select randomly from population
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(copy.deepcopy(population[winner_index]))
        return selected

    @staticmethod
    def local_refinement(sequence: List[float], steps: int = 20) -> List[float]:
        """Apply enhanced local refinement with momentum and adaptive learning rates."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)

        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq

        # Initialize momentum terms
        velocity = np.zeros_like(seq)
        alpha = 1e-4  # Initial learning rate

        for step in range(steps):
            # Compute convolution
            conv = fftconvolve(seq, seq, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = np.max(conv)

            # Compute gradient estimate using finite differences
            grad = np.zeros_like(seq)
            epsilon = 1e-6
            for i in range(n):
                # Perturb forward
                seq_forward = seq.copy()
                seq_forward[i] += epsilon
                seq_forward /= np.sum(seq_forward)

                # Perturb backward
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

            # Apply gradient clipping to prevent extreme updates
            grad = np.clip(grad, -100.0, 100.0)

            # Adaptive learning rate with decay
            adaptive_lr = alpha * (0.99 ** step)

            # Update momentum term
            velocity = 0.9 * velocity + adaptive_lr * grad

            # Update sequence using momentum
            seq += velocity
            seq = np.maximum(seq, 0)  # Ensure non-negative

            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq

        return seq.tolist()

    @staticmethod
    def tournament_selection(population: List[List[float]],
                           fitnesses: List[float],
                           tournament_size: int = 3) -> List[List[float]]:
        """Select individuals using tournament selection."""
        selected = []
        for _ in range(len(population)):
            # Select randomly from population
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(copy.deepcopy(population[winner_index]))
        return selected

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer(pop_size=50, generations=50)
    try:
        best_sequence = optimizer.optimize()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to a basic sequence if nothing worked
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")