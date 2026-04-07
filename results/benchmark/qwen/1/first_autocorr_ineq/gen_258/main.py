# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial
import copy
from typing import List, Tuple, Optional
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class AdaptiveGradientEvolutionaryOptimizer:
    """
    An adaptive hybrid optimizer combining evolutionary and gradient-based methods to maximize 1/C₁.
    Draws upon best practices from DEAP-based evolutionary strategies, gradient refinement,
    and adaptive mechanisms for improved performance.
    """

    def __init__(self, pop_size: int = 50, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.best_score = 0.0
        self.best_sequence = None
        self.fitness_cache = {}  # Memoization for fitness evaluations
        self.top_patterns_memory = []  # Memory for learned top patterns
        self.generation_counter = 0  # Track generations for progressive init

    @staticmethod
    def evaluate_sequence(sequence: List[float]) -> Tuple[float, float, float, float]:
        """
        Evaluate a sequence and return its performance metrics.

        Args:
            sequence: List of non-negative real numbers representing step heights

        Returns:
            tuple: (C₁, 1/C₁, max_convolution_value, sum_of_sequence)
        """
        # Check cache first
        seq_tuple = tuple(sequence)
        if seq_tuple in AdaptiveGradientEvolutionaryOptimizer.fitness_cache:
            return AdaptiveGradientEvolutionaryOptimizer.fitness_cache[seq_tuple]

        try:
            # Convert to numpy array
            a = np.array(sequence)
            sum_a = np.sum(a)

            # Avoid division by zero or negligible sums
            if sum_a < 1e-10:
                result = (float('inf'), 0.0, 0.0, sum_a)
                AdaptiveGradientEvolutionaryOptimizer.fitness_cache[seq_tuple] = result
                return result

            # Compute autoconvolution using FFT for efficiency
            b = fftconvolve(a, a, mode='full')
            b = b[len(a)-1:2*len(a)-1]  # Convolution part

            max_b = np.max(b)

            # Compute C₁ = 2n * max(b) / (sum(a))^2
            n = len(a)
            c1 = 2 * n * max_b / (sum_a ** 2)

            # Return inverse for maximization
            inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

            result = (c1, inv_c1, max_b, sum_a)
            AdaptiveGradientEvolutionaryOptimizer.fitness_cache[seq_tuple] = result
            return result
        except Exception as e:
            result = (float('inf'), 0.0, 0.0, 0.0)
            AdaptiveGradientEvolutionaryOptimizer.fitness_cache[seq_tuple] = result
            return result

    def generate_multiscale_sequence(self) -> List[float]:
        """Generate a structured sequence using a multi-scale pattern for better initialization."""
        # Progressive initialization strategy
        if self.generation_counter < 10:
            # Early generations: pure random initialization
            length = random.randint(100, 500)
            sequence = [random.uniform(0, 1000) for _ in range(length)]
        elif self.generation_counter < 20 and self.top_patterns_memory:
            # Mid generations: use learned patterns with some randomization
            # Choose pattern from memory or random
            if random.random() < 0.7:
                # Use learned pattern
                pattern = random.choice(self.top_patterns_memory)
                sequence = [max(0, x + random.uniform(-100, 100)) for x in pattern]
                length = len(pattern)
            else:
                # Random pattern
                length = random.randint(100, 500)
                sequence = [random.uniform(0, 1000) for _ in range(length)]
        else:
            # Later generations: balanced approach
            method = random.choice(['sinusoidal', 'exponential', 'power', 'mixed'])
            length = random.randint(100, 500)

            if method == 'sinusoidal':
                sequence = [abs(np.sin(i * np.pi / length)) * 1000 for i in range(length)]
            elif method == 'exponential':
                sequence = [np.exp(-i / (length / 3)) * 1000 for i in range(length)]
            elif method == 'power':
                sequence = [(i + 1) ** (-1.5) * 1000 for i in range(length)]
            else:  # mixed
                # Mix of patterns to increase diversity
                sequence = []
                for i in range(length):
                    if random.random() < 0.3:
                        sequence.append(abs(np.sin(i * np.pi / length)) * 1000)
                    elif random.random() < 0.6:
                        sequence.append(np.exp(-i / (length / 3)) * 1000)
                    else:
                        sequence.append((i + 1) ** (-1.5) * 1000)

        # Add some randomness to avoid perfect symmetry
        for i in range(length):
            if random.random() < 0.1:
                sequence[i] += random.uniform(-100, 100)

        # Ensure non-negative
        sequence = [max(0, x) for x in sequence]

        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01

        return sequence

    def store_top_patterns(self, population: List[List[float]], fitnesses: List[float]):
        """Store top performing patterns in memory for progressive initialization."""
        if len(population) < 10:
            return

        # Get top 5 sequences
        top_indices = np.argsort(fitnesses)[-5:][::-1]
        for idx in top_indices:
            if len(self.top_patterns_memory) < 10:  # Limit memory size
                self.top_patterns_memory.append(copy.deepcopy(population[idx]))
            else:
                # Replace older patterns with newer high performers
                self.top_patterns_memory[random.randint(0, len(self.top_patterns_memory)-1)] = \
                    copy.deepcopy(population[idx])

    @staticmethod
    def gradient_refine(sequence: List[float], steps: int = 50, lr: float = 1e-4) -> List[float]:
        """Apply gradient-based refinement to improve the sequence."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)
        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq

        prev_max_conv = float('inf')

        for step in range(steps):
            # Compute convolution
            conv = fftconvolve(seq, seq, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = np.max(conv)

            # Early stopping if improvement is minimal
            if abs(prev_max_conv - max_conv) < 1e-10:
                break
            prev_max_conv = max_conv

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

            # Adaptive learning rate that decreases over iterations
            adaptive_lr = lr * (1.0 - 0.9 * (step / steps))

            # Update using gradient ascent
            seq += adaptive_lr * grad
            seq = np.maximum(seq, 0)  # Ensure non-negative

            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq

        return seq.tolist()

    @staticmethod
    def mutate_sequence_gradient_ascent(sequence: List[float], steps: int = 10, lr: float = 1e-3) -> List[float]:
        """Mutate a sequence using gradient ascent to improve C₁ directly."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)

        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq

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

            # Update using gradient ascent (note: we want to maximize 1/C1, so we maximize -C1)
            # This effectively minimizes C1 which maximizes 1/C1
            seq += lr * grad
            seq = np.maximum(seq, 0)  # Ensure non-negative

            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq

        return seq.tolist()

    @staticmethod
    def mutate_sequence_cauchy(sequence: List[float], scale: float = 100.0) -> List[float]:
        """Mutate a sequence with Cauchy noise for heavy-tailed exploration."""
        mutated = copy.deepcopy(sequence)
        for i in range(len(mutated)):
            # Use Cauchy distribution instead of Gaussian for more diverse mutations
            mutated[i] += np.random.standard_cauchy() * scale
            mutated[i] = max(0, mutated[i])  # Ensure non-negative
        return mutated

    @staticmethod
    def compute_diversity(population: List[List[float]]) -> float:
        """Compute population diversity using standard deviation of fitnesses."""
        fitnesses = [AdaptiveGradientEvolutionaryOptimizer.evaluate_sequence(seq)[1] for seq in population]
        return np.std(fitnesses) if len(fitnesses) > 1 else 0.0

    def adaptive_optimize(self) -> List[float]:
        """
        Run the adaptive hybrid optimization process.
        """
        start_time = time.time()
        timeout = 170  # Leave 10 seconds for cleanup

        # Initial population with diverse structures
        population = [self.generate_multiscale_sequence() for _ in range(self.pop_size)]

        # Evaluate initial population
        fitnesses = [self.evaluate_sequence(seq)[1] for seq in population]

        # Track best individual
        best_idx = np.argmax(fitnesses)
        self.best_score = fitnesses[best_idx]
        self.best_sequence = copy.deepcopy(population[best_idx])

        print(f"Initial best score: {self.best_score:.6f}")

        # Track fitness history for convergence detection
        fitness_history = [self.best_score]
        stagnation_counter = 0
        prev_best_score = self.best_score

        # Begin evolution
        for gen in range(self.generations):
            self.generation_counter = gen  # Update generation counter

            if time.time() - start_time > timeout:
                break

            # Store top patterns for progressive initialization
            self.store_top_patterns(population, fitnesses)

            # Adaptive population management based on performance
            if gen > 10:
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < 1e-6:
                    # Reduce population size if stagnated
                    self.pop_size = max(20, int(self.pop_size * 0.9))

            # Selection using tournament selection
            selected = self.tournament_selection(population, fitnesses)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover_sequences(selected[i], selected[i+1])
                    # Apply gradient ascent mutation to the better of the two children
                    fitness_child1 = self.evaluate_sequence(child1)[1]
                    fitness_child2 = self.evaluate_sequence(child2)[1]
                    if fitness_child1 > fitness_child2:
                        child1 = self.mutate_sequence_gradient_ascent(child1)
                        offspring.append(child1)
                        offspring.append(child2)
                    else:
                        child2 = self.mutate_sequence_gradient_ascent(child2)
                        offspring.append(child1)
                        offspring.append(child2)
                else:
                    # Handle odd population size
                    offspring.append(self.mutate_sequence_cauchy(selected[i]))

            # Keep offspring size consistent
            offspring = offspring[:self.pop_size]

            # Adaptive gradient refinement based on current performance
            top_individuals = np.argsort(fitnesses)[-min(10, len(offspring)):][::-1]
            refine_count = max(1, int(0.3 * len(top_individuals)))
            for i in range(refine_count):
                idx = top_individuals[i]
                offspring[idx] = self.gradient_refine(offspring[idx])

            # Evaluate offspring
            offspring_fitnesses = [self.evaluate_sequence(seq)[1] for seq in offspring]

            # Replace population with offspring
            population = offspring
            fitnesses = offspring_fitnesses

            # Update best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_sequence = copy.deepcopy(population[best_idx])
                stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                stagnation_counter += 1  # Increment if no improvement

            # Update fitness history
            fitness_history.append(self.best_score)
            if len(fitness_history) > 10:
                fitness_history.pop(0)  # Keep only last 10 values

            print(f"Gen {gen}: Best = {self.best_score:.6f}")

            # Adaptive termination check
            if gen > 10:
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < 1e-8:
                    print("Fitness plateau detected, stopping early")
                    break

                if stagnation_counter > 5:
                    print("Population converged and stagnated, stopping early")
                    break

                # Check if we're approaching the time limit
                if time.time() - start_time > timeout - 5:
                    print("Approaching time limit, stopping early")
                    break

        return self.best_sequence

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
        """
        Execute the optimization procedure.
        """
        # Run adaptive optimization routine
        return self.adaptive_optimize()

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AdaptiveGradientEvolutionaryOptimizer(pop_size=50, generations=50)
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