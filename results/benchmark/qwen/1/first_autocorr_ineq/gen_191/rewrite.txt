# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal, optimize
import random
import time
import copy
from joblib import Parallel, delayed
import warnings
from typing import List, Tuple, Optional

class AutocorrelationEvaluator:
    """Handles computation of C₁ and related metrics for sequences."""
    
    @staticmethod
    def compute_c1(sequence: List[float]) -> float:
        """Compute C₁ for a given sequence using FFT for efficiency."""
        if len(sequence) == 0:
            return float('inf')

        # Convert to numpy array
        a = np.array(sequence)

        # Compute autoconvolution using FFT for efficiency
        conv = signal.fftconvolve(a, a, mode='full')
        conv = conv[len(a)-1:]  # Take the relevant part

        # Max convolution value
        max_conv = np.max(conv)

        # Sum of sequence squared
        sum_sq = np.sum(a)**2

        if sum_sq == 0:
            return float('inf')

        # Compute C₁
        c1 = 2 * len(a) * max_conv / sum_sq

        return c1

    @staticmethod
    def compute_inv_c1(sequence: List[float]) -> float:
        """Compute 1/C₁ for a given sequence."""
        c1 = AutocorrelationEvaluator.compute_c1(sequence)
        if c1 == 0 or np.isnan(c1):
            return 0
        return 1.0 / c1

class SequenceInitializer:
    """Generates initial sequences for the optimization process."""
    
    @staticmethod
    def generate_structured_sequence(min_length=10, max_length=1000, max_height=1000) -> List[float]:
        """Generate a structured sequence using Gaussian distribution for better initialization."""
        n = random.randint(min_length, max_length)
        # Generate heights from a truncated normal distribution
        sequence = np.random.normal(loc=max_height/2, scale=max_height/6, size=n)
        # Clip to [0, max_height] and ensure at least one element is non-zero
        sequence = np.clip(sequence, 0, max_height)
        if np.sum(sequence) < 0.01:
            sequence[random.randint(0, n-1)] = random.uniform(0.1, max_height)
        return sequence.tolist()

class GeneticOperators:
    """Manages genetic operators like crossover, mutation, and selection."""
    
    @staticmethod
    def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
        """Perform crossover between two parents."""
        # Uniform crossover
        child = []
        min_len = min(len(parent1), len(parent2))
        for i in range(min_len):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

        # Add remaining elements from longer parent
        if len(parent1) > len(parent2):
            child.extend(parent1[min_len:])
        elif len(parent2) > len(parent1):
            child.extend(parent2[min_len:])

        return child

    @staticmethod
    def mutate(sequence: List[float], mutation_rate: float, max_height=1000, 
               generation=0, max_generations=200) -> List[float]:
        """Mutate a sequence with adaptive strategy."""
        mutated = copy.deepcopy(sequence)
        mutation_type_prob = min(0.7, generation / max_generations * 0.7)  # Increase Gaussian mutation probability over time

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Adaptive mutation type selection
                if random.random() < mutation_type_prob:
                    # Gaussian mutation for fine-tuning
                    mutated[i] = max(0, mutated[i] + np.random.normal(scale=max_height/10))
                else:
                    # Uniform mutation for broader exploration
                    mutated[i] = random.uniform(0, max_height)

                mutated[i] = min(max_height, mutated[i])
        return mutated

    @staticmethod
    def tournament_selection(population: List[List[float]], fitnesses: List[float], 
                             generation: int, population_size: int) -> List[float]:
        """Select an individual using adaptive tournament selection."""
        # Calculate population diversity
        if len(fitnesses) < 2:
            tournament_size = 3
        else:
            fitness_std = np.std(fitnesses)
            fitness_mean = np.mean(fitnesses)
            diversity = fitness_std / (fitness_mean + 1e-8)  # Avoid division by zero

            # Dynamic tournament size based on diversity and generation
            if generation <= 20 or diversity > 0.1:
                # High diversity or early generations: larger tournament
                tournament_size = min(9, max(3, int(5 + diversity * 10)))
            elif generation >= 50 or diversity < 0.05:
                # Low diversity or late generations: smaller tournament
                tournament_size = max(3, int(3 + diversity * 5))
            else:
                tournament_size = 5

        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return copy.deepcopy(population[winner_index])

class LocalSearchOptimizer:
    """Provides local search enhancements to improve individual sequences."""
    
    @staticmethod
    def solve_convolution_lp(f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
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

        # Non-negativity constraints: b_i >= 0
        a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
        b_ub_nonneg = np.zeros(n)  # Zero vector

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            print('LP optimization failed.')
            return None

    @staticmethod
    def get_good_direction_to_move_into(sequence: List[float], max_iterations=50) -> Optional[List[float]]:
        """Returns the direction to move into the sequence using multi-start local search."""
        n = len(sequence)
        if n == 0:
            return None

        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        # Normalize sequence
        normalized_sequence = np.array(sequence) / sum_sequence

        # Multi-start local search
        best_sequence = normalized_sequence.copy()
        best_score = AutocorrelationEvaluator.compute_inv_c1(normalized_sequence * sum_sequence)

        # Try multiple starting points around the current sequence
        for _ in range(5):
            # Perturb slightly
            perturbed = normalized_sequence + np.random.normal(0, 0.01, n)
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negativity
            perturbed = perturbed / np.sum(perturbed) * sum_sequence  # Renormalize

            # Simple gradient ascent-like step
            step_size = 0.001
            for _ in range(max_iterations):
                current_score = AutocorrelationEvaluator.compute_inv_c1(perturbed)

                # Simple finite difference gradient estimation
                grad = np.zeros(n)
                eps = 1e-6
                for i in range(n):
                    pert_plus = perturbed.copy()
                    pert_plus[i] += eps
                    score_plus = AutocorrelationEvaluator.compute_inv_c1(pert_plus)

                    grad[i] = (score_plus - current_score) / eps

                # Update with gradient ascent
                new_perturbed = perturbed + step_size * grad
                new_perturbed = np.maximum(new_perturbed, 0)  # Ensure non-negativity

                # Renormalize
                new_perturbed = new_perturbed / np.sum(new_perturbed) * sum_sequence

                perturbed = new_perturbed

                # Check convergence
                if abs(AutocorrelationEvaluator.compute_inv_c1(perturbed) - current_score) < 1e-8:
                    break

            # Check if this local search improved the solution
            final_score = AutocorrelationEvaluator.compute_inv_c1(perturbed)
            if final_score > best_score:
                best_score = final_score
                best_sequence = perturbed.copy()

        # Return the best sequence in original scale
        return (best_sequence / sum_sequence * sum_sequence).tolist()

class HybridAutocorrelationOptimizer:
    """Main optimizer that orchestrates the evolutionary and local search processes."""
    
    def __init__(self, pop_size: int = 100, generations: int = 200, 
                 initial_mutation_rate: float = 0.1, elite_size: int = 10):
        self.pop_size = pop_size
        self.generations = generations
        self.initial_mutation_rate = initial_mutation_rate
        self.elite_size = elite_size
        self.best_score = 0.0
        self.best_sequence = None

    def evaluate_fitness_parallel(self, individuals: List[List[float]]) -> List[float]:
        """Evaluate fitness for a batch of individuals in parallel."""
        def evaluate_single(individual):
            inv_c1 = AutocorrelationEvaluator.compute_inv_c1(individual)
            return inv_c1 if np.sum(individual) > 0.01 else 0

        results = Parallel(n_jobs=-1)(delayed(evaluate_single)(ind) for ind in individuals)
        return results

    def optimize(self) -> List[float]:
        """Run the hybrid evolutionary-local optimization process."""
        # Initialize population with structured sequences
        population = [SequenceInitializer.generate_structured_sequence()
                      for _ in range(self.pop_size)]

        best_score = 0
        best_individual = None
        stagnation_counter = 0
        max_stagnation = 30
        start_time = time.time()

        for generation in range(self.generations):
            # Calculate adaptive mutation rate (decreases over generations)
            mutation_rate = self.initial_mutation_rate * (1 - generation / self.generations)
            if mutation_rate < 0.01:
                mutation_rate = 0.01

            # Evaluate fitness for all individuals in parallel
            fitnesses = self.evaluate_fitness_parallel(population)

            # Track best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_score:
                best_score = fitnesses[best_idx]
                best_individual = copy.deepcopy(population[best_idx])
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adjust max_stagnation dynamically based on generation
            adjusted_max_stagnation = max_stagnation + generation // 20

            # Check for stagnation
            if stagnation_counter >= adjusted_max_stagnation:
                break

            # Create new population
            new_population = []

            # Elitism: keep the best individuals
            sorted_indices = np.argsort(fitnesses)[::-1][:self.elite_size]
            for idx in sorted_indices:
                new_population.append(copy.deepcopy(population[idx]))

            # Generate offspring
            while len(new_population) < self.pop_size:
                # Tournament selection with adaptation
                parent1 = GeneticOperators.tournament_selection(population, fitnesses, generation, self.pop_size)
                parent2 = GeneticOperators.tournament_selection(population, fitnesses, generation, self.pop_size)

                # Crossover
                child = GeneticOperators.crossover(parent1, parent2)

                # Mutation with adaptation
                child = GeneticOperators.mutate(child, mutation_rate, generation=generation, max_generations=self.generations)

                # Apply improved local search
                refined_child = LocalSearchOptimizer.get_good_direction_to_move_into(child)
                if refined_child is not None:
                    child = refined_child

                new_population.append(child)

            population = new_population[:self.pop_size]

            # Check time limit
            if time.time() - start_time > 170:  # Leave some buffer
                break

        self.best_score = best_score
        self.best_sequence = best_individual
        return best_individual

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run adaptive evolutionary optimization
    optimizer = HybridAutocorrelationOptimizer(
        pop_size=100,
        generations=200,
        initial_mutation_rate=0.1,
        elite_size=10
    )
    best_sequence = optimizer.optimize()

    # Ensure we have a valid sequence
    if best_sequence is None:
        best_sequence = SequenceInitializer.generate_structured_sequence()

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")