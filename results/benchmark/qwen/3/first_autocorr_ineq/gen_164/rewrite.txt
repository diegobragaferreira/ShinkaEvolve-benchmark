# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import random
import time
from collections import defaultdict
from joblib import Parallel, delayed
import math

class AutocorrelationOptimizer:
    def __init__(self):
        self.cache = {}
        self.best_sequence = None
        self.best_c1 = float('inf')
        
    def compute_autocorrelation_constant(self, sequence):
        """
        Compute C₁ for a given sequence using FFT for efficiency.
        C₁ = 2n * max(convolution) / (sum(sequence))^2
        """
        n = len(sequence)
        if n == 0:
            return float('inf')

        # Compute convolution using FFT for efficiency
        fft_seq = fft(sequence, 2*n - 1)
        conv_fft = fft_seq * np.conj(fft_seq)
        conv = ifft(conv_fft).real[:2*n-1]
        max_conv = np.max(conv)

        sum_seq = np.sum(sequence)
        if sum_seq < 0.01:
            return float('inf')

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        return c1

    def evaluate_objective(self, sequence):
        """
        Evaluate the objective function: -1/C₁ (we minimize this to maximize 1/C₁)
        """
        c1 = self.compute_autocorrelation_constant(sequence)
        if c1 == float('inf'):
            return float('inf')  # Invalid solution
        return -1.0 / c1  # Negative because we want to maximize 1/C₁

    def evaluate_sequence_with_cache(self, sequence):
        """
        Evaluate sequence with caching to avoid redundant computations.
        """
        key = tuple(sequence)
        if key in self.cache:
            return self.cache[key]

        result = self.evaluate_objective(sequence)
        self.cache[key] = result
        return result

    def generate_initial_sequence(self):
        """
        Generate a good initial random sequence with more structure.
        """
        n = random.randint(100, 1000)
        if random.random() < 0.3:
            # Power law distribution - heavy tail
            sequence = [random.expovariate(0.1) for _ in range(n)]
            # Normalize to prevent extreme values
            max_val = max(sequence)
            sequence = [x * 100.0 / max_val if max_val > 0 else 1.0 for x in sequence]
        elif random.random() < 0.6:
            # Uniform distribution with some peaks
            sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
            # Add a few peaks
            for i in range(min(10, len(sequence)//20)):
                peak_pos = random.randint(0, len(sequence)-1)
                sequence[peak_pos] = random.uniform(100.0, 1000.0)
        else:
            # Mixed distribution
            sequence = []
            for i in range(n):
                if random.random() < 0.7:
                    sequence.append(random.uniform(0.1, 10.0))
                else:
                    sequence.append(random.uniform(50.0, 100.0))

        return sequence

    def generate_population(self, size, min_size=100, max_size=1000):
        """Generate a population of sequences."""
        return [self.generate_initial_sequence() for _ in range(size)]

    def quadratic_optimization_step(self, current_seq, max_iter=100):
        """
        Perform a quadratic optimization step to improve the sequence.
        """
        n = len(current_seq)
        # Define bounds: all elements must be in [0, 1000]
        bounds = [(0.0, 1000.0) for _ in range(n)]

        # Define constraints
        def sum_constraint(x):
            return np.sum(x) - 0.01  # Require sum >= 0.01

        constraints = [{'type': 'ineq', 'fun': sum_constraint}]

        # Objective function to minimize
        def objective(x):
            return self.evaluate_objective(x)

        # Try multiple optimization methods
        methods_to_try = ['SLSQP', 'L-BFGS-B']

        for method in methods_to_try:
            try:
                # Use smaller tolerance for faster convergence
                result = minimize(objective, current_seq, method=method, bounds=bounds,
                                constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
                if result.success:
                    return result.x.tolist()
            except:
                continue

        # If all methods fail, return the original sequence slightly perturbed
        perturbed = [max(0.0, x + random.gauss(0, 0.05)) for x in current_seq]
        if np.sum(perturbed) < 0.01:
            perturbed[0] = max(0.0, perturbed[0] + 0.01)
        return perturbed

    def mutate_sequence(self, sequence, mutation_rate=0.1):
        """Mutate a sequence by randomly changing some elements."""
        mutated = sequence.copy()
        # Adaptive mutation rate: lower for longer sequences
        if len(sequence) > 500:
            mutation_rate *= 0.5
        elif len(sequence) < 200:
            mutation_rate *= 1.5

        # Calculate standard deviation for mutation scaling
        std_dev = np.std(sequence) if len(sequence) > 0 else 1.0
        mutation_scale = max(0.1, std_dev * 0.1)  # Scale mutation by sequence variability

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate with Gaussian noise scaled by sequence variability
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, mutation_scale))
        return mutated

    def crossover_sequences(self, parent1, parent2):
        """Perform crossover between two sequences with adaptive mixing."""
        # Use a blend crossover that considers characteristics of parents
        n1, n2 = len(parent1), len(parent2)
        min_len = min(n1, n2)

        # Create offspring with blended elements
        offspring = []

        # Determine if we're doing crossover or just taking one parent
        if random.random() < 0.7:  # 70% chance of crossover
            # Blend elements with weight based on parent characteristics
            for i in range(max(n1, n2)):
                if i < min_len:
                    # Blend based on similarity of elements
                    if random.random() < 0.5:
                        offspring.append(parent1[i])
                    else:
                        offspring.append(parent2[i])
                elif i < n1:
                    # Extending beyond shorter parent
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
        else:
            # Just take one parent with some variation
            parent = parent1 if random.random() < 0.5 else parent2
            offspring = parent.copy()

        # Ensure all elements are non-negative
        offspring = [max(0.0, x) for x in offspring]
        return offspring

    def evaluate_population(self, population, n_jobs=-1):
        """Evaluate a population in parallel."""
        return Parallel(n_jobs=n_jobs)(delayed(self.evaluate_sequence_with_cache)(seq) for seq in population)

    def search_for_best_sequence(self):
        """
        Main function to search for the best coefficient sequence using an enhanced evolutionary approach.
        """
        start_time = time.time()
        population_size = 30
        generations = 60
        keep_top = 8
        elite_preservation = 2

        # Generate initial population
        population = self.generate_population(population_size)

        # Evaluate initial population
        fitness_scores = list(zip(population, self.evaluate_population(population)))

        # Sort population by fitness (lower is better)
        fitness_scores.sort(key=lambda x: x[1])

        # Track best solution globally
        self.best_sequence = fitness_scores[0][0]
        self.best_c1 = fitness_scores[0][1]

        # Main evolution loop
        for gen in range(generations):
            if time.time() - start_time > 170:  # Leave 10 seconds for finalization
                break
                
            # Keep top performers (elite)
            top_performers = [seq for seq, _ in fitness_scores[:keep_top]]

            # Create new population
            new_population = top_performers[:]

            # Preserve elites
            if elite_preservation > 0:
                elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[:elite_preservation]
                elites = [fitness_scores[i][0] for i in elite_indices]
                new_population.extend(elites)

            # Add mutated versions of top performers
            for i in range(population_size - len(new_population)):
                if random.random() < 0.7:  # 70% chance of mutation
                    parent = random.choice(top_performers)
                    child = self.mutate_sequence(parent)
                else:  # 30% chance of crossover
                    p1, p2 = random.sample(top_performers, 2)
                    child = self.crossover_sequences(p1, p2)

                new_population.append(child)

            # Apply local optimization to some individuals
            for i in range(0, len(new_population), 2):
                if random.random() < 0.6:  # 60% chance of local optimization
                    new_population[i] = self.quadratic_optimization_step(new_population[i])

            # Evaluate new population
            fitness_scores = list(zip(new_population, self.evaluate_population(new_population)))

            # Sort population by fitness
            fitness_scores.sort(key=lambda x: x[1])

            # Update global best
            if fitness_scores[0][1] < self.best_c1:
                self.best_sequence = fitness_scores[0][0]
                self.best_c1 = fitness_scores[0][1]

        # Final optimization of the best sequence
        final_best = self.quadratic_optimization_step(self.best_sequence, max_iter=200)

        # Return the best sequence found
        return final_best

def search_for_best_sequence():
    """Wrapper function to maintain interface compatibility."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")