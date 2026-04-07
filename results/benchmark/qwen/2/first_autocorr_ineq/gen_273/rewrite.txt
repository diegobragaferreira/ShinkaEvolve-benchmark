# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
import copy
from typing import List, Tuple

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class GradientFreeEvolutionaryOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.population_size = 50
        self.num_generations = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def convolve_fft(self, a, b):
        """Compute convolution using FFT for better performance."""
        n = len(a)
        # Zero-pad to avoid circular convolution effects
        padded_length = 2 * n - 1
        fa = fft(a, padded_length)
        fb = fft(b, padded_length)
        result = ifft(fa * fb).real
        return result[:n]
    
    def compute_c1(self, sequence):
        """Compute the C1 constant from the sequence."""
        if len(sequence) == 0:
            return float('inf')
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')

        convolved = self.convolve_fft(sequence, sequence)
        max_conv = np.max(convolved)
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1
    
    def compute_inv_c1(self, sequence):
        """Compute inverse of C1 (the value we want to maximize)."""
        c1 = self.compute_c1(sequence)
        return 1.0 / c1 if c1 > 0 else 0.0

    def create_individual(self, length=None):
        """Create a new individual with specified length or random length."""
        if length is None:
            length = random.randint(50, 1000)
        individual = [random.uniform(0.01, 100.0) for _ in range(length)]
        return individual

    def mutate(self, individual):
        """Mutate an individual with probability."""
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] *= random.uniform(0.5, 2.0)  # Scale factor
                mutated[i] = max(0.01, mutated[i])  # Ensure non-negative
        return mutated

    def crossover(self, parent1, parent2):
        """Perform crossover between two individuals."""
        if len(parent1) != len(parent2):
            # If lengths differ, truncate to shorter
            min_len = min(len(parent1), len(parent2))
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]
            
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1[:], parent2[:]

    def selection(self, population, fitness_scores):
        """Select individuals for reproduction."""
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_index][:])
        return selected

    def adaptive_length_adjustment(self, population):
        """Adjust lengths of individuals to balance exploration and exploitation."""
        new_population = []
        for ind in population:
            if random.random() < 0.3:  # 30% chance to adjust length
                new_length = max(50, min(1000, len(ind) + random.randint(-20, 20)))
                if new_length < len(ind):
                    ind = ind[:new_length]
                else:
                    ind.extend([random.uniform(0.01, 100.0)] * (new_length - len(ind)))
            new_population.append(ind)
        return new_population

    def local_refinement(self, individual):
        """Refine an individual using gradient-free methods."""
        # Simple hill climbing with local perturbations
        current_fitness = self.compute_inv_c1(individual)
        best_individual = individual[:]
        best_fitness = current_fitness
        
        for _ in range(10):
            # Create neighbor by small random changes
            neighbor = individual[:]
            for i in range(len(neighbor)):
                if random.random() < 0.1:
                    neighbor[i] *= random.uniform(0.9, 1.1)
                    neighbor[i] = max(0.01, neighbor[i])
            
            neighbor_fitness = self.compute_inv_c1(neighbor)
            if neighbor_fitness > best_fitness:
                best_fitness = neighbor_fitness
                best_individual = neighbor[:]
        
        return best_individual

    def run_evolution(self, initial_population=None):
        """Run the evolutionary optimization process."""
        if initial_population is None:
            population = [self.create_individual() for _ in range(self.population_size)]
        else:
            population = initial_population[:]
        
        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness_scores = [self.compute_inv_c1(ind) for ind in population]
            
            # Track best individual in this generation
            best_gen_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[best_gen_idx] > self.best_inv_c1:
                self.best_inv_c1 = fitness_scores[best_gen_idx]
                self.best_sequence = population[best_gen_idx][:]
            
            # Selection
            selected_population = self.selection(population, fitness_scores)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            new_population = new_population[:self.population_size]
            
            # Length adjustment
            population = self.adaptive_length_adjustment(new_population)
            
            # Local refinement on some individuals
            for i in range(0, len(population), 3):
                if i < len(population):
                    population[i] = self.local_refinement(population[i])
        
        return self.best_sequence

    def optimize_sequence(self):
        """Main optimization routine with multiple restarts."""
        start_time = time.time()
        
        # Multiple restarts with different strategies
        for restart in range(5):
            if time.time() - start_time > self.timeout_seconds - 10:
                break
                
            # Strategy 1: Random initialization
            population = [self.create_individual() for _ in range(self.population_size)]
            self.run_evolution(population)
            
            # Strategy 2: Initialize with some structure
            population_2 = []
            for _ in range(self.population_size):
                length = random.randint(50, 1000)
                # Exponential decay pattern that often works well
                pattern = [1.0 * (0.95 ** i) for i in range(length)]
                pattern = [max(0.01, x) for x in pattern]
                # Add some noise
                noise = [random.uniform(-0.1, 0.1) for _ in range(length)]
                pattern = [max(0.01, p + n) for p, n in zip(pattern, noise)]
                population_2.append(pattern)
                
            self.run_evolution(population_2)
            
        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = GradientFreeEvolutionaryOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")