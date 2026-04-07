# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import time
import copy
from joblib import Parallel, delayed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class SequenceOptimizer:
    def __init__(self, max_generations=200, population_size=100, 
                 initial_mutation_rate=0.1, elite_size=10):
        self.max_generations = max_generations
        self.population_size = population_size
        self.initial_mutation_rate = initial_mutation_rate
        self.elite_size = elite_size
        self.best_score = 0
        self.best_individual = None
        self.start_time = None
        
    def compute_c1(self, sequence):
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
    
    def compute_inv_c1(self, sequence):
        """Compute 1/C₁ for a given sequence."""
        c1 = self.compute_c1(sequence)
        if c1 == 0 or np.isnan(c1):
            return 0
        return 1.0 / c1
    
    def generate_structured_sequence(self, min_length=10, max_length=1000, max_height=1000):
        """Generate a structured sequence using Gaussian distribution for better initialization."""
        n = random.randint(min_length, max_length)
        # Generate heights from a truncated normal distribution
        sequence = np.random.normal(loc=max_height/2, scale=max_height/6, size=n)
        # Clip to [0, max_height] and ensure at least one element is non-zero
        sequence = np.clip(sequence, 0, max_height)
        if np.sum(sequence) < 0.01:
            sequence[random.randint(0, n-1)] = random.uniform(0.1, max_height)
        return sequence.tolist()
    
    def crossover(self, parent1, parent2):
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
    
    def mutate(self, sequence, mutation_rate, max_height=1000):
        """Mutate a sequence."""
        mutated = copy.deepcopy(sequence)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Use Gaussian mutation for better exploration
                mutated[i] = max(0, mutated[i] + np.random.normal(scale=max_height/10))
                mutated[i] = min(max_height, mutated[i])
        return mutated
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return copy.deepcopy(population[winner_index])
    
    def evaluate_fitness_parallel(self, individuals):
        """Evaluate fitness for a batch of individuals in parallel."""
        def evaluate_single(individual):
            inv_c1 = self.compute_inv_c1(individual)
            return inv_c1 if np.sum(individual) > 0.01 else 0
        
        results = Parallel(n_jobs=-1)(delayed(evaluate_single)(ind) for ind in individuals)
        return results
    
    def adapt_parameters(self, generation):
        """Adaptively adjust evolutionary parameters based on generation."""
        mutation_rate = self.initial_mutation_rate * (1 - generation / self.max_generations)
        # Ensure minimum mutation rate
        mutation_rate = max(mutation_rate, 0.01)
        
        # Dynamic stagnation threshold
        max_stagnation = 30 + generation // 20
        
        return mutation_rate, max_stagnation
    
    def run_evolution(self):
        """Run the evolutionary optimization process."""
        # Initialize population with structured sequences
        population = [self.generate_structured_sequence()
                      for _ in range(self.population_size)]

        stagnation_counter = 0
        self.start_time = time.time()
        
        for generation in range(self.max_generations):
            # Adapt parameters
            mutation_rate, max_stagnation = self.adapt_parameters(generation)
            
            # Evaluate fitness for all individuals in parallel
            fitnesses = self.evaluate_fitness_parallel(population)

            # Track best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_individual = copy.deepcopy(population[best_idx])
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check for stagnation
            if stagnation_counter >= max_stagnation:
                break

            # Create new population
            new_population = []

            # Elitism: keep the best individuals
            sorted_indices = np.argsort(fitnesses)[::-1][:self.elite_size]
            for idx in sorted_indices:
                new_population.append(copy.deepcopy(population[idx]))

            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child, mutation_rate)

                new_population.append(child)

            population = new_population[:self.population_size]

            # Check time limit
            if time.time() - self.start_time > 170:  # Leave some buffer
                break

        return self.best_individual, self.best_score

def search_for_best_sequence():
    """Function to search for the best coefficient sequence using improved evolutionary approach."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize optimizer
    optimizer = SequenceOptimizer(
        max_generations=200,
        population_size=100,
        initial_mutation_rate=0.1,
        elite_size=10
    )
    
    # Run optimization
    best_sequence, best_score = optimizer.run_evolution()

    # Ensure we have a valid sequence
    if best_sequence is None:
        best_sequence = optimizer.generate_structured_sequence()

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")