# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time
from collections import deque

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with optimized numerical methods"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the autoconvolution g = f*f and its norms efficiently.
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        if not f_values or len(f_values) < 2:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4] with equal spacing
        n = len(f_values)
        
        # Step size in x domain [-1/4, 1/4]
        dx = 0.5 / (n - 1) if n > 1 else 0.5

        # Compute autoconvolution using numpy's convolution
        g = signal.convolve(f_values, f_values, mode='full')
        
        # Extract the central portion representing the actual convolution on [-1/2, 1/2]
        # For two functions of length n on [-1/4, 1/4], convolution produces 2*n-1 points
        center_start = len(g) // 2 - (n - 1)
        center_end = center_start + (2 * n - 1)
        g = g[center_start:center_end]

        # Compute the three norms
        # ||g||∞ = max of |g|
        norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0

        # ||g||₁ = sum of |g| * dx
        norm_1 = np.sum(np.abs(g)) * dx if len(g) > 1 else 0.0

        # ||g||₂² = ∫ g² dx using trapezoidal-like integration
        if len(g) <= 1:
            norm_2_squared = 0.0
        else:
            # Use piecewise linear integration for g^2
            # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
            norm_2_squared = 0.0
            for i in range(len(g)-1):
                y1, y2 = g[i], g[i+1]
                norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)

        return norm_2_squared, norm_1, norm_inf

    @classmethod
    def compute_c2(cls, f_values: List[float]) -> float:
        """Compute the C2 value for given step function."""
        norm_2_squared, norm_1, norm_inf = cls.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class PopulationManager:
    """Manages population creation, selection, and evolution"""
    
    def __init__(self, population_size: int = 50, elite_ratio: float = 1/3):
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.tournament_size = 3
        
    def generate_initial_population(self, min_length: int = 100, max_length: int = 1000) -> List[List[float]]:
        """Create diverse initial population using multiple strategies"""
        population = []
        
        # Strategy 1: Structured Gaussian-like approach
        for _ in range(self.population_size // 2):
            length = np.random.randint(min_length, max_length)
            # Create base Gaussian with decreasing amplitudes
            base_shape = np.exp(-np.linspace(-2, 2, length)**2 / 2)
            # Add controlled noise
            noise = np.random.normal(0, 0.05 * np.mean(base_shape) if np.mean(base_shape) > 0 else 0.01, length)
            individual = np.clip(base_shape + noise, 0, 10.0)
            population.append(individual.tolist())
        
        # Strategy 2: Random exponential approach with more diversity  
        for _ in range(self.population_size // 2):
            length = np.random.randint(min_length, max_length)
            individual = np.clip(np.random.exponential(scale=0.5, size=length), 0, 10.0)
            population.append(individual.tolist())
            
        return population
    
    def tournament_selection(self, fitnesses: np.ndarray) -> int:
        """Select an index using tournament selection"""
        tournament_indices = random.sample(range(len(fitnesses)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index
    
    def mutate_individual(self, individual: np.ndarray, 
                         generation: int = 0, best_fitness: float = 0.0) -> np.ndarray:
        """Apply mutation with enhanced adaptive strategy"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Dynamic mutation parameters based on generation and performance
        if best_fitness > 0.97:
            effective_mutation_rate = 0.03
            noise_sigma = 0.02
        elif best_fitness > 0.95:
            effective_mutation_rate = 0.05
            noise_sigma = 0.03
        elif best_fitness > 0.92:
            effective_mutation_rate = 0.08
            noise_sigma = 0.04
        else:
            effective_mutation_rate = 0.12
            noise_sigma = 0.05

        # Apply Gaussian perturbation to some elements
        for i in range(n):
            if random.random() < effective_mutation_rate:
                # Use mixed noise types for robust exploration  
                if random.random() < 0.7:  # 70% Gaussian noise
                    mutated[i] += np.random.normal(0, noise_sigma * np.mean(mutated) if np.mean(mutated) > 0 else 0.01)
                else:  # 30% Cauchy noise for heavy-tailed exploration
                    mutated[i] += np.random.standard_cauchy() * noise_sigma * 2
                
                # Ensure non-negativity
                mutated[i] = max(0.0, mutated[i])
                
        # Occasionally perform a local smoothing mutation
        if random.random() < 0.3 and n > 20:  # 30% chance of local smoothing
            # Adaptive window size based on sequence length
            window_size = min(5, max(1, n // 15))
            
            if window_size > 1:
                # Apply convolution smoothing
                smoothed = np.convolve(mutated, np.ones(window_size)/window_size, mode='same')
                # Mix with original using adaptive alpha based on fitness
                alpha = random.uniform(0.2, 0.7) if best_fitness > 0.95 else random.uniform(0.1, 0.5)
                mutated = alpha * mutated + (1 - alpha) * smoothed
                    
        return mutated

class StatefulOptimizer:
    """Main optimizer with state management for efficient iterative optimization"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 200):
        self.evaluator = AutoconvolutionEvaluator()
        self.population_manager = PopulationManager(population_size)
        self.max_generations = max_generations
        self.recent_improvements = deque(maxlen=10)
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.population = None
        self.fitnesses = None
        self.generation = 0
        
    def evaluate_population(self, population: List[List[float]]) -> np.ndarray:
        """Evaluate fitness for entire population in parallel"""
        def evaluate_single(individual):
            try:
                return self.evaluator.compute_c2(individual)
            except Exception:
                return 0.0
        
        # Parallel evaluation
        fitnesses = Parallel(n_jobs=-1, backend='threading')(
            delayed(evaluate_single)(ind) for ind in population
        )
        
        return np.array(fitnesses)
    
    def initialize_population(self, min_length: int = 100, max_length: int = 1000):
        """Initialize the optimization state"""
        self.population = self.population_manager.generate_initial_population(min_length, max_length)
        self.fitnesses = self.evaluate_population(self.population)
        
        # Initialize best
        current_best_idx = np.argmax(self.fitnesses)
        self.best_fitness = self.fitnesses[current_best_idx]
        self.best_individual = self.population[current_best_idx].copy()
        self.recent_improvements.append(self.best_fitness)
        
    def evolve_generation(self):
        """Perform one generation of evolution"""
        # Sort by fitness (descending)
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elites = [self.population[i] for i in sorted_indices[:self.population_manager.elite_size]]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_manager.population_size - self.population_manager.elite_size:
            parent_idx = self.population_manager.tournament_selection(self.fitnesses)
            parent = self.population[parent_idx]
            mutated = self.population_manager.mutate_individual(
                np.array(parent), self.generation, self.best_fitness
            ).tolist()
            offspring.append(mutated)
        
        # Combine elites and offspring
        self.population = elites + offspring
    
    def update_state(self):
        """Update the optimization state with current population results"""
        current_best_idx = np.argmax(self.fitnesses)
        current_fitness = self.fitnesses[current_best_idx]
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_individual = self.population[current_best_idx].copy()
            self.recent_improvements.append(current_fitness)
    
    def optimize(self, max_time_seconds: int = 85) -> List[float]:
        """Main optimization routine"""
        start_time = time.time()
        
        # Initialize state
        self.initialize_population()
        
        # Evolution loop with enhanced termination conditions
        stagnation_counter = 0
        max_stagnation = 50
        last_best_fitness = 0.0
        
        while self.generation < self.max_generations and (time.time() - start_time) < max_time_seconds - 1:
            # Update best solution
            self.update_state()
            
            # Check for stagnation
            if abs(self.best_fitness - last_best_fitness) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = self.best_fitness
                
            # Early stop if stagnating too much
            if stagnation_counter >= max_stagnation:
                break
            
            # Evolve population
            self.evolve_generation()
            
            # Evaluate new population
            self.fitnesses = self.evaluate_population(self.population)
            
            self.generation += 1
        
        # Final refinement for top solutions
        if self.best_individual is not None and self.best_fitness > 0.95:
            refined_array = self.population_manager.mutate_individual(
                np.array(self.best_individual), self.generation, self.best_fitness
            )
            refined_c2 = self.evaluator.compute_c2(refined_array.tolist())
            if refined_c2 > self.best_fitness:
                self.best_individual = refined_array.tolist()
        
        return self.best_individual if self.best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        optimizer = StatefulOptimizer(population_size=50, max_generations=200)
        f_values = optimizer.optimize(max_time_seconds=85)
        return f_values
    except Exception as e:
        # Fallback to random generation if anything fails
        print(f"Error in optimization: {e}")
        f_values = [np.random.random()] * np.random.randint(100, 1000)
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")