# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple, Optional
import time
from collections import deque
from scipy.ndimage import gaussian_filter1d

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

class EvolutionEngine:
    """Manages the evolutionary process with enhanced adaptive strategies"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 200):
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_size = max(1, population_size // 3)
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
        
        # Strategy 2: Random exponential approach  
        for _ in range(self.population_size // 2):
            length = np.random.randint(min_length, max_length)
            individual = np.clip(np.random.exponential(scale=0.5, size=length), 0, 10.0)
            population.append(individual.tolist())
            
        return population
    
    def tournament_selection(self, population: List[List[float]], 
                           fitnesses: List[float]) -> List[float]:
        """Select an individual using tournament selection with adaptive pressure"""
        # Adjust tournament size based on optimization stage
        current_tournament_size = max(2, min(5, self.tournament_size + int(len(fitnesses) > 100)))
        tournament_indices = random.sample(range(len(population)), current_tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()
    
    def mutate_individual(self, individual: List[float], 
                         mutation_rate: float = 0.1, 
                         generation: int = 0) -> List[float]:
        """Apply mutation to individual with adaptive strategy"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Apply Gaussian perturbation to some elements
        for i in range(n):
            if random.random() < mutation_rate:
                # Add small Gaussian noise to element with adaptive scale
                noise_scale = 0.05 * np.mean(mutated) if np.mean(mutated) > 0 else 0.01
                mutated[i] += np.random.normal(0, noise_scale)
                # Ensure non-negativity
                mutated[i] = max(0.0, mutated[i])
                
        # Occasionally perform advanced smoothing that preserves features
        if random.random() < 0.3:  # 30% chance of local enhancement
            # Choose window size based on problem size and generation
            min_window = min(3, max(1, n // 20))
            max_window = min(15, max(3, n // 5))
            window_size = np.random.randint(min_window, max_window + 1)
            if window_size > 1:
                # Apply Gaussian smoothing to remove high-frequency noise
                smoothed = gaussian_filter1d(mutated, sigma=window_size/3.0, mode='nearest')
                # Blend with original to preserve important features
                alpha = 0.3 + 0.4 * random.random()  # Blend factor
                mutated = [alpha * old + (1 - alpha) * new for old, new in zip(mutated, smoothed)]
                    
        return mutated
    
    def evolve_generation(self, population: List[List[float]], 
                         fitnesses: List[float], generation: int) -> List[List[float]]:
        """Generate next generation using tournament selection and mutation"""
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elites = [population[i] for i in sorted_indices[:self.elite_size]]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size - self.elite_size:
            parent = self.tournament_selection(population, fitnesses)
            mutated = self.mutate_individual(parent, generation=generation)
            offspring.append(mutated)
        
        # Combine elites and offspring
        return elites + offspring
    
    def adaptive_population_adjustment(self, generation: int, population_size: int, 
                                     recent_improvements: deque, diversity_metric: float) -> int:
        """Adjust population size based on performance trends and diversity"""
        # If we're making steady progress and diversity is low, expand population
        if generation % 50 == 0 and generation > 0:
            recent_std = np.std(list(recent_improvements)[-3:]) if len(recent_improvements) >= 3 else 0
            if recent_std < 0.001 and diversity_metric < 0.05:  # Very homogeneous population
                return min(120, population_size + 20)
            elif recent_std > 0.005:  # Making significant progress
                return max(40, min(120, population_size + 5))
            else:
                return max(40, min(120, population_size - 10))
        return population_size

class StepFunctionOptimizer:
    """Main optimizer class orchestrating the complete process with improved control flow"""
    
    def __init__(self):
        self.evaluator = AutoconvolutionEvaluator()
        self.engine = EvolutionEngine()
        self.recent_improvements = deque(maxlen=10)
    
    def evaluate_population(self, population: List[List[float]], 
                          early_terminate_threshold: float = 0.8) -> List[float]:
        """Evaluate fitness for entire population in parallel"""
        def evaluate_single(individual):
            try:
                c2 = self.evaluator.compute_c2(individual)
                return c2
            except Exception:
                return 0.0
        
        # Parallel evaluation
        fitnesses = Parallel(n_jobs=-1, backend='threading')(
            delayed(evaluate_single)(ind) for ind in population
        )
        
        return fitnesses
    
    def calculate_diversity(self, population: List[List[float]]) -> float:
        """Calculate population diversity metric"""
        if len(population) < 2:
            return 0.0
        # Calculate standard deviation of mean values
        means = [np.mean(ind) for ind in population]
        return np.std(means) / (np.mean(means) + 1e-10) if np.mean(means) > 0 else 0.0
    
    def optimize(self, max_time_seconds: int = 85) -> List[float]:
        """Main optimization routine with enhanced control logic"""
        start_time = time.time()
        
        # Initialize population
        population = self.engine.generate_initial_population()
        fitnesses = self.evaluate_population(population)
        
        best_fitness = -float('inf')
        best_individual = None
        
        # Evolution loop with enhanced termination conditions
        generation = 0
        stagnation_counter = 0
        max_stagnation = 50
        last_best_fitness = 0.0
        
        while generation < self.engine.max_generations and (time.time() - start_time) < max_time_seconds - 1:
            # Update best solution
            current_best_idx = np.argmax(fitnesses)
            current_fitness = fitnesses[current_best_idx]
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_individual = population[current_best_idx].copy()
                self.recent_improvements.append(current_fitness)
            
            # Check for stagnation
            if abs(current_fitness - last_best_fitness) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = current_fitness
                
            # Early stop if stagnating too much
            if stagnation_counter >= max_stagnation:
                break
            
            # Calculate diversity for adaptive population sizing
            diversity = self.calculate_diversity(population)
            
            # Evolve population
            population = self.engine.evolve_generation(population, fitnesses, generation)
            
            # Evaluate new population
            fitnesses = self.evaluate_population(population)
            
            # Adaptive population size adjustment
            population_size = self.engine.adaptive_population_adjustment(
                generation, self.engine.population_size, self.recent_improvements, diversity
            )
            self.engine.population_size = population_size
            
            # Adjust elite size based on generation
            elite_percentage = 0.2 + 0.1 * (generation / self.engine.max_generations)
            self.engine.elite_size = max(1, int(population_size * elite_percentage))
            
            # Adjust tournament size based on progress
            if generation > 100:
                self.engine.tournament_size = 3
            else:
                self.engine.tournament_size = 2 + int(best_fitness > 0.9)
            
            # Adjust mutation rate based on progress
            if best_fitness > 0.95:  # If we're close to good solution
                mutation_rate = 0.03 + 0.02 * random.random()
            elif best_fitness > 0.9:
                mutation_rate = 0.05 + 0.03 * random.random()
            else:
                mutation_rate = 0.08 + 0.07 * random.random()
                
            # Apply new mutation rate to all individuals (for next round)
            # Note: mutation is applied during evolution step itself
            
            generation += 1
        
        # Final refinement for top solutions
        if best_individual is not None and best_fitness > 0.95:
            refined = self.engine.mutate_individual(best_individual, mutation_rate=0.03, generation=generation)
            refined_c2 = self.evaluator.compute_c2(refined)
            if refined_c2 > best_fitness:
                best_individual = refined
        
        return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        optimizer = StepFunctionOptimizer()
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