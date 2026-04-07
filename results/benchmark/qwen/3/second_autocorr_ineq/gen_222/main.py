# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple, Optional
import time
import nevergrad as ng
from functools import wraps
from numba import jit

# Global constants for deterministic behavior
RANDOM_SEED = 42
MAX_EVALUATIONS = 1000
TIMEOUT_SECONDS = 85.0

class AutoconvolutionCalculator:
    """Efficient computation of autoconvolution norms"""

    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_fast(f_vals):
        """Fast numba-based autoconvolution computation for step functions"""
        n = len(f_vals)
        # Result has length 2*n-1
        g = np.zeros(2*n - 1)

        # Manual computation of the convolution sum for step functions
        for i in range(n):
            for j in range(n):
                # In convolution, the value at index i+j comes from f[i] * f[j]
                g[i + j] += f_vals[i] * f_vals[j]

        return g

    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation:
        ||g||₂², ||g||₁, ||g||∞ where g = f*f
        """
        # Convert to numpy array for efficient computation
        f = np.array(f_values)

        # Compute autoconvolution g = f * f using fast numba implementation
        g = AutoconvolutionCalculator.compute_autoconvolution_fast(f)

        # Only keep the middle portion corresponding to valid convolution
        center_idx = len(g) // 2
        half_len = len(f)
        g = g[center_idx - half_len + 1:center_idx + half_len]

        # Compute norms
        g_squared = g * g
        norm_l2_sq = np.sum(g_squared)

        norm_l1 = np.sum(np.abs(g))
        norm_linf = np.max(np.abs(g))

        return norm_l2_sq, norm_l1, norm_linf

    @staticmethod
    def calculate_c2(f_values: List[float]) -> float:
        """Calculate C2 value for given step function"""
        try:
            norm_l2_sq, norm_l1, norm_linf = AutoconvolutionCalculator.compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_l1 <= 1e-12 or norm_linf <= 1e-12:
                return 0.0

            c2 = norm_l2_sq / (norm_l1 * norm_linf)
            return c2
        except Exception:
            return 0.0

class PopulationManager:
    """Handles population creation, selection, and evolution"""

    @staticmethod
    def generate_geometric_initial_function(size: int) -> List[float]:
        """
        Generate an initial function using enhanced geometric patterns that encourage good C2 values.
        Creates a function with strategic peaks and valleys to promote flat convolution profiles.
        """
        # Create a more sophisticated multi-scale pattern
        f_values = np.zeros(size)

        # Main pattern with multiple strategically placed peaks
        x = np.linspace(0, 1, size)

        # Multiple peaks that create constructive interference in convolution
        pattern1 = 0.5 * np.exp(-((x - 0.2)**2) / (2 * 0.08**2)) + \
                   0.3 * np.exp(-((x - 0.5)**2) / (2 * 0.06**2)) + \
                   0.4 * np.exp(-((x - 0.8)**2) / (2 * 0.07**2))

        # Add complementary pattern to create more balanced distribution
        pattern2 = 0.2 * np.sin(10 * np.pi * x) + 0.1 * np.cos(20 * np.pi * x)

        # Add some structured variation to avoid overly regular patterns
        pattern3 = 0.1 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * x)

        # Combine patterns
        combined = pattern1 + pattern2 + pattern3 + 0.1  # Add offset to ensure positivity

        # Normalize and clip to [0, 1]
        combined = np.clip(combined, 0, 1)

        # Apply a more aggressive smoothing to reduce extreme variations
        # This helps prevent sharp peaks that would hurt C2
        smoothed = np.convolve(combined, np.ones(5)/5, mode='same')

        # Ensure non-negativity and reasonable distribution
        f_values = np.clip(smoothed, 0, 1)

        return f_values.tolist()

    @staticmethod
    def generate_individual(size: int) -> List[float]:
        """Generate a single individual with geometric initialization instead of purely random"""
        return PopulationManager.generate_geometric_initial_function(size)

    @staticmethod
    def generate_population(population_size: int, individual_size: int) -> List[List[float]]:
        """Generate initial population using geometric patterns"""
        return [PopulationManager.generate_individual(individual_size)
                for _ in range(population_size)]

    @staticmethod
    def tournament_selection(population: List[List[float]], fitness: List[float],
                             k: int = 3) -> List[float]:
        """Select individual using tournament selection"""
        selected_indices = random.sample(range(len(population)), k)
        best_idx = max(selected_indices, key=lambda i: fitness[i])
        return population[best_idx]

    @staticmethod
    def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
        """Single-point crossover between two parents"""
        point = random.randint(1, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]
        return child

    @staticmethod
    def mutate_individual(individual: List[float], mutation_rate: float = 0.1, generation: int = 0, max_gens: int = 100) -> List[float]:
        """Mutate an individual with adaptive mutation rate"""
        # Adapt mutation rate: start high, decrease over time
        adapted_rate = mutation_rate * (1.0 - (generation / max_gens) * 0.8)
        adapted_rate = max(0.01, adapted_rate)  # Minimum mutation rate

        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < adapted_rate:
                # Use smaller mutation step for smaller values to maintain proportionality
                mutation_strength = 0.1 * mutated[i] if mutated[i] > 0 else 0.1
                mutated[i] = max(0, mutated[i] + random.gauss(0, mutation_strength))  # Non-negative constraint
        return mutated

class EvolutionaryOptimizer:
    """Evolutionary optimization engine"""

    @staticmethod
    def optimize_population(population: List[List[float]],
                          fitness: List[float],
                          max_generations: int,
                          population_size: int) -> Tuple[float, List[float]]:
        """Perform evolutionary optimization with enhanced strategies"""
        best_c2 = 0.0
        best_individual = None

        for gen in range(max_generations):
            # Track best solution
            max_fitness_idx = np.argmax(fitness)
            current_best_c2 = fitness[max_fitness_idx]

            if current_best_c2 > best_c2:
                best_c2 = current_best_c2
                best_individual = population[max_fitness_idx].copy()

            # Evolve population with enhanced strategy
            # Sort by fitness (descending)
            sorted_pairs = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            elite_count = max(2, population_size // 8)  # Smaller elite in early generations
            elite = [pair[0] for pair in sorted_pairs[:elite_count]]

            # Generate new population with adaptive parameters
            new_population = elite.copy()

            # Additional diversity mechanism: introduce some random individuals occasionally
            if gen % 10 == 0 and gen > 0:
                # Add some fresh individuals to maintain diversity
                for _ in range(2):
                    new_population.append(PopulationManager.generate_individual(len(population[0])))

            while len(new_population) < population_size:
                # Tournament selection with adaptive size
                tournament_size = max(3, min(8, population_size // 4 + gen // 5))
                parent1 = PopulationManager.tournament_selection(population, fitness, tournament_size)
                parent2 = PopulationManager.tournament_selection(population, fitness, tournament_size)
                child = PopulationManager.crossover(parent1, parent2)
                child = PopulationManager.mutate_individual(child, generation=gen, max_gens=max_generations)
                new_population.append(child)

            population = new_population[:population_size]

            # Evaluate new population
            fitness = [AutoconvolutionCalculator.calculate_c2(individual)
                      for individual in population]

        return best_c2, best_individual

class HybridOptimizer:
    """Main hybrid optimizer combining multiple strategies"""

    def __init__(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _nevergrad_optimize(self, initial_guess: List[float]) -> List[float]:
        """Use nevergrad for optimization"""
        def objective(x):
            # Clip negative values
            x = np.maximum(x, 0)
            # Calculate C2 (we want to maximize it, so minimize negative)
            return -AutoconvolutionCalculator.calculate_c2(x.tolist())

        # Create optimizer with appropriate settings
        optimizer = ng.optimizers.OnePlusOne(
            dimension=len(initial_guess),
            budget=min(MAX_EVALUATIONS, 500),  # Limit evaluations
            num_workers=1
        )

        # Optimize
        recommendation = optimizer.minimize(objective, verbosity=0)

        # Get the best solution
        result = recommendation.args[0]

        # Ensure non-negativity
        result = np.maximum(result, 0)

        # Return as list
        return result.tolist()

    def _evolutionary_optimize(self, initial_population: List[List[float]],
                             max_generations: int = 50) -> Tuple[float, List[float]]:
        """Use evolutionary search"""
        population = initial_population.copy()
        fitness = [AutoconvolutionCalculator.calculate_c2(individual)
                  for individual in population]

        return EvolutionaryOptimizer.optimize_population(population, fitness,
                                                       max_generations,
                                                       len(population))

    def run_hybrid_optimization(self, individual_size: int = 150) -> List[float]:
        """Run hybrid optimization with multiple strategies"""
        start_time = time.time()

        # Strategy 1: Basic evolutionary search
        initial_pop = PopulationManager.generate_population(30, individual_size)
        best_c2, best_individual = self._evolutionary_optimize(initial_pop, 30)

        # Strategy 2: Nevergrad optimization on best result
        if time.time() - start_time < TIMEOUT_SECONDS - 10:
            try:
                nevergrad_result = self._nevergrad_optimize(best_individual)
                nevergrad_c2 = AutoconvolutionCalculator.calculate_c2(nevergrad_result)
                if nevergrad_c2 > best_c2:
                    best_c2 = nevergrad_c2
                    best_individual = nevergrad_result
            except:
                pass

        # Strategy 3: Fresh nevergrad optimization
        if time.time() - start_time < TIMEOUT_SECONDS - 10:
            try:
                fresh_start = PopulationManager.generate_individual(individual_size)
                fresh_result = self._nevergrad_optimize(fresh_start)
                fresh_c2 = AutoconvolutionCalculator.calculate_c2(fresh_result)
                if fresh_c2 > best_c2:
                    best_c2 = fresh_c2
                    best_individual = fresh_result
            except:
                pass

        return best_individual if best_individual is not None else PopulationManager.generate_individual(individual_size)

def optimized_construct_function() -> List[float]:
    """Construct step function using hybrid optimization"""
    optimizer = HybridOptimizer()
    try:
        result = optimizer.run_hybrid_optimization(150)
        return result
    except Exception:
        # Fallback to simpler approach if optimization fails
        return PopulationManager.generate_individual(100)

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Try hybrid optimization first
    start_time = time.time()
    try:
        result = optimized_construct_function()
        elapsed = time.time() - start_time
        if elapsed < TIMEOUT_SECONDS:
            return result
    except Exception:
        pass

    # Fallback to simpler approach if optimization fails or times out
    return PopulationManager.generate_individual(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")