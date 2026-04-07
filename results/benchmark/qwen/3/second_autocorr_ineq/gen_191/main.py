# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple, Optional
import time
import nevergrad as ng
from functools import wraps

# Global constants for deterministic behavior
RANDOM_SEED = 42
MAX_EVALUATIONS = 1000
TIMEOUT_SECONDS = 85.0

class AutoconvolutionCalculator:
    """Efficient computation of autoconvolution norms"""

    @staticmethod
    @njit
    def compute_autoconvolution_numba(f_values):
        """
        Compute autoconvolution using JIT compiled function for efficiency.
        This implements the correct piecewise linear integration approach.
        """
        n_steps = len(f_values)
        if n_steps == 0:
            return np.array([])

        # Create the full convolution grid with 2*n_steps-1 points
        g_size = 2 * n_steps - 1
        g = np.zeros(g_size)

        # Compute autoconvolution using direct summation
        # g[k] = sum_{i+j=k} f[i] * f[j]
        for i in range(n_steps):
            for j in range(n_steps):
                k = i + j
                if 0 <= k < g_size:
                    g[k] += f_values[i] * f_values[j]

        return g

    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation:
        ||g||₂², ||g||₁, ||g||∞ where g = f*f
        """
        # Convert to numpy array for efficient computation
        f = np.array(f_values)

        # Compute autoconvolution using JIT compiled function
        g = AutoconvolutionCalculator.compute_autoconvolution_numba(f_values)

        # Compute norms using the piecewise linear integration method
        # For ||g||₂² we use trapezoidal-like integration:
        # ∫ g(x)² dx ≈ (dx/3)(g₀² + g₀g₁ + g₁²) + (dx/3)(g₁² + g₁g₂ + g₂²) + ...
        dx = 0.5 / len(f_values)  # Step width

        if len(g) <= 1:
            g2_sq = 0.0
        else:
            g2_sq = 0.0
            for i in range(len(g)-1):
                g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

        # ||g||₁ = sum(|g_i| * dx)
        g1 = np.sum(np.abs(g)) * dx

        # ||g||∞ = max(|g_i|)
        ginf = np.max(np.abs(g))

        return g2_sq, g1, ginf

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
    def generate_individual(size: int) -> List[float]:
        """Generate a single individual with random initialization"""
        return [random.uniform(0, 1) for _ in range(size)]

    @staticmethod
    def generate_population(population_size: int, individual_size: int) -> List[List[float]]:
        """Generate initial population"""
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
    def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Mutate an individual with given mutation rate"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1))  # Non-negative constraint
        return mutated

class EvolutionaryOptimizer:
    """Evolutionary optimization engine"""

    @staticmethod
    def optimize_population(population: List[List[float]],
                          fitness: List[float],
                          max_generations: int,
                          population_size: int) -> Tuple[float, List[float]]:
        """Perform evolutionary optimization"""
        best_c2 = 0.0
        best_individual = None

        for gen in range(max_generations):
            # Track best solution
            max_fitness_idx = np.argmax(fitness)
            current_best_c2 = fitness[max_fitness_idx]

            if current_best_c2 > best_c2:
                best_c2 = current_best_c2
                best_individual = population[max_fitness_idx].copy()

            # Evolve population
            # Sort by fitness (descending)
            sorted_pairs = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            elite = [pair[0] for pair in sorted_pairs[:population_size//4]]

            # Generate new population
            new_population = elite.copy()
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = PopulationManager.tournament_selection(population, fitness)
                parent2 = PopulationManager.tournament_selection(population, fitness)
                child = PopulationManager.crossover(parent1, parent2)
                child = PopulationManager.mutate_individual(child)
                new_population.append(child)

            population = new_population

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