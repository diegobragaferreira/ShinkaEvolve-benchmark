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
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation:
        ||g||₂², ||g||₁, ||g||∞ where g = f*f
        """
        # Convert to numpy array for efficient computation
        f = np.array(f_values)

        # Compute autoconvolution g = f * f using fast convolution
        g = signal.convolve(f, f, mode='full')

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
    def generate_multiscale_geometric_initial_function(size: int) -> List[float]:
        """
        Generate an enhanced initial function using multi-scale geometric patterns
        that encourage good C2 values with better control over convolution characteristics.
        """
        # Create a multi-scale geometric pattern with adaptive properties
        f_values = np.zeros(size)

        # Scale 1: Main body with smooth transitions
        x = np.linspace(0, 1, size)

        # Create a more sophisticated multi-scale bell pattern
        # Multiple gaussian peaks at different scales to promote uniform convolution
        pattern1 = 0.2 * np.exp(-((x - 0.2)**2) / (2 * 0.08**2)) + \
                   0.3 * np.exp(-((x - 0.5)**2) / (2 * 0.12**2)) + \
                   0.2 * np.exp(-((x - 0.8)**2) / (2 * 0.08**2)) + \
                   0.3 * np.exp(-((x - 0.35)**2) / (2 * 0.15**2))

        # Scale 2: Medium frequency oscillations for diversity
        pattern2 = 0.1 * np.sin(8 * np.pi * x) + 0.05 * np.cos(16 * np.pi * x) + \
                   0.1 * np.sin(4 * np.pi * x) * np.exp(-((x - 0.6)**2) / (2 * 0.1**2))

        # Scale 3: Low frequency modulation
        pattern3 = 0.15 * np.sin(2 * np.pi * x) * (1 + 0.2 * np.sin(6 * np.pi * x))

        # Combine all patterns
        combined = pattern1 + pattern2 + pattern3

        # Add controlled random variation to prevent over-fitting to patterns
        noise = np.random.normal(0, 0.02, size)
        combined += noise

        # Normalize and clip to [0, 1]
        combined = np.clip(combined, 0, 1)

        # Apply a soft normalization to maintain good distribution characteristics
        if np.max(combined) > 0:
            combined = combined / np.max(combined) * 0.8

        return combined.tolist()

    @staticmethod
    def generate_individual(size: int) -> List[float]:
        """Generate a single individual with enhanced geometric initialization"""
        return PopulationManager.generate_multiscale_geometric_initial_function(size)

    @staticmethod
    def generate_population(population_size: int, individual_size: int) -> List[List[float]]:
        """Generate initial population using enhanced geometric patterns"""
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
    def mutate_individual(individual: List[float], generation: int = 0,
                         max_generations: int = 50, adaptive_mutation: bool = True) -> List[float]:
        """Mutate an individual with adaptive mutation rate and step size"""
        mutated = individual.copy()

        # Adaptive mutation rate that decreases over generations
        if adaptive_mutation:
            base_mut_rate = 0.15 * (1.0 - generation/max_generations)
            # Additional factor to encourage exploration early, exploitation later
            mutation_rate = max(base_mut_rate, 0.02)
        else:
            mutation_rate = 0.1

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Adaptive step size based on current value - smaller mutations for small values
                current_val = mutated[i]
                if current_val > 0:
                    step_size = min(0.1 * current_val, 0.5)  # Cap step size
                else:
                    step_size = 0.2

                mutated[i] = max(0, mutated[i] + random.gauss(0, step_size))  # Non-negative constraint

        return mutated

class EvolutionaryOptimizer:
    """Evolutionary optimization engine"""

    @staticmethod
    def optimize_population(population: List[List[float]],
                          fitness: List[float],
                          max_generations: int,
                          population_size: int) -> Tuple[float, List[float]]:
        """Perform evolutionary optimization with adaptive parameters"""
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
                # Pass generation info for adaptive mutation
                child = PopulationManager.mutate_individual(child, gen, max_generations)
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
        """Use evolutionary search with enhanced parameters"""
        population = initial_population.copy()
        fitness = [AutoconvolutionCalculator.calculate_c2(individual)
                  for individual in population]

        return EvolutionaryOptimizer.optimize_population(population, fitness,
                                                       max_generations,
                                                       len(population))

    def run_hybrid_optimization(self, individual_size: int = 150) -> List[float]:
        """Run hybrid optimization with multiple strategies and enhanced exploration"""
        start_time = time.time()

        # Strategy 1: Enhanced evolutionary search with more generations
        initial_pop = PopulationManager.generate_population(30, individual_size)
        best_c2, best_individual = self._evolutionary_optimize(initial_pop, 40)

        # Strategy 2: Nevergrad optimization on best result with adaptive parameters
        if time.time() - start_time < TIMEOUT_SECONDS - 10:
            try:
                nevergrad_result = self._nevergrad_optimize(best_individual)
                nevergrad_c2 = AutoconvolutionCalculator.calculate_c2(nevergrad_result)
                if nevergrad_c2 > best_c2:
                    best_c2 = nevergrad_c2
                    best_individual = nevergrad_result
            except:
                pass

        # Strategy 3: Fresh nevergrad optimization with better initial population
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

        # Strategy 4: Local search refinement
        if time.time() - start_time < TIMEOUT_SECONDS - 5:
            try:
                # Apply a few rounds of coordinate-wise local search
                refined_individual = best_individual.copy()
                old_c2 = best_c2

                for coord_iter in range(10):  # Fewer iterations to save time
                    improved = False
                    for i in range(len(refined_individual)):
                        original_value = refined_individual[i]
                        # Try different step sizes for better exploration
                        step_sizes = [0.02, 0.05, 0.1]

                        for step in step_sizes:
                            for direction in [1, -1]:
                                test_individual = refined_individual.copy()
                                new_val = original_value + direction * step
                                test_individual[i] = max(0, new_val)

                                new_c2 = AutoconvolutionCalculator.calculate_c2(test_individual)
                                if new_c2 > old_c2:
                                    refined_individual = test_individual
                                    old_c2 = new_c2
                                    improved = True

                    if not improved:
                        break

                if old_c2 > best_c2:
                    best_c2 = old_c2
                    best_individual = refined_individual

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