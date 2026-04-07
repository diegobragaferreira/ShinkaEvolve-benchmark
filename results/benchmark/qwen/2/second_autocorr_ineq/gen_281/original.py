# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple, Deque
import time
from collections import deque
from abc import ABC, abstractmethod

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class NormComputer(ABC):
    """Abstract base class for norm computation strategies"""

    @abstractmethod
    def compute_autoconvolution_norms(self, f_values: List[float]) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def compute_c2(self, f_values: List[float]) -> float:
        pass

class AutoconvolutionNormComputer(NormComputer):
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

    def compute_c2(self, f_values: List[float]) -> float:
        """Compute the C2 value for given step function."""
        norm_2_squared, norm_1, norm_inf = self.compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class PopulationGenerationStrategy(ABC):
    """Abstract base class for population generation strategies"""

    @abstractmethod
    def generate_initial_population(self, population_size: int, min_length: int, max_length: int) -> List[List[float]]:
        pass

class StandardPopulationGenerator(PopulationGenerationStrategy):
    """Standard population generation with structured and random approaches"""

    def generate_initial_population(self, population_size: int, min_length: int, max_length: int) -> List[List[float]]:
        """Create diverse initial population with mixed strategies"""
        population = []

        # Strategy 1: Structured Gaussian-like approach
        structured_count = population_size // 2
        for _ in range(structured_count):
            length = np.random.randint(min_length, max_length)
            # Create base Gaussian with decreasing amplitudes
            base_shape = np.exp(-np.linspace(-2, 2, length)**2 / 2)
            # Add controlled noise
            noise = np.random.normal(0, 0.05 * np.mean(base_shape) if np.mean(base_shape) > 0 else 0.01, length)
            individual = np.clip(base_shape + noise, 0, 10.0)
            population.append(individual.tolist())

        # Strategy 2: Random approach for diversity
        random_count = population_size - structured_count
        for _ in range(random_count):
            length = np.random.randint(min_length, max_length)
            individual = np.clip(np.random.exponential(scale=0.5, size=length), 0, 10.0)
            population.append(individual.tolist())

        return population

class MutationStrategy(ABC):
    """Abstract base class for mutation strategies"""

    @abstractmethod
    def mutate_individual(self, individual: List[float],
                         generation: int = 0, best_fitness: float = 0.0,
                         recent_improvements: Deque = None) -> List[float]:
        pass

class AdaptiveMutationStrategy(MutationStrategy):
    """Enhanced adaptive mutation strategy"""

    def mutate_individual(self, individual: List[float],
                         generation: int = 0, best_fitness: float = 0.0,
                         recent_improvements: Deque = None) -> List[float]:
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

        # Occasionally perform a local smoothing or enhancement mutation
        if random.random() < 0.3 and n > 20:  # 30% chance of local smoothing
            # Adaptive window size based on sequence length and recent performance
            if recent_improvements and len(recent_improvements) >= 3:
                recent_std = np.std(list(recent_improvements)[-3:])
                if recent_std < 0.001:
                    window_size = min(5, max(2, n // 25))  # More aggressive smoothing if stagnant
                else:
                    window_size = min(5, max(1, n // 15))  # Normal smoothing
            else:
                window_size = min(5, max(1, n // 15))

            if window_size > 1:
                # Apply convolution smoothing
                smoothed = np.convolve(mutated, np.ones(window_size)/window_size, mode='same')
                # Mix with original using adaptive alpha based on fitness
                alpha = random.uniform(0.2, 0.7) if best_fitness > 0.95 else random.uniform(0.1, 0.5)
                mutated = [alpha * old + (1 - alpha) * new for old, new in zip(mutated, smoothed)]

        return mutated

class EvolutionManager:
    """Manages evolutionary operations with clear separation of concerns"""

    def __init__(self, population_size: int = 50, elite_ratio: float = 1/3, tournament_size: int = 3):
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.tournament_size = tournament_size
        self.mutation_strategy = AdaptiveMutationStrategy()

    def _tournament_selection(self, fitnesses: np.ndarray) -> int:
        """Select an index using tournament selection"""
        tournament_indices = random.sample(range(len(fitnesses)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

    def _select_parents(self, population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
        """Select parents for reproduction"""
        parents = []
        for _ in range(self.population_size - self.elite_size):
            parent_idx = self._tournament_selection(np.array(fitnesses))
            parents.append(population[parent_idx])
        return parents

    def _generate_offspring(self, parents: List[List[float]],
                          best_fitness: float, recent_improvements: Deque) -> List[List[float]]:
        """Generate new offspring through mutation"""
        offspring = []
        for parent in parents:
            mutated = self.mutation_strategy.mutate_individual(
                parent, best_fitness=best_fitness, recent_improvements=recent_improvements
            )
            offspring.append(mutated)
        return offspring

    def evolve_generation(self, population: List[List[float]],
                         fitnesses: List[float],
                         best_fitness: float,
                         recent_improvements: Deque) -> List[List[float]]:
        """Perform one generation of evolution"""
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elites = [population[i] for i in sorted_indices[:self.elite_size]]

        # Generate offspring from selected parents
        parents = self._select_parents(population, fitnesses)
        offspring = self._generate_offspring(parents, best_fitness, recent_improvements)

        # Combine elites and offspring
        return elites + offspring

class OptimizationState:
    """Manages optimization state with better encapsulation"""

    def __init__(self):
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.recent_improvements = deque(maxlen=10)
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        self.max_stagnation = 50

class OptimizerController:
    """Main orchestrator managing the complete optimization process"""

    def __init__(self,
                 norm_computer: NormComputer,
                 population_generator: PopulationGenerationStrategy,
                 evolution_manager: EvolutionManager,
                 max_generations: int = 200,
                 max_time_seconds: int = 85):
        self.norm_computer = norm_computer
        self.population_generator = population_generator
        self.evolution_manager = evolution_manager
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.state = OptimizationState()

    def evaluate_population(self, population: List[List[float]]) -> List[float]:
        """Evaluate fitness for entire population in parallel"""
        def evaluate_single(individual):
            try:
                return self.norm_computer.compute_c2(individual)
            except Exception:
                return 0.0

        # Parallel evaluation
        fitnesses = Parallel(n_jobs=-1, backend='threading')(
            delayed(evaluate_single)(ind) for ind in population
        )

        return fitnesses

    def _update_state(self, population: List[List[float]], fitnesses: List[float]):
        """Update optimization state with current population results"""
        current_best_idx = np.argmax(fitnesses)
        current_fitness = fitnesses[current_best_idx]

        if current_fitness > self.state.best_fitness:
            self.state.best_fitness = current_fitness
            self.state.best_individual = population[current_best_idx].copy()
            self.state.recent_improvements.append(current_fitness)
            self.state.stagnation_counter = 0
        else:
            self.state.stagnation_counter += 1

        self.state.last_best_fitness = current_fitness

    def _should_terminate(self, start_time: float) -> bool:
        """Check if optimization should terminate"""
        # Time limit check
        if (time.time() - start_time) >= self.max_time_seconds - 1:
            return True

        # Stagnation check
        if self.state.stagnation_counter >= self.state.max_stagnation:
            return True

        # Generation limit check
        if self.state.generation >= self.max_generations:
            return True

        return False

    def optimize(self, min_length: int = 100, max_length: int = 1000) -> List[float]:
        """Main optimization routine with improved control flow"""
        start_time = time.time()

        # Phase 1: Initialization
        population = self.population_generator.generate_initial_population(
            self.evolution_manager.population_size, min_length, max_length
        )
        fitnesses = self.evaluate_population(population)

        # Initialize state
        self._update_state(population, fitnesses)

        # Phase 2: Evolution Loop
        while not self._should_terminate(start_time):
            # Evolve population
            population = self.evolution_manager.evolve_generation(
                population, fitnesses, self.state.best_fitness, self.state.recent_improvements
            )

            # Evaluate new population
            fitnesses = self.evaluate_population(population)

            # Update state
            self._update_state(population, fitnesses)

            self.state.generation += 1

        # Phase 3: Final refinement
        if self.state.best_individual is not None and self.state.best_fitness > 0.95:
            refined = self.evolution_manager.mutation_strategy.mutate_individual(
                self.state.best_individual,
                generation=self.state.generation,
                best_fitness=self.state.best_fitness,
                recent_improvements=self.state.recent_improvements
            )
            refined_c2 = self.norm_computer.compute_c2(refined)
            if refined_c2 > self.state.best_fitness:
                self.state.best_individual = refined

        return self.state.best_individual if self.state.best_individual is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        # Initialize components
        norm_computer = AutoconvolutionNormComputer()
        population_generator = StandardPopulationGenerator()
        evolution_manager = EvolutionManager(population_size=50, elite_ratio=1/3, tournament_size=3)

        # Create controller
        optimizer = OptimizerController(
            norm_computer=norm_computer,
            population_generator=population_generator,
            evolution_manager=evolution_manager,
            max_generations=200,
            max_time_seconds=85
        )

        # Run optimization
        f_values = optimizer.optimize()
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