# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from scipy import signal
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormCalculator:
    """Handles all norm computations for C2 calculation"""
    
    def __init__(self, fine_grid_points: int = 1000):
        self.fine_grid_points = fine_grid_points
    
    @staticmethod
    @jit(nopython=True)
    def _compute_convolution_norms_numba(f_values: List[float]) -> Tuple[float, float, float]:
        """Compute the three norms needed for C2 calculation using numba JIT acceleration"""
        # Convert to numpy array
        f = np.array(f_values)
        n_steps = len(f)

        # Create step function on [-1/4, 1/4] with proper spacing
        step_width = 0.5 / n_steps
        step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)

        # For autoconvolution, we'll work with a finer grid
        # Create piecewise constant function on refined grid
        x_fine = np.linspace(-0.25, 0.25, 1000)  # Fine grid for convolution
        dx = x_fine[1] - x_fine[0]

        # Build piecewise constant function
        f_func = np.zeros_like(x_fine)
        for i in range(n_steps):
            pos = step_positions[i]
            left = pos - step_width/2
            right = pos + step_width/2
            # Find indices where x_fine falls in this step
            mask = (x_fine >= left) & (x_fine <= right)
            f_func[mask] = f[i]

        # Perform autoconvolution efficiently using scipy.signal.convolve
        # Note: For symmetric functions, we want g = f * f (autoconvolution)
        g = signal.convolve(f_func, f_func, mode='full')
        g = g[:len(g)//2 + 1]  # Take only first half (since it's symmetric)

        # Adjust for proper scaling due to discretization
        g = g * dx

        # Compute the required norms
        g_squared = g**2
        g_abs = np.abs(g)

        # ||g||₂² (using trapezoidal rule for integration)
        norm_2_squared = np.trapz(g_squared, dx=dx)

        # ||g||₁ (L1 norm)
        norm_1 = np.trapz(g_abs, dx=dx)

        # ||g||∞ (infinity norm)
        norm_inf = np.max(g_abs)

        return norm_2_squared, norm_1, norm_inf
    
    def compute_c2(self, f_values: List[float]) -> float:
        """Calculate C₂ from step function values"""
        try:
            norm_2_squared, norm_1, norm_inf = self._compute_convolution_norms_numba(f_values)

            # Avoid division by zero
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 0.0

            c2 = norm_2_squared / (norm_1 * norm_inf)
            return c2
        except Exception as e:
            logger.error(f"Error in compute_c2: {e}")
            return 0.0

class PopulationManager:
    """Manages population initialization and genetic operations"""
    
    def __init__(self, min_steps: int, max_steps: int, seed: int = 42):
        self.min_steps = min_steps
        self.max_steps = max_steps
        np.random.seed(seed)
    
    def initialize_population(self, pop_size: int) -> List[List[float]]:
        """Initialize population with diverse step functions"""
        population = []
        for _ in range(pop_size):
            # Random number of steps
            n_steps = np.random.randint(self.min_steps, self.max_steps)
            # Random heights with some structure
            heights = np.random.exponential(scale=1.0, size=n_steps)
            # Clip negative values
            heights = np.maximum(heights, 0)
            population.append(heights.tolist())
        return population
    
    @staticmethod
    def crossover(parent1: List[float], parent2: List[float], crossover_rate: float = 0.8) -> Tuple[List[float], List[float]]:
        """Perform crossover between two parents"""
        if len(parent1) != len(parent2):
            # Make them same length by truncating or padding
            min_len = min(len(parent1), len(parent2))
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]

        if np.random.random() < crossover_rate:
            # Uniform crossover
            child1, child2 = [], []
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
            return child1, child2
        else:
            return parent1, parent2
    
    @staticmethod
    def mutate(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Mutate individual with Gaussian noise"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1 * mutated[i] + 0.01)
                mutated[i] = max(0, mutated[i] + noise)
        return mutated

class EvolutionaryOptimizer:
    """Main evolutionary optimization engine"""
    
    def __init__(self, 
                 population_size: int = 200,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 5,
                 min_steps: int = 100,
                 max_steps: int = 50000,
                 seed: int = 42):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.seed = seed
        
        # Initialize components
        self.norm_calculator = NormCalculator()
        self.population_manager = PopulationManager(min_steps, max_steps, seed)
        
        # Statistics tracking
        self.best_fitness_history = []
        self.best_individual_history = []
    
    def evaluate_fitness(self, population: List[List[float]]) -> List[float]:
        """Evaluate fitness of entire population"""
        results = []
        for ind in population:
            results.append(self.norm_calculator.compute_c2(ind))
        return results
    
    @staticmethod
    def tournament_selection(population: List[List[float]], 
                           fitness_scores: List[float], 
                           tournament_size: int = 5) -> List[List[float]]:
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            # Tournament
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected
    
    @staticmethod
    def elitism(population: List[List[float]], 
               fitness_scores: List[float], 
               elite_count: int) -> List[List[float]]:
        """Keep best individuals"""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i].copy() for i in sorted_indices[:elite_count]]
        return elite
    
    def run_evolution(self) -> Tuple[List[float], float]:
        """Run the main evolutionary algorithm"""
        # Initialize population
        population = self.population_manager.initialize_population(self.population_size)

        best_individual = None
        best_fitness = -np.inf

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(population)

            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                
                # Store history
                self.best_fitness_history.append(best_fitness)
                self.best_individual_history.append(best_individual.copy())

            # Print progress every 10 generations
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best C2 = {best_fitness:.4f}")

            # Elitism
            elite = self.elitism(population, fitness_scores, self.elitism_count)

            # Selection
            parents = self.tournament_selection(population, fitness_scores)

            # Crossover and mutation
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.population_manager.crossover(
                    parents[p1], parents[p2], self.crossover_rate)

                child1 = self.population_manager.mutate(child1, self.mutation_rate)
                child2 = self.population_manager.mutate(child2, self.mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

        return best_individual, best_fitness

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Create optimizer instance with specified parameters
    optimizer = EvolutionaryOptimizer(
        population_size=200,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=5,
        min_steps=100,
        max_steps=50000,
        seed=42
    )
    
    # Run evolution
    best_individual, best_fitness = optimizer.run_evolution()

    end_time = time.time()
    eval_time = end_time - start_time

    logger.info(f"Evaluated in {eval_time:.2f} seconds")
    logger.info(f"Best C2 found: {best_fitness:.6f}")

    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")