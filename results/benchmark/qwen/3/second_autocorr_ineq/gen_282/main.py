# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import warnings
import time
from typing import List, Tuple, Optional, Callable
warnings.filterwarnings('ignore')

# Configuration constants
MAX_TIME_SECONDS = 90
POPULATION_SIZE = 30
GENERATIONS = 50
INITIAL_MUTATION_RATE = 0.3
MIN_MUTATION_RATE = 0.05
ELITISM_COUNT = 1
TOURNAMENT_SIZE = 4
LOCAL_REFINEMENT_ITERATIONS = 20

@njit
def compute_autoconvolution_norms_numba(f_array: np.ndarray) -> tuple:
    """
    Compute the L2, L1, and L-infinity norms of the autoconvolution of f.
    JIT compiled version for speed.

    Args:
        f_array: Numpy array of step heights

    Returns:
        Tuple of (||g||₂², ||g||₁, ||g||∞)
    """
    if len(f_array) == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution g = f * f (discrete convolution)
    # Manual implementation for better control and JIT compatibility
    g_len = 2 * len(f_array) - 1
    g = np.zeros(g_len, dtype=np.float64)

    # Direct convolution computation - fully JIT compiled
    for i in range(len(f_array)):
        for j in range(len(f_array)):
            g[i + j] += f_array[i] * f_array[j]

    # Compute norms using manual loop for JIT compatibility
    # ||g||₂² - integrate g² using trapezoidal rule approximation
    g_squared = g * g
    trapz_sum = 0.0

    # Use trapezoidal integration for ||g||₂²
    if len(g) >= 2:
        h = 1.0 / (len(g) - 1)  # Normalized spacing
        for i in range(len(g) - 1):
            y1, y2 = g_squared[i], g_squared[i+1]
            # Correct trapezoidal formula for g² integration: (h/3)(y₁² + y₁y₂ + y₂²)
            trapz_sum += (y1*y1 + y1*y2 + y2*y2) * h / 3.0
    else:
        trapz_sum = g_squared[0] if len(g_squared) > 0 else 0.0

    # ||g||₁ - integrate |g| using trapezoidal rule
    g_abs = np.abs(g)
    trapz_l1_sum = 0.0

    if len(g) >= 2:
        h = 1.0 / (len(g) - 1)  # Normalized spacing
        for i in range(len(g) - 1):
            y1, y2 = g_abs[i], g_abs[i+1]
            trapz_l1_sum += (y1 + y2) * h / 2.0
    else:
        trapz_l1_sum = g_abs[0] if len(g_abs) > 0 else 0.0

    # ||g||∞ - infinity norm (maximum absolute value)
    g_max = np.max(np.abs(g)) if len(g) > 0 else 0.0

    return trapz_sum, trapz_l1_sum, g_max

class NormComputer:
    """Encapsulates normalization computations for better modularity."""
    
    @staticmethod
    def compute_autoconvolution_norms(f: list) -> tuple:
        """
        Compute the L2, L1, and L-infinity norms of the autoconvolution of f.

        Args:
            f: List of step heights

        Returns:
            Tuple of (||g||₂², ||g||₁, ||g||∞)
        """
        if not f:
            return 0.0, 0.0, 0.0

        # Convert to numpy array for easier manipulation
        f_array = np.array(f, dtype=np.float64)

        return compute_autoconvolution_norms_numba(f_array)

class ObjectiveFunction:
    """Encapsulates objective function calculations."""
    
    @staticmethod
    def calculate_c2(f: list) -> float:
        """
        Calculate C₂ = ||g||₂² / (||g||₁ · ||g||∞) where g = f * f.

        Args:
            f: List of step heights

        Returns:
            C₂ value
        """
        try:
            g_norm2_sq, g_norm1, g_norm_inf = NormComputer.compute_autoconvolution_norms(f)

            # Avoid division by zero
            if g_norm1 <= 1e-15 or g_norm_inf <= 1e-15:
                return 0.0

            return g_norm2_sq / (g_norm1 * g_norm_inf)
        except Exception as e:
            return 0.0

class Initializer:
    """Handles initialization of step function parameters."""
    
    @staticmethod
    def generate_multiscale_gaussian_pattern(size: int) -> np.ndarray:
        """
        Generate a multi-scale Gaussian pattern for initialization.
        This creates a more sophisticated hierarchical structure with optimized scale distribution
        and strategic positioning for better autoconvolution behavior.
        """
        # Create multi-scale Gaussian pattern for structured initialization
        pattern = np.zeros(size)

        # Generate multiple scales with logarithmic distribution for better coverage
        # Using log-space scales to ensure we sample both very wide and very narrow features
        num_scales = 6
        base_scale = size // 16
        scale_factors = np.logspace(np.log10(1), np.log10(size//4), num_scales, base=2.0)
        scale_factors = np.unique(scale_factors.astype(int))  # Remove duplicates

        # Amplitude decreases with scale for hierarchical structure
        amplitudes = 1.0 / (np.arange(1, len(scale_factors) + 1) * 2.0)

        # Place Gaussian bumps strategically across the domain
        for i, (scale, amp) in enumerate(zip(scale_factors, amplitudes)):
            if scale >= 1:  # Only create bumps with meaningful scale
                # Position bumps more evenly across domain for better exploration
                position = int((i + 1) * size / (len(scale_factors) + 2))
                # Ensure position is within bounds
                position = max(0, min(position, size - 1))
                # Generate Gaussian bump
                indices = np.arange(size)
                gaussian = amp * np.exp(-0.5 * (indices - position)**2 / scale**2)
                pattern += gaussian

        # Add more complex harmonic structure with multiple frequencies
        t = np.linspace(-2, 2, size)
        harmonic_pattern = (
            0.8 +
            0.3 * np.sin(2 * np.pi * t) +
            0.2 * np.cos(4 * np.pi * t) +
            0.1 * np.sin(6 * np.pi * t) +
            0.05 * np.cos(8 * np.pi * t) +
            0.03 * np.sin(10 * np.pi * t)
        )

        # Combine all components and ensure non-negativity
        pattern = np.maximum(pattern + harmonic_pattern, 0.0)

        # Add some randomness to break symmetry and escape local optima
        noise = np.random.exponential(0.05, size)
        pattern = np.maximum(pattern + noise, 0.0)

        # Normalize to reasonable magnitude
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * 100

        return pattern

class ProjectionHandler:
    """Handles projection of continuous values to valid discrete step function heights."""
    
    @staticmethod
    def create_discrete_projection(x: np.ndarray) -> np.ndarray:
        """
        Project continuous values to valid discrete step function heights.
        Ensures non-negativity and reasonable scaling.
        """
        # Ensure non-negativity
        x_proj = np.maximum(x, 0.0)

        # Normalize to prevent extreme values that might cause numerical issues
        if np.sum(x_proj) > 0:
            x_proj = x_proj / np.sum(x_proj) * 100

        return x_proj

class EvolutionaryOperator:
    """Handles evolutionary operations like mutation and selection."""
    
    @staticmethod
    def evolve_individual(parent: np.ndarray, mutation_strength: float = 0.1) -> np.ndarray:
        """
        Evolve an individual with adaptive mutation.
        """
        child = parent.copy()

        # Apply mutations to random positions
        mutation_mask = np.random.random(len(child)) < 0.1
        if np.any(mutation_mask):
            noise = np.random.normal(0, mutation_strength, len(child))
            child[mutation_mask] += noise[mutation_mask]
            child = np.maximum(child, 0.0)  # Ensure non-negativity

        return child

    @staticmethod
    def tournament_selection(population: list, fitness_scores: list) -> np.ndarray:
        """
        Select an individual using tournament selection.
        """
        tournament_indices = np.random.choice(len(population),
                                             size=TOURNAMENT_SIZE,
                                             replace=False)
        best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
        return population[best_idx].copy()

    @staticmethod
    def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1, parent2

        # Uniform crossover
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # Use element-wise random selection for crossover
        mask = np.random.random(len(parent1)) < 0.5
        child1[mask] = parent1[mask]
        child1[~mask] = parent2[~mask]
        child2[mask] = parent2[mask]
        child2[~mask] = parent1[~mask]
        
        return child1, child2

class EvolutionaryOptimizer:
    """Main evolutionary optimization class with modular design."""
    
    def __init__(self):
        self.start_time = None
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.fitness_history = []
        
        # Set seeds for reproducibility
        np.random.seed(42)
        np.random.seed(42)
    
    def check_time_limit(self) -> bool:
        """Check if we should stop due to time limit."""
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time > MAX_TIME_SECONDS
    
    def generate_initial_population(self, size: int, pop_size: int) -> List[np.ndarray]:
        """Generate initial population with diverse patterns."""
        population = []
        
        # Add structured individuals
        for _ in range(pop_size // 2):
            population.append(Initializer.generate_multiscale_gaussian_pattern(size))
            
        # Add random individuals
        for _ in range(pop_size // 2):
            individual = np.random.uniform(0, 100, size)
            population.append(individual)
            
        return population
    
    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness for entire population."""
        fitness_scores = []
        for individual in population:
            # Convert back to list for objective function
            individual_list = individual.tolist()
            fitness = ObjectiveFunction.calculate_c2(individual_list)
            fitness_scores.append(fitness)
        return fitness_scores
    
    def adapt_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adaptively adjust mutation rate based on generation and diversity."""
        # Start with high mutation for exploration, decrease with generation
        base_mutation = INITIAL_MUTATION_RATE * (1.0 - generation / GENERATIONS * 0.8)
        base_mutation = max(MIN_MUTATION_RATE, base_mutation)
        
        # Increase mutation if diversity is low (premature convergence risk)
        if diversity < 0.1:
            return base_mutation * 2.0
        else:
            return base_mutation
    
    def evolve_generation(self, population: List[np.ndarray], 
                         fitness_scores: List[float], generation: int) -> List[np.ndarray]:
        """Evolve one generation with adaptive parameters."""
        new_population = []
        pop_size = len(population)

        # Calculate population diversity
        fitness_array = np.array(fitness_scores)
        diversity = np.std(fitness_array) / (np.mean(fitness_array) + 1e-10) if len(fitness_array) > 1 else 0.0

        # Adaptive mutation strength
        mutation_rate = self.adapt_mutation_rate(generation, diversity)

        # Elitism: keep best individual
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx].copy())

        # Generate rest through selection, crossover, and mutation
        for _ in range(pop_size - ELITISM_COUNT):
            # Selection
            parent1 = EvolutionaryOperator.tournament_selection(population, fitness_scores)
            parent2 = EvolutionaryOperator.tournament_selection(population, fitness_scores)

            # Crossover
            child1, child2 = EvolutionaryOperator.uniform_crossover(parent1, parent2)

            # Mutation
            child1 = EvolutionaryOperator.evolve_individual(child1, mutation_rate)
            child2 = EvolutionaryOperator.evolve_individual(child2, mutation_rate)

            # Add children to new population (alternating)
            if len(new_population) < pop_size:
                new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        return new_population[:pop_size]
    
    def local_refinement(self, individual: List[float]) -> List[float]:
        """Perform local refinement on a solution."""
        current_individual = individual.copy()
        current_c2 = ObjectiveFunction.calculate_c2(current_individual)
        
        for _ in range(LOCAL_REFINEMENT_ITERATIONS):
            if self.check_time_limit():
                break
                
            # Try small random changes
            idx = np.random.randint(len(current_individual))
            old_val = current_individual[idx]

            # Try small changes
            new_val = max(0, old_val + np.random.normal(0, 0.1))
            current_individual[idx] = new_val

            # Test if this improves the solution
            test_c2 = ObjectiveFunction.calculate_c2(current_individual)
            if test_c2 > current_c2:
                current_c2 = test_c2
            else:
                current_individual[idx] = old_val  # Revert if worse
        
        return current_individual
    
    def optimize(self, size: int) -> List[float]:
        """Run the evolutionary optimization process."""
        # Initialize population
        population = self.generate_initial_population(size, POPULATION_SIZE)

        # Evolve
        for gen in range(GENERATIONS):
            if self.check_time_limit():
                break
                
            # Evaluate fitness
            fitness_scores = self.evaluate_population(population)

            # Track best
            max_fitness = max(fitness_scores)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                best_individual_idx = fitness_scores.index(max_fitness)
                self.best_individual = population[best_individual_idx].copy()

            # Print progress every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: Best C2 = {self.best_fitness:.6f}")

            # Evolve
            population = self.evolve_generation(population, fitness_scores, gen)

            # Store fitness history
            self.fitness_history.append(max_fitness)

        # Final local refinement
        if self.best_individual is not None:
            refined_solution = self.local_refinement(self.best_individual.tolist())
            final_c2 = ObjectiveFunction.calculate_c2(refined_solution)
            if final_c2 > self.best_fitness:
                self.best_fitness = final_c2
                self.best_individual = np.array(refined_solution)
                
        return self.best_individual.tolist() if self.best_individual is not None else [1.0] * size

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value using modular evolutionary optimization.

    Returns:
        List of step heights that maximize C2
    """
    # Create optimizer instance
    optimizer = EvolutionaryOptimizer()
    
    # Try different sizes for better results with careful time management
    sizes_to_try = [1000, 1250, 1500]  # Focus on medium-to-large sizes
    best_c2 = -float('inf')
    best_solution = None

    for size in sizes_to_try:
        if optimizer.check_time_limit():
            break
            
        try:
            # Use evolutionary optimization for global search
            evol_solution = optimizer.optimize(size)

            # Evaluate final solution
            c2_value = ObjectiveFunction.calculate_c2(evol_solution)
            print(f"Size {size}: C2 = {c2_value:.6f}")

            if c2_value > best_c2:
                best_c2 = c2_value
                best_solution = evol_solution
                
        except Exception as e:
            print(f"Failed at size {size}: {e}")
            continue

    return best_solution if best_solution is not None else [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")