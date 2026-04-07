# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from numba import jit, prange
import random
import time
from typing import List, Tuple, Optional
from joblib import Parallel, delayed

# Global constants
SEED = 42
TIME_LIMIT = 85  # seconds
MAX_STEPS = 50000
MIN_STEPS = 100
POPULATION_SIZE = 120
MAX_GENERATIONS = 100
INIT_STRATEGIES = 4
ELITISM_COUNT = 8
TOURNAMENT_SIZE = 4
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.05

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

class FastConvolutionCalculator:
    """High-performance convolution calculator using numba JIT compilation"""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_autoconvolution_fast(f_vals):
        """Fast autoconvolution computation using numba"""
        n = len(f_vals)
        g_size = 2 * n - 1
        g_vals = np.zeros(g_size, dtype=np.float64)

        for i in prange(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_size:
                    g_vals[idx] += f_vals[i] * f_vals[j]

        return g_vals

    @staticmethod
    @jit(nopython=True)
    def compute_norms_fast(g_vals, dx):
        """Fast computation of L1, L2², and L∞ norms"""
        n = len(g_vals)

        l1_norm = 0.0
        l2_norm_sq = 0.0
        linf_norm = 0.0

        for i in range(n):
            val = g_vals[i]
            abs_val = abs(val)

            l1_norm += abs_val
            l2_norm_sq += val * val

            if abs_val > linf_norm:
                linf_norm = abs_val

        return l1_norm, l2_norm_sq, linf_norm

class StepFunctionEvaluator:
    """Handles all evaluation logic for step functions"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Optimized autoconvolution norm computation using FFT-based convolution
        """
        if not f_values or len(f_values) == 0:
            return 0.0, 0.0, 0.0

        f = np.array(f_values, dtype=np.float64)
        n_steps = len(f)

        if n_steps == 0:
            return 0.0, 0.0, 0.0

        # Create step function with proper spacing for convolution
        step_width = 0.5 / n_steps
        step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps, dtype=np.float64)

        # Create fine grid for better convolution accuracy
        fine_grid_points = max(1000, n_steps * 3)
        x_fine = np.linspace(-0.25, 0.25, fine_grid_points, dtype=np.float64)
        dx = x_fine[1] - x_fine[0]

        # Build piecewise constant function using vectorized operations
        f_func = np.zeros_like(x_fine, dtype=np.float64)
        for i in range(n_steps):
            pos = step_positions[i]
            height = f[i]
            left = pos - step_width/2
            right = pos + step_width/2
            mask = (x_fine >= left) & (x_fine <= right)
            f_func[mask] = height

        # Compute autoconvolution using FFT-based convolution
        g = signal.fftconvolve(f_func, f_func, mode='full')
        g = g[:len(g)//2 + 1]

        # Apply proper scaling
        g = g * dx

        # Compute norms using fast numba methods
        g_abs = np.abs(g)
        norm_1, norm_2_sq, norm_inf = FastConvolutionCalculator.compute_norms_fast(g_abs, dx)

        return norm_2_sq, norm_1, norm_inf

    @staticmethod
    def evaluate_c2_single(individual: List[float]) -> float:
        """Fast evaluation of C2 for a single individual"""
        try:
            # Ensure non-negative values
            f_values = [max(0.0, float(x)) for x in individual]

            # Compute the norms
            norm_2_sq, norm_1, norm_inf = StepFunctionEvaluator.compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 0.0

            # Calculate C₂
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2
        except Exception:
            return 0.0

class MultiScaleInitializer:
    """Generates diverse initial solutions using multiple strategies"""
    
    @staticmethod
    def generate_multiscale_gaussian_function(n_steps):
        """Generate a step function with multi-scale Gaussian bumps"""
        step_width = 0.5 / n_steps
        step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)

        # Generate multi-scale Gaussian components
        heights = np.zeros(n_steps)
        scales = [0.15, 0.1, 0.05, 0.025]
        amplitudes = [1.0, 0.7, 0.5, 0.3]

        for scale, amp in zip(scales, amplitudes):
            bump_centers = np.linspace(-0.2, 0.2, 7)
            for center in bump_centers:
                gaussian = amp * np.exp(-0.5 * ((step_positions - center) / scale)**2)
                heights += gaussian

        # Add additional variation for diversity
        noise_factor = 0.15
        heights += np.random.normal(0, noise_factor * np.mean(heights), n_steps)
        heights = np.maximum(heights, 0)

        return heights.tolist()

    @staticmethod
    def generate_exponential_distribution(n_steps):
        """Generate exponential distribution step function"""
        heights = np.random.exponential(scale=1.0, size=n_steps)
        heights = np.maximum(heights, 0)
        return heights.tolist()

    @staticmethod
    def generate_lognormal_distribution(n_steps):
        """Generate log-normal distribution step function"""
        heights = np.random.lognormal(0, 0.5, size=n_steps)
        heights = np.maximum(heights, 0)
        return heights.tolist()

    @staticmethod
    def generate_structured_pattern(n_steps):
        """Generate structured pattern with multiple components"""
        heights = np.zeros(n_steps)

        # Central peak
        center = n_steps // 2
        peak_height = np.random.exponential(1.0)
        spread = max(1, n_steps // 10)
        for i in range(n_steps):
            heights[i] += peak_height * np.exp(-((i - center) ** 2) / (2 * spread ** 2))

        # Exponential decay on sides
        for i in range(n_steps // 3):
            heights[i] += 0.5 * np.exp(-i / (n_steps // 10))
        for i in range(2 * n_steps // 3, n_steps):
            heights[i] += 0.5 * np.exp(-(n_steps - i) / (n_steps // 10))

        heights = np.maximum(heights, 0)
        return heights.tolist()

    @staticmethod
    def initialize_population(pop_size: int, min_steps: int, max_steps: int) -> List[List[float]]:
        """Generate diverse population with multiple initialization strategies"""
        population = []

        # Strategy 1: Multi-scale Gaussian pattern
        for _ in range(pop_size // 4):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = MultiScaleInitializer.generate_multiscale_gaussian_function(n_steps)
            population.append(heights)

        # Strategy 2: Exponential distribution
        for _ in range(pop_size // 4):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = MultiScaleInitializer.generate_exponential_distribution(n_steps)
            population.append(heights)

        # Strategy 3: Log-normal distribution
        for _ in range(pop_size // 4):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = MultiScaleInitializer.generate_lognormal_distribution(n_steps)
            population.append(heights)

        # Strategy 4: Structured pattern
        for _ in range(pop_size // 4):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = MultiScaleInitializer.generate_structured_pattern(n_steps)
            population.append(heights)

        # Ensure we have exactly pop_size individuals
        if len(population) < pop_size:
            for i in range(pop_size - len(population)):
                n_steps = np.random.randint(min_steps, max_steps)
                heights = np.random.exponential(scale=1.0, size=n_steps)
                heights = np.maximum(heights, 0)
                population.append(heights.tolist())

        return population[:pop_size]

class AdvancedEvolutionaryOptimizer:
    """Advanced evolutionary optimization engine"""
    
    def __init__(self):
        self.best_individual = None
        self.best_fitness = -np.inf
        self.start_time = 0.0

    @staticmethod
    def tournament_selection(population: List[List[float]], fitness_scores: List[float],
                            tournament_size: int) -> List[List[float]]:
        """Efficient tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected

    @staticmethod
    def crossover(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Uniform crossover between two parents"""
        if len(parent1) != len(parent2):
            min_len = min(len(parent1), len(parent2))
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]

        child1, child2 = [], []
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2

    @staticmethod
    def mutate(individual: List[float], mutation_rate: float, generation: int = None, 
               total_generations: int = None) -> List[float]:
        """Mutate individual with adaptive noise"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                noise_scale = 0.1 * mutated[i] + 0.01
                noise = np.random.normal(0, noise_scale)
                mutated[i] = max(0, mutated[i] + noise)
        return mutated

    @staticmethod
    def elitism(population: List[List[float]], fitness_scores: List[float],
               elite_count: int) -> List[List[float]]:
        """Keep best individuals"""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i].copy() for i in sorted_indices[:elite_count]]
        return elite

    @staticmethod
    def adaptive_local_refinement(individual: List[float], max_iterations: int = 10) -> List[float]:
        """Perform aggressive local refinement"""
        current_function = individual.copy()
        current_c2 = StepFunctionEvaluator.evaluate_c2_single(current_function)
        
        for iteration in range(max_iterations):
            # Create perturbed version
            candidate_function = current_function.copy()
            n_changes = max(1, len(candidate_function) // 10)  # Small number of changes
            indices_to_modify = np.random.choice(len(candidate_function), n_changes, replace=False)
            
            for idx in indices_to_modify:
                # Adaptive perturbation strength
                current_value = candidate_function[idx]
                if current_value > 0:
                    noise_scale = 0.1 * current_value + 0.01
                    noise = np.random.normal(0, noise_scale)
                    new_value = max(0, current_value + noise)
                    candidate_function[idx] = new_value
                    
            # Evaluate the candidate
            candidate_c2 = StepFunctionEvaluator.evaluate_c2_single(candidate_function)
            
            # Accept if better or sometimes accept worse (simulated annealing component)
            if candidate_c2 > current_c2 or (iteration < max_iterations//2 and np.random.rand() < 0.1):
                current_function = candidate_function.copy()
                current_c2 = candidate_c2
                
        return current_function

    def run_evolution(self, initial_pop: List[List[float]] = None) -> List[float]:
        """Run the main evolutionary optimization process"""
        self.start_time = time.time()

        # Initialize population
        if initial_pop is None:
            population = MultiScaleInitializer.initialize_population(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)
        else:
            population = initial_pop

        # Evaluate initial population
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(StepFunctionEvaluator.evaluate_c2_single)(ind) for ind in population
        )

        # Track best solution
        best_gen_index = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_gen_index]
        self.best_individual = population[best_gen_index].copy()

        # Evolution parameters
        generation = 0
        mutation_rate = MUTATION_RATE_START
        stall_count = 0
        max_stall = 15

        # Main evolution loop
        while generation < MAX_GENERATIONS:
            if time.time() - self.start_time > TIME_LIMIT:
                break

            generation += 1

            # Elitism
            elite = self.elitism(population, fitness_scores, ELITISM_COUNT)

            # Selection
            parents = self.tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)

            # Crossover and mutation
            new_population = elite.copy()
            while len(new_population) < POPULATION_SIZE:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.crossover(parents[p1], parents[p2])

                child1 = self.mutate(child1, mutation_rate, generation, MAX_GENERATIONS)
                child2 = self.mutate(child2, mutation_rate, generation, MAX_GENERATIONS)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:POPULATION_SIZE]

            # Evaluate new population
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(StepFunctionEvaluator.evaluate_c2_single)(ind) for ind in population
            )

            # Update best solution
            best_gen_index = np.argmax(fitness_scores)
            if fitness_scores[best_gen_index] > self.best_fitness:
                self.best_fitness = fitness_scores[best_gen_index]
                self.best_individual = population[best_gen_index].copy()
                stall_count = 0
                mutation_rate = min(0.3, mutation_rate * 1.03)
            else:
                stall_count += 1
                mutation_rate = max(0.05, mutation_rate * 0.97)

            # Adaptive stopping condition
            if stall_count > max_stall:
                # Restart with better initialization
                population = MultiScaleInitializer.initialize_population(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)
                fitness_scores = Parallel(n_jobs=-1)(
                    delayed(StepFunctionEvaluator.evaluate_c2_single)(ind) for ind in population
                )
                best_gen_index = np.argmax(fitness_scores)
                self.best_fitness = fitness_scores[best_gen_index]
                self.best_individual = population[best_gen_index].copy()
                stall_count = 0
                mutation_rate = MUTATION_RATE_START

        # Final refinement
        final_candidate = self.adaptive_local_refinement(self.best_individual, 5)
        final_fitness = StepFunctionEvaluator.evaluate_c2_single(final_candidate)
        
        if final_fitness > self.best_fitness:
            self.best_fitness = final_fitness
            self.best_individual = final_candidate

        return self.best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using hybrid approach."""
    start_time = time.time()

    best_result = None
    best_fitness = -np.inf

    # Try multiple initialization strategies
    for strategy in range(INIT_STRATEGIES):
        # Set different seed for each try
        np.random.seed(SEED + strategy * 100)
        random.seed(SEED + strategy * 100)

        try:
            optimizer = AdvancedEvolutionaryOptimizer()
            result = optimizer.run_evolution()
            fitness = optimizer.best_fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_result = result
        except Exception as e:
            continue

    # If no good result was found, return a fallback
    if best_result is None:
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        heights = np.random.lognormal(0, 0.5, size=n_steps)
        heights = np.maximum(heights, 0)
        best_result = heights.tolist()

    end_time = time.time()
    eval_time = end_time - start_time

    print(f"Evaluated in {eval_time:.2f} seconds")
    print(f"Best C2 found: {StepFunctionEvaluator.evaluate_c2_single(best_result):.6f}")

    return [float(x) for x in best_result]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")