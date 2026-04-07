# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
from joblib import Parallel, delayed
from typing import List, Tuple, Optional
import math

# Global constants
SEED = 42
TIME_LIMIT = 85  # seconds
MAX_STEPS = 50000
MIN_STEPS = 100
BASE_POP_SIZE = 50
MAX_GENERATIONS = 200
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.5
ELITISM_COUNT = 5
TOURNAMENT_SIZE = 3

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

class C2Optimizer:
    def __init__(self):
        self.best_individual = None
        self.best_fitness = 0.0
        self.start_time = 0.0
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_autoconvolution_parallel(f_vals):
        """Compute autoconvolution using parallel numba for speed"""
        n = len(f_vals)
        # Autoconvolution size is 2*n - 1
        g_size = 2 * n - 1
        g_vals = np.zeros(g_size, dtype=np.float64)

        # Compute convolution directly with parallel numba optimization
        for i in prange(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_size:
                    g_vals[idx] += f_vals[i] * f_vals[j]

        return g_vals
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms_fast(g_vals):
        """
        Compute L1, L2², and L∞ norms efficiently with numba
        """
        n = len(g_vals)
        
        # Initialize accumulators
        l1_norm = 0.0
        l2_norm_sq = 0.0
        linf_norm = 0.0
        
        # Single pass through array
        for i in range(n):
            val = g_vals[i]
            abs_val = abs(val)
            
            # Accumulate L1 norm
            l1_norm += abs_val
            
            # Accumulate L2² norm
            l2_norm_sq += val * val
            
            # Update infinity norm
            if abs_val > linf_norm:
                linf_norm = abs_val
                
        return l1_norm, l2_norm_sq, linf_norm
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C₂ calculation:
        ||g||₂², ||g||₁, ||g||∞ where g = f*f
        """
        if not f_values:
            return 0.0, 0.0, 0.0

        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        n_steps = len(f)
        
        if n_steps == 0:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4] with proper spacing
        step_width = 0.5 / n_steps
        x_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps, dtype=np.float64)
        
        # Convert to piecewise constant function on fine grid for convolution
        fine_grid_points = 1000
        x_fine = np.linspace(-0.25, 0.25, fine_grid_points, dtype=np.float64)
        dx = x_fine[1] - x_fine[0]
        
        # Build piecewise constant function using broadcasting
        f_func = np.zeros_like(x_fine, dtype=np.float64)
        for i in range(n_steps):
            pos = x_positions[i]
            height = f[i]
            left = pos - step_width/2
            right = pos + step_width/2
            mask = (x_fine >= left) & (x_fine <= right)
            f_func[mask] = height
        
        # Compute autoconvolution using numpy (more reliable than manual loops)
        g = np.convolve(f_func, f_func, mode='full')
        g = g[:len(g)//2 + 1]  # Take only first half (symmetric)
        
        # Scale appropriately for the discretization
        g = g * dx
        
        # Extract central portion that corresponds to our domain
        # This is an approximation - we want to extract the convolution values
        # that correspond to the valid convolution region
        g_centered = g
        
        # Compute norms using numba optimized version
        g_abs = np.abs(g_centered)
        
        # Compute norms
        norm_1, norm_2_sq, norm_inf = C2Optimizer.compute_norms_fast(g_abs)
        
        return norm_2_sq, norm_1, norm_inf
    
    @staticmethod
    def evaluate_c2_single(individual: List[float]) -> float:
        """Evaluate fitness for a single individual"""
        try:
            # Ensure non-negative values
            f_values = [max(0.0, float(x)) for x in individual]
            
            # Compute the norms
            norm_2_sq, norm_1, norm_inf = C2Optimizer.compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0

            # Calculate C₂
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2
        except Exception:
            return 0.0
    
    @classmethod
    def evaluate_fitness_parallel(cls, population: List[List[float]], n_jobs: int = -1) -> List[float]:
        """Evaluate fitness for entire population in parallel"""
        return Parallel(n_jobs=n_jobs)(
            delayed(cls.evaluate_c2_single)(ind) for ind in population
        )
    
    @classmethod
    def initialize_population(cls, pop_size: int, min_steps: int, max_steps: int) -> List[List[float]]:
        """Initialize population with diverse step functions"""
        population = []
        for _ in range(pop_size):
            # Random number of steps
            n_steps = np.random.randint(min_steps, max_steps)
            # Generate heights using log-normal distribution for better diversity
            heights = np.random.lognormal(0, 0.5, size=n_steps)
            # Ensure non-negative
            heights = np.maximum(heights, 0)
            population.append(heights.tolist())
        return population
    
    @staticmethod
    def tournament_selection(population: List[List[float]], fitness_scores: List[float], 
                            tournament_size: int) -> List[List[float]]:
        """Tournament selection for choosing parents"""
        selected = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected
    
    @staticmethod
    def crossover(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Uniform crossover between two parents"""
        if len(parent1) != len(parent2):
            # Make them same length by truncating or padding
            min_len = min(len(parent1), len(parent2))
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]

        if np.random.random() < CROSSOVER_RATE:
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
    def mutate(individual: List[float], mutation_rate: float) -> List[float]:
        """Mutate individual with log-normal noise"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Add log-normal noise to preserve positivity
                noise_factor = np.random.lognormal(0, 0.3)
                mutated[i] = max(0, mutated[i] * noise_factor)
        return mutated
    
    @staticmethod
    def elitism(population: List[List[float]], fitness_scores: List[float], 
               elite_count: int) -> List[List[float]]:
        """Keep best individuals"""
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i].copy() for i in sorted_indices[:elite_count]]
        return elite
    
    def run_evolution(self, initial_pop: List[List[float]] = None) -> List[float]:
        """Run the main evolutionary optimization process"""
        self.start_time = time.time()
        
        # Initialize population
        if initial_pop is None:
            population = self.initialize_population(BASE_POP_SIZE, MIN_STEPS, MAX_STEPS)
        else:
            population = initial_pop
            
        # Evaluate initial population
        fitness_scores = self.evaluate_fitness_parallel(population)
        
        # Track best solution
        best_gen_index = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_gen_index]
        self.best_individual = population[best_gen_index].copy()
        
        # Evolution parameters
        generation = 0
        stall_count = 0
        max_stall = 20
        
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
            while len(new_population) < BASE_POP_SIZE:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = self.crossover(parents[p1], parents[p2])
                
                child1 = self.mutate(child1, MUTATION_RATE)
                child2 = self.mutate(child2, MUTATION_RATE)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:BASE_POP_SIZE]
            
            # Evaluate new population
            fitness_scores = self.evaluate_fitness_parallel(population)
            
            # Update best solution
            best_gen_index = np.argmax(fitness_scores)
            if fitness_scores[best_gen_index] > self.best_fitness:
                self.best_fitness = fitness_scores[best_gen_index]
                self.best_individual = population[best_gen_index].copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Adaptive stopping condition
            if stall_count > max_stall:
                # Try to restart with better initialization if stagnation occurs
                population = self.initialize_population(BASE_POP_SIZE, MIN_STEPS, MAX_STEPS)
                fitness_scores = self.evaluate_fitness_parallel(population)
                best_gen_index = np.argmax(fitness_scores)
                self.best_fitness = fitness_scores[best_gen_index]
                self.best_individual = population[best_gen_index].copy()
                stall_count = 0
                
        return self.best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using evolutionary optimization."""
    
    optimizer = C2Optimizer()
    
    # Run main evolution process
    try:
        result = optimizer.run_evolution()
        return [float(x) for x in result]
    except Exception as e:
        # Fallback to simple random initialization
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        heights = np.random.lognormal(0, 0.5, size=n_steps)
        heights = np.maximum(heights, 0)
        return heights.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")