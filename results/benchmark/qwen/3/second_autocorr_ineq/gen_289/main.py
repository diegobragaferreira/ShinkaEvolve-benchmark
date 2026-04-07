# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
from scipy import signal
import random
import time
from typing import List, Tuple, Optional
from joblib import Parallel, delayed

# Global constants
SEED = 42
TIME_LIMIT = 85  # seconds
MAX_STEPS = 50000
MIN_STEPS = 100
BASE_POP_SIZE = 100
MAX_GENERATIONS = 150
INIT_STRATEGIES = 5  # Number of different initializations to try
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 5

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

class FastNormCalculator:
    """Highly optimized norm calculator using numba JIT compilation"""

    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_fast(f_vals):
        """Fast autoconvolution computation using numba - optimized for step functions"""
        n = len(f_vals)
        # Autoconvolution size is 2*n - 1
        g_size = 2 * n - 1
        g_vals = np.zeros(g_size, dtype=np.float64)

        # Direct computation without parallelization for better cache usage
        # and to avoid thread overhead for small arrays
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_size:
                    g_vals[idx] += f_vals[i] * f_vals[j]

        return g_vals

    @staticmethod
    @jit(nopython=True)
    def compute_norms_fast(g_vals):
        """Fast computation of L1, L2², and L∞ norms using proper piecewise linear integration"""
        n = len(g_vals)

        l1_norm = 0.0
        l2_norm_sq = 0.0
        linf_norm = 0.0

        # For proper L2 computation using trapezoidal-like integration
        # Use the formula (dx/3)(y1² + y1*y2 + y2²) for adjacent pairs
        if n >= 2:
            # Assuming unit step size, so dx = 1 for our normalized case
            for i in range(n - 1):
                y1 = g_vals[i]
                y2 = g_vals[i+1]
                # Trapezoidal-like integration for L2 norm squared
                l2_norm_sq += (1.0/3.0) * (y1*y1 + y1*y2 + y2*y2)

        # For L1 norm, sum the absolute values
        for i in range(n):
            abs_val = abs(g_vals[i])
            l1_norm += abs_val
            if abs_val > linf_norm:
                linf_norm = abs_val

        return l1_norm, l2_norm_sq, linf_norm

class StepFunctionOptimizer:
    """Core optimizer for step function construction"""

    def __init__(self):
        self.best_individual = None
        self.best_fitness = 0.0
        self.start_time = 0.0

    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Optimized autoconvolution norm computation using fast methods
        """
        if not f_values or len(f_values) == 0:
            return 0.0, 0.0, 0.0

        # Convert to numpy array
        f = np.array(f_values, dtype=np.float64)
        n_steps = len(f)

        if n_steps == 0:
            return 0.0, 0.0, 0.0

        # Create step function with proper spacing
        step_width = 0.5 / n_steps
        step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps, dtype=np.float64)

        # Create fine grid for convolution
        fine_grid_points = 1000
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

        # Compute autoconvolution using optimized scipy
        g = signal.convolve(f_func, f_func, mode='full')
        g = g[:len(g)//2 + 1]  # Take first half (symmetric)

        # Apply proper scaling
        g = g * dx

        # Compute norms using fast numba methods
        g_abs = np.abs(g)
        norm_1, norm_2_sq, norm_inf = FastNormCalculator.compute_norms_fast(g_abs)

        return norm_2_sq, norm_1, norm_inf

    @staticmethod
    def evaluate_c2_single(individual: List[float]) -> float:
        """Fast evaluation of C2 for a single individual"""
        try:
            # Ensure non-negative values
            f_values = [max(0.0, float(x)) for x in individual]

            # Compute the norms
            norm_2_sq, norm_1, norm_inf = StepFunctionOptimizer.compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_1 <= 1e-12 or norm_inf <= 1e-12:
                return 0.0

            # Calculate C₂
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return c2
        except Exception:
            return 0.0

    @staticmethod
    def generate_multiscale_gaussian_initial_function(n_steps: int) -> List[float]:
        """Generate an initial function with multi-scale Gaussian pattern construction.
        This creates a structured pattern that combines multiple Gaussian bumps with
        varying scales and positions for better convolution behavior."""
        heights = np.zeros(n_steps)

        # Define multiple scales for Gaussian bumps
        scales = [n_steps // 20, n_steps // 15, n_steps // 10, n_steps // 5]

        # Create bumps at different scales
        for scale in scales:
            if scale < 2:  # Skip very small scales
                continue

            # Create a sufficient number of bumps for this scale
            num_bumps = max(1, n_steps // (scale * 2))

            for _ in range(num_bumps):
                # Position the bump randomly (avoiding edges)
                peak_pos = np.random.randint(scale, n_steps - scale)
                peak_height = np.random.exponential(1.0)

                # Create Gaussian bump with this scale
                x = np.arange(n_steps)
                gaussian = peak_height * np.exp(-((x - peak_pos) ** 2) / (2 * scale ** 2))
                heights += gaussian

        # Ensure non-negativity
        heights = np.maximum(heights, 0)

        # Normalize to prevent extreme values
        total = np.sum(heights)
        if total > 0:
            heights = heights / total * 10

        return heights.tolist()

    @staticmethod
    def initialize_population(pop_size: int, min_steps: int, max_steps: int) -> List[List[float]]:
        """Generate diverse population with multiple strategies"""
        population = []

        # Strategy 1: Random exponential distribution
        for _ in range(pop_size // 5):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = np.random.exponential(scale=1.0, size=n_steps)
            heights = np.maximum(heights, 0)
            population.append(heights.tolist())

        # Strategy 2: Log-normal distribution
        for _ in range(pop_size // 5):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = np.random.lognormal(0, 0.5, size=n_steps)
            heights = np.maximum(heights, 0)
            population.append(heights.tolist())

        # Strategy 3: Gaussian bumps pattern
        for _ in range(pop_size // 5):
            n_steps = np.random.randint(min_steps, max_steps)
            # Create some structured pattern with random variations
            heights = np.zeros(n_steps)
            # Add some random peaks
            n_peaks = max(1, n_steps // 20)
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, n_steps)
                peak_height = np.random.exponential(1.0)
                # Spread the peak over a few bins
                spread = max(1, n_steps // 50)
                for i in range(max(0, peak_pos - spread), min(n_steps, peak_pos + spread)):
                    heights[i] += peak_height * np.exp(-((i - peak_pos) ** 2) / (2 * spread ** 2))
            heights = np.maximum(heights, 0)
            population.append(heights.tolist())

        # Strategy 4: Structured exponential + periodic pattern (more advanced)
        for _ in range(pop_size // 5):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = np.zeros(n_steps)

            # Create a pattern that decays exponentially from center with periodic modulation
            center = n_steps // 2
            decay_factor = 4.0 / n_steps

            for i in range(n_steps):
                # Exponential decay from center
                distance_from_center = abs(i - center)
                exp_decay = np.exp(-decay_factor * distance_from_center)

                # Add periodic modulation to avoid regular patterns
                period_mod = 1.0 + 0.3 * np.sin(2 * np.pi * i / max(1, n_steps // 8))

                # Add some random component
                random_component = 0.5 + 0.5 * np.random.random()

                heights[i] = exp_decay * period_mod * random_component

            heights = np.maximum(heights, 0)
            population.append(heights.tolist())

        # Strategy 5: Multi-scale Gaussian pattern
        for _ in range(pop_size // 5):
            n_steps = np.random.randint(min_steps, max_steps)
            heights = StepFunctionOptimizer.generate_multiscale_gaussian_initial_function(n_steps)
            population.append(heights)

        # Ensure we have exactly pop_size individuals
        if len(population) < pop_size:
            # Fill with random ones
            for i in range(pop_size - len(population)):
                n_steps = np.random.randint(min_steps, max_steps)
                heights = np.random.exponential(scale=1.0, size=n_steps)
                heights = np.maximum(heights, 0)
                population.append(heights.tolist())

        return population[:pop_size]

    @staticmethod
    def tournament_selection(population: List[List[float]], fitness_scores: List[float],
                            tournament_size: int) -> List[List[float]]:
        """Efficient tournament selection"""
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

    @staticmethod
    def mutate(individual: List[float], mutation_rate: float) -> List[float]:
        """Mutate individual with adaptive noise"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise with adaptive scale
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

    def run_evolution(self, initial_pop: List[List[float]] = None) -> List[float]:
        """Run the main evolutionary optimization process with adaptive parameters"""
        self.start_time = time.time()

        # Initialize population
        if initial_pop is None:
            population = self.initialize_population(BASE_POP_SIZE, MIN_STEPS, MAX_STEPS)
        else:
            population = initial_pop

        # Evaluate initial population
        fitness_scores = []
        for ind in population:
            fitness_scores.append(self.evaluate_c2_single(ind))

        # Track best solution
        best_gen_index = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_gen_index]
        self.best_individual = population[best_gen_index].copy()

        # Evolution parameters
        generation = 0
        # Adaptive parameters
        mutation_rate = 0.15
        crossover_rate = 0.8
        stall_count = 0
        max_stall = 15
        improvement_window = 5
        recent_improvements = []

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

                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:BASE_POP_SIZE]

            # Evaluate new population
            fitness_scores = []
            for ind in population:
                fitness_scores.append(self.evaluate_c2_single(ind))

            # Update best solution
            best_gen_index = np.argmax(fitness_scores)
            current_best = fitness_scores[best_gen_index]

            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_individual = population[best_gen_index].copy()
                stall_count = 0

                # Track recent improvements for adaptive tuning
                recent_improvements.append(current_best)
                if len(recent_improvements) > improvement_window:
                    recent_improvements.pop(0)

                # Increase mutation rate if we're consistently improving
                if len(recent_improvements) == improvement_window:
                    avg_improvement = (recent_improvements[-1] - recent_improvements[0]) / improvement_window
                    if avg_improvement > 0.0001:  # Significant improvement
                        mutation_rate = min(0.3, mutation_rate * 1.02)
            else:
                stall_count += 1
                # Decrease mutation rate if we're stuck
                mutation_rate = max(0.05, mutation_rate * 0.97)

            # More sophisticated adaptive stopping condition
            if stall_count > max_stall:
                # Check if recent improvement has plateaued
                if len(recent_improvements) >= 3:
                    improvement_rate = (recent_improvements[-1] - recent_improvements[0]) \
                                     / max(1, len(recent_improvements) - 1)
                    if improvement_rate < 0.0001:  # Very slow improvement
                        # Restart with better initialization
                        population = self.initialize_population(BASE_POP_SIZE, MIN_STEPS, MAX_STEPS)
                        fitness_scores = []
                        for ind in population:
                            fitness_scores.append(self.evaluate_c2_single(ind))
                        best_gen_index = np.argmax(fitness_scores)
                        self.best_fitness = fitness_scores[best_gen_index]
                        self.best_individual = population[best_gen_index].copy()
                        stall_count = 0
                        mutation_rate = 0.15
                        recent_improvements.clear()
                    else:
                        # Continue with normal adjustment
                        mutation_rate = max(0.05, mutation_rate * 0.95)
                else:
                    # Restart with better initialization
                    population = self.initialize_population(BASE_POP_SIZE, MIN_STEPS, MAX_STEPS)
                    fitness_scores = []
                    for ind in population:
                        fitness_scores.append(self.evaluate_c2_single(ind))
                    best_gen_index = np.argmax(fitness_scores)
                    self.best_fitness = fitness_scores[best_gen_index]
                    self.best_individual = population[best_gen_index].copy()
                    stall_count = 0
                    mutation_rate = 0.15
                    recent_improvements.clear()

        return self.best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using multi-start evolutionary optimization."""

    best_result = None
    best_fitness = -np.inf

    # Try multiple random initializations to improve chances of finding better optimum
    for i in range(INIT_STRATEGIES):
        # Set different seed for each try
        np.random.seed(SEED + i)
        random.seed(SEED + i)

        optimizer = StepFunctionOptimizer()

        try:
            result = optimizer.run_evolution()
            fitness = optimizer.best_fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_result = result
        except Exception as e:
            # If any error occurs, continue with next strategy
            continue

    # If no good result was found, return a fallback
    if best_result is None:
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        heights = np.random.lognormal(0, 0.5, size=n_steps)
        heights = np.maximum(heights, 0)
        best_result = heights.tolist()

    return [float(x) for x in best_result]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")