# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time
from typing import List, Tuple, Optional

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutoconvolutionEvaluator:
    """Handles all autoconvolution and norm computations"""

    @staticmethod
    @numba.jit(nopython=True)
    def compute_autoconvolution(f_vals):
        """Compute autoconvolution efficiently using numba"""
        n = len(f_vals)
        # Create output array for autoconvolution
        g = np.zeros(2*n - 1)

        # Compute convolution manually with numba optimization
        for i in range(n):
            for j in range(n):
                g[i + j] += f_vals[i] * f_vals[j]

        return g

    @staticmethod
    @numba.jit(nopython=True)
    def compute_norms(g_vals):
        """Compute norms efficiently with numba"""
        n = len(g_vals)

        # L2 norm squared (using trapezoidal-like scheme)
        l2_sq = 0.0
        for i in range(n - 1):
            y1 = g_vals[i]
            y2 = g_vals[i + 1]
            l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

        # L1 norm
        l1 = 0.0
        for i in range(n):
            l1 += abs(g_vals[i])

        # L-infinity norm
        linf = 0.0
        for i in range(n):
            abs_val = abs(g_vals[i])
            if abs_val > linf:
                linf = abs_val

        return l2_sq, l1, linf

class StepFunctionBuilder:
    """Creates structured step functions for optimization"""

    @staticmethod
    def adaptive_gaussian_construction(n_steps: int = None) -> List[float]:
        """Create a structured step function with Gaussian peaks for better C2"""
        if n_steps is None:
            n_steps = random.randint(200, 1000)

        # Start with a base structure of Gaussian peaks
        f_vals = np.zeros(n_steps)

        # Determine number of peaks based on function length
        n_peaks = max(2, min(10, n_steps // 100))

        # Place peaks strategically with minimum gap enforcement
        peak_positions = []
        peak_widths = []
        peak_heights = []

        # Generate peak parameters
        for i in range(n_peaks):
            # Ensure minimum spacing between peaks
            if i == 0:
                # First peak at beginning
                center = random.uniform(0.1 * n_steps, 0.3 * n_steps)
            elif i == n_peaks - 1:
                # Last peak at end
                center = random.uniform(0.7 * n_steps, 0.9 * n_steps)
            else:
                # Middle peaks with spacing consideration
                if len(peak_positions) > 0:
                    prev_center = peak_positions[-1]
                    min_gap = max(20, n_steps // 20)
                    center = random.uniform(prev_center + min_gap, n_steps - min_gap)
                else:
                    center = random.uniform(0.3 * n_steps, 0.7 * n_steps)

            peak_positions.append(center)
            # Width inversely related to height for better control
            width = random.uniform(10, 30)
            peak_widths.append(width)
            # Height inversely proportional to width to maintain balance
            height = random.uniform(0.5, 2.0)
            peak_heights.append(height)

        # Create Gaussian curves for each peak
        for center, width, height in zip(peak_positions, peak_widths, peak_heights):
            x = np.arange(n_steps)
            gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
            f_vals += gaussian

        # Apply smoothing to reduce extreme variations
        if n_steps > 50:
            # Use Savitzky-Golay filter for better preservation of shape
            f_vals = signal.savgol_filter(f_vals, min(51, n_steps-1), 3)

        # Ensure non-negativity
        f_vals = np.maximum(f_vals, 0)

        # Normalize to reasonable range
        if np.max(f_vals) > 0:
            f_vals = f_vals / np.max(f_vals) * 2.0

        # Apply constraint-aware normalization to prevent extreme autoconvolution spikes
        # This helps avoid numerical instability in later processing
        max_allowed = np.percentile(f_vals, 90) if len(f_vals) > 10 else 1.0
        if max_allowed > 0:
            f_vals = np.minimum(f_vals, max_allowed * 2.0)

        return f_vals.tolist()

    @staticmethod
    def gamma_distribution_construction(n_steps: int = 500) -> List[float]:
        """Construct function using gamma distribution with smoothing"""
        f_values = np.random.gamma(2, 2, n_steps)  # Gamma distribution gives positive values
        f_values = f_values / np.max(f_values) * 2  # Scale to reasonable range
        f_values = np.maximum(f_values, 0)

        # Apply some smoothing to reduce extreme variations
        f_values = signal.savgol_filter(f_values, min(51, len(f_values)-1), 3) if len(f_values) > 50 else f_values
        f_values = np.maximum(f_values, 0)

        return f_values.tolist()

class EvolutionaryOptimizer:
    """Manages the evolutionary optimization process"""

    def __init__(self):
        self.toolbox = None
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Initialize DEAP toolbox with custom operators"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self, size: int = None):
        """Create a structured individual using Gaussian construction"""
        return StepFunctionBuilder.adaptive_gaussian_construction(size)

    def _mutate_individual(self, individual):
        """Enhanced mutation with adaptive scaling"""
        for i in range(len(individual)):
            if random.random() < 0.1:  # 10% mutation rate
                # Use adaptive mutation based on current value
                current_value = individual[i]
                if current_value > 0:
                    # Scale mutation based on value magnitude
                    mutation_scale = 0.1 * current_value
                    individual[i] = max(0, individual[i] + random.gauss(0, mutation_scale))
                else:
                    individual[i] = max(0, individual[i] + random.gauss(0, 0.1))
        return individual

    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual (step function)"""
        try:
            # Convert to numpy array and ensure non-negative
            f_vals = np.array(individual, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)

            # Skip if all zeros
            if np.sum(f_vals) == 0:
                return (0.0,)

            # Compute autoconvolution
            g_vals = AutoconvolutionEvaluator.compute_autoconvolution(f_vals)

            # Compute norms
            l2_sq, l1, linf = AutoconvolutionEvaluator.compute_norms(g_vals)

            # Avoid division by zero
            if l1 <= 1e-15 or linf <= 1e-15:
                return (0.0,)

            # Compute C2
            c2 = l2_sq / (l1 * linf)
            return (c2,)
        except:
            return (0.0,)

    def run_evolution(self, pop_size: int = 50, n_generations: int = 100) -> Optional[List[float]]:
        """Run evolutionary optimization"""
        # Initialize population with more structured individuals
        population = self.toolbox.population(n=pop_size)

        # Evolve
        best_individual = None
        best_fitness = 0

        for generation in range(n_generations):
            # Evaluate population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Track best
            for ind in population:
                if ind.fitness.values[0] > best_fitness and len(ind) > 0:
                    best_fitness = ind.fitness.values[0]
                    best_individual = list(ind)

            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace old population
            population[:] = offspring

        return best_individual if best_individual is not None else None

class MainOptimizer:
    """Main orchestrator class for function construction"""

    def __init__(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        self.evaluator = AutoconvolutionEvaluator()
        self.builder = StepFunctionBuilder()

    def construct_function(self) -> List[float]:
        """Function to construct step-function with high C2 value."""
        start_time = time.time()

        # Try multiple approaches to find good solution
        best_result = []
        best_c2 = 0

        # Approach 1: Evolutionary algorithm
        try:
            evolved_result = self.evolutionary_optimizer.run_evolution()
            if evolved_result:
                # Evaluate evolved result
                f_vals = np.array(evolved_result, dtype=np.float64)
                f_vals = np.maximum(f_vals, 0.0)
                if np.sum(f_vals) > 0:
                    g_vals = self.evaluator.compute_autoconvolution(f_vals)
                    l2_sq, l1, linf = self.evaluator.compute_norms(g_vals)

                    if l1 > 1e-15 and linf > 1e-15:
                        c2 = l2_sq / (l1 * linf)
                        if c2 > best_c2:
                            best_c2 = c2
                            best_result = evolved_result
        except Exception as e:
            pass

        # If no good result from evolution, fallback to a more informed approach
        if len(best_result) == 0:
            # Use a heuristic approach with more structured sampling
            best_result = self.builder.gamma_distribution_construction()

        # Limit execution time
        elapsed = time.time() - start_time
        if elapsed > 85:  # Leave buffer for cleanup
            return best_result[:1000]  # Truncate if needed

        return best_result

# Global instance for the main function
_main_optimizer = MainOptimizer()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    return _main_optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")