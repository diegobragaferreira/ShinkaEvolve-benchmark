# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import differential_evolution
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum

class OptimizationStrategy(Enum):
    EVOLUTIONARY = "evolutionary"
    LOCAL_REFINEMENT = "local_refinement"
    MULTI_SCALE = "multi_scale"

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    max_time_seconds: float = 90.0
    min_steps: int = 100
    max_steps: int = 1000
    population_size: int = 15
    max_iterations: int = 100
    seed: int = 42
    tolerance: float = 1e-6
    recombination_rate: float = 0.7

class AutoconvolutionCalculator:
    """Handles computation of autoconvolution and related norms"""

    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals: np.ndarray) -> np.ndarray:
        """Efficiently compute autoconvolution using Numba JIT compilation"""
        n = len(f_vals)
        g = np.zeros(2 * n - 1)

        # Manual convolution loop for efficiency
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < len(g):
                    g[idx] += f_vals[i] * f_vals[j]

        return g

    @staticmethod
    @jit(nopython=True)
    def compute_norms(g_vals: np.ndarray) -> Tuple[float, float, float]:
        """Compute L1, L2^2, and L-infinity norms efficiently"""
        # L1 norm (sum of absolute values)
        l1_norm = 0.0
        # L2^2 norm (sum of squares)
        l2_sq_norm = 0.0
        # L-infinity norm (maximum absolute value)
        linf_norm = 0.0

        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            l1_norm += abs_val
            l2_sq_norm += g_vals[i] * g_vals[i]
            if abs_val > linf_norm:
                linf_norm = abs_val

        return l1_norm, l2_sq_norm, linf_norm

    @classmethod
    def compute_c2(cls, f_vals: np.ndarray) -> float:
        """Compute C2 value using optimized functions"""
        # Compute autoconvolution
        g_vals = cls.compute_autoconvolution(f_vals)

        # Compute norms
        l1, l2_sq, linf = cls.compute_norms(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return 0.0

        # Return C2 value
        return l2_sq / (l1 * linf)

class Initializer:
    """Handles creation of initial step function configurations"""

    @staticmethod
    def create_bell_shaped_pattern(n_steps: int) -> np.ndarray:
        """Create a bell-shaped pattern emphasizing edges"""
        x = np.linspace(0, 1, n_steps)
        # Gaussian-like shape with emphasis on edges
        pattern = (1.0 + 0.8 * np.exp(-15 * (x - 0.5)**2) -
                  0.3 * np.exp(-5 * x**2) - 0.3 * np.exp(-5 * (1-x)**2))
        return np.clip(pattern, 0, np.inf)

    @staticmethod
    def create_alternating_pattern(n_steps: int) -> np.ndarray:
        """Create alternating high/low pattern"""
        pattern = []
        for i in range(n_steps):
            if i % 2 == 0:
                pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
            else:
                pattern.append(max(0.0, 0.1 + np.random.normal(0, 0.05)))
        return np.array(pattern)

    @staticmethod
    def create_peak_centered_pattern(n_steps: int) -> np.ndarray:
        """Create peak-centered pattern with tapering edges"""
        pattern = np.zeros(n_steps)
        center = n_steps // 2
        width = max(1, n_steps // 6 + np.random.randint(-1, 2))

        # Create a central peak
        pattern[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0

        # Add tapering to edges
        for i in range(center - width//2):
            pattern[i] *= (i / (center - width//2))
        for i in range(center + width//2, n_steps):
            pattern[i] *= ((n_steps - i) / (width//2 + 1))

        # Add some noise
        noise = np.random.normal(0, 0.05, n_steps)
        pattern = pattern + noise
        return np.clip(pattern, 0, np.inf)

    @staticmethod
    def create_smooth_transition_pattern(n_steps: int) -> np.ndarray:
        """Create smooth transition pattern"""
        pattern = np.zeros(n_steps)
        # Create smooth ramp with some random variation
        for i in range(n_steps):
            x = i / (n_steps - 1) if n_steps > 1 else 0.5
            pattern[i] = 0.5 + 0.5 * np.sin(np.pi * x) + np.random.normal(0, 0.1)
        return np.clip(pattern, 0, np.inf)

    @classmethod
    def create_multi_scale_initialization(cls, n_steps: int) -> np.ndarray:
        """Create diverse initial solution using multiple strategies"""
        strategies = [
            cls.create_bell_shaped_pattern,
            cls.create_alternating_pattern,
            cls.create_peak_centered_pattern,
            cls.create_smooth_transition_pattern
        ]

        # Choose a strategy randomly
        strategy = np.random.choice(strategies)
        pattern = strategy(n_steps)
        return pattern / np.sum(pattern) * n_steps

class Optimizer:
    """Handles optimization procedures"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_solution = None
        self.best_c2 = -float('inf')
        self.start_time = None

    def evaluate_function(self, f_vals: List[float]) -> float:
        """Evaluate a step function and return C2 value"""
        try:
            # Ensure non-negative values
            f_vals = np.array([max(0.0, x) for x in f_vals])

            # Handle edge cases
            if len(f_vals) == 0:
                return 0.0

            # Compute C2 value using optimized calculator
            c2 = AutoconvolutionCalculator.compute_c2(f_vals)

            # Ensure finite values
            if np.isnan(c2) or np.isinf(c2):
                return 0.0

            return c2
        except Exception:
            return 0.0

    def adaptive_evolutionary_optimization(self, initial_solution: List[float]) -> List[float]:
        """Enhanced evolutionary optimization with adaptive population sizing and early stopping"""
        # Track convergence
        best_scores = []
        patience_counter = 0
        max_patience = 10
        population_size = self.config.population_size

        # Start with initial solution
        current_solution = initial_solution.copy()
        current_c2 = self.evaluate_function(current_solution)
        best_scores.append(current_c2)

        # Time tracking
        time_limit = self.config.max_time_seconds * 0.9  # Leave some buffer

        # Adaptive parameters based on convergence behavior
        for generation in range(self.config.max_iterations):
            # Early termination check
            if time.time() - self.start_time > time_limit:
                break

            # Check for convergence
            if len(best_scores) >= 5:
                recent_improvement = best_scores[-1] - best_scores[-5]
                if recent_improvement < self.config.tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= max_patience:
                    # Increase population size to escape local minimum
                    population_size = min(population_size * 2, 50)
                    patience_counter = 0

            # Define bounds for differential evolution
            bounds = [(0.0, 10.0)] * len(current_solution)

            # Run differential evolution with adaptive parameters
            try:
                result = differential_evolution(
                    lambda x: -self.evaluate_function(x),  # Negative because we want to maximize
                    bounds,
                    maxiter=5,  # Fewer iterations per generation for speed
                    popsize=population_size,
                    seed=self.config.seed + generation,
                    strategy='best1bin',
                    tol=self.config.tolerance,
                    recombination=self.config.recombination_rate,
                    disp=False
                )

                if result.success:
                    new_solution = result.x.tolist()
                    new_c2 = self.evaluate_function(new_solution)

                    if new_c2 > current_c2:
                        current_solution = new_solution
                        current_c2 = new_c2
                        best_scores.append(current_c2)

            except Exception:
                pass  # Continue with current solution if optimization fails

        return current_solution

    def hybrid_local_search(self, solution: List[float]) -> List[float]:
        """Combine global and local search strategies"""
        # First, run the adaptive evolutionary optimization
        evolved_solution = self.adaptive_evolutionary_optimization(solution)

        # Then apply a simple gradient-like local refinement
        refined_solution = evolved_solution.copy()
        improvement_threshold = 1e-6

        # Simple gradient ascent approach
        for i in range(len(refined_solution)):
            # Try small positive perturbations
            original_value = refined_solution[i]
            perturbations = [0.01, 0.05, 0.1]

            best_value = original_value
            best_c2 = self.evaluate_function(refined_solution)

            for delta in perturbations:
                # Try increasing the value
                test_solution = refined_solution.copy()
                test_solution[i] = max(0, original_value + delta)

                c2_test = self.evaluate_function(test_solution)
                if c2_test > best_c2:
                    best_c2 = c2_test
                    best_value = original_value + delta

                # Try decreasing the value
                test_solution = refined_solution.copy()
                test_solution[i] = max(0, original_value - delta)

                c2_test = self.evaluate_function(test_solution)
                if c2_test > best_c2:
                    best_c2 = c2_test
                    best_value = original_value - delta

            refined_solution[i] = best_value

        return refined_solution

    def create_advanced_initialization(self, n_steps: int) -> np.ndarray:
        """Create more sophisticated initial pattern with mathematical insight"""
        # Create a pattern that encourages flat autoconvolution
        # This involves balancing high and low regions strategically

        # Strategy: Create a pattern with alternating high/low regions
        # but with some mathematical structure to encourage good convolutions

        pattern = np.zeros(n_steps)

        # Create strategic high-value regions
        high_regions = []
        n_high_regions = max(2, n_steps // 50)  # Varies with size

        for i in range(n_high_regions):
            start = int((i / n_high_regions) * n_steps)
            end = int(((i + 1) / n_high_regions) * n_steps)
            # Make some regions higher than others
            height_factor = 1.0 + np.random.random() * 2.0
            pattern[start:end] = 1.0 * height_factor

        # Add some smoothing to reduce sharp edges
        kernel = np.ones(5) / 5  # Simple moving average kernel
        pattern = np.convolve(pattern, kernel, mode='same')

        # Add a small Gaussian component to make it more refined
        x = np.linspace(-1, 1, n_steps)
        gaussian = np.exp(-0.5 * (x / 0.2)**2)
        pattern = pattern * 0.7 + gaussian * 0.3

        # Ensure non-negativity and normalization
        pattern = np.clip(pattern, 0, np.inf)
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * n_steps

        return pattern

    def multi_scale_optimization(self) -> List[float]:
        """Perform enhanced multi-scale optimization with adaptive strategies"""
        # Initialize with multiple random samples
        best_solution = None
        best_c2 = -float('inf')

        # Time tracking
        self.start_time = time.time()

        # Try several different initializations with different strategies
        for attempt in range(10):  # Increased attempts
            # Early termination check
            if time.time() - self.start_time > self.config.max_time_seconds * 0.9:
                break

            # Create diverse initial solution
            n_steps = np.random.randint(self.config.min_steps, self.config.max_steps)

            # Use different initialization strategies
            if attempt % 3 == 0:
                # Use advanced initialization
                initial_solution = self.create_advanced_initialization(n_steps)
            elif attempt % 3 == 1:
                # Use multi-scale initialization
                initial_solution = Initializer.create_multi_scale_initialization(n_steps)
            else:
                # Use standard approach with some variation
                initial_solution = np.random.random(n_steps) + 0.1
                initial_solution = initial_solution / np.sum(initial_solution) * n_steps

            # Optimize this initialization with hybrid approach
            optimized_solution = self.hybrid_local_search(initial_solution.tolist())

            # Evaluate result
            c2 = self.evaluate_function(optimized_solution)

            if c2 > best_c2:
                best_c2 = c2
                best_solution = optimized_solution

        # Final check of time limit
        if time.time() - self.start_time > self.config.max_time_seconds * 0.95:
            # Last resort: return the best solution found so far
            pass

        return best_solution if best_solution is not None else [1.0] * 100

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses enhanced modular optimization approach with multiple strategies.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Initialize configuration
    config = OptimizationConfig()

    # Initialize optimizer
    optimizer = Optimizer(config)

    # Set start time
    start_time = time.time()

    try:
        # Use enhanced multi-scale optimization approach
        best_solution = optimizer.multi_scale_optimization()

        # Final evaluation
        final_c2 = optimizer.evaluate_function(best_solution)

        end_time = time.time()
        eval_time = end_time - start_time

        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {final_c2:.6f}")

        return best_solution

    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * 100
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")