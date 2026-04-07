# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from numba import njit
import random
from typing import List, Tuple, Optional, Callable
import time
from functools import lru_cache

class AutoconvolutionComputation:
    """Handles all autoconvolution and norm computations."""

    @staticmethod
    @njit
    def compute_autoconvolution(f_vals):
        """Compute autoconvolution g = f*f efficiently."""
        n = len(f_vals)
        g = np.zeros(2*n - 1)

        # Optimized nested loop for convolution
        for i in range(n):
            f_i = f_vals[i]
            for j in range(n):
                g[i + j] += f_i * f_vals[j]

        return g

    @staticmethod
    @njit
    def compute_norms(g_vals):
        """Compute the three required norms from autoconvolution result."""
        g_squared = g_vals * g_vals
        norm_g2_squared = np.sum(g_squared)
        norm_g1 = np.sum(np.abs(g_vals))
        norm_g_inf = np.max(np.abs(g_vals))
        return norm_g2_squared, norm_g1, norm_g_inf

    @classmethod
    def calculate_c2(cls, f_vals):
        """Calculate C2 value for given step function values."""
        g = cls.compute_autoconvolution(f_vals)
        norm_g2_squared, norm_g1, norm_g_inf = cls.compute_norms(g)

        # Avoid division by zero
        if norm_g1 < 1e-15 or norm_g_inf < 1e-15:
            return 0.0

        return norm_g2_squared / (norm_g1 * norm_g_inf)

class StrategyManager:
    """Manages different initialization and optimization strategies."""

    @staticmethod
    def generate_gamma_pattern(size: int) -> List[float]:
        """Generate gamma-distributed pattern."""
        initial_f_vals = np.random.gamma(2.0, 2.0, size)
        if np.sum(initial_f_vals) > 0:
            initial_f_vals = initial_f_vals / np.sum(initial_f_vals) * 100
        return initial_f_vals.tolist()

    @staticmethod
    def generate_exponential_pattern(size: int) -> List[float]:
        """Generate exponential-distributed pattern."""
        initial_f_vals = np.random.exponential(scale=0.5, size=size)
        initial_f_vals = np.maximum(initial_f_vals, 0.0)
        if np.sum(initial_f_vals) > 0:
            initial_f_vals = initial_f_vals / np.sum(initial_f_vals) * 100
        return initial_f_vals.tolist()

    @staticmethod
    def generate_multi_scale_gaussian_pattern(size: int) -> List[float]:
        """Generate multi-scale Gaussian pattern with hierarchical structure."""
        # Create base Gaussian distribution with multiple scales
        pattern = np.zeros(size)

        # Define multiple scales and positions for Gaussian bumps
        scales = [size // 20, size // 15, size // 10, size // 8]
        positions = [size // 4, size // 2, 3 * size // 4]

        # Generate hierarchical Gaussian patterns
        for scale in scales:
            for pos in positions:
                # Create Gaussian kernel
                x = np.arange(size)
                gaussian_kernel = np.exp(-0.5 * ((x - pos) / scale) ** 2)
                gaussian_kernel = gaussian_kernel / np.max(gaussian_kernel)  # Normalize
                pattern += gaussian_kernel * np.random.uniform(0.5, 2.0)

        # Add some additional structure with smaller scale components
        fine_scale = size // 50
        for _ in range(5):
            pos = np.random.randint(0, size)
            x = np.arange(size)
            fine_kernel = np.exp(-0.5 * ((x - pos) / fine_scale) ** 2)
            fine_kernel = fine_kernel / np.max(fine_kernel)
            pattern += fine_kernel * np.random.uniform(0.1, 0.5)

        # Ensure non-negativity and reasonable scaling
        pattern = np.maximum(pattern, 0.0)

        # Normalize to prevent extreme values
        if np.sum(pattern) > 0:
            pattern = pattern / np.sum(pattern) * 50

        return pattern.tolist()

    @staticmethod
    def generate_uniform_pattern(size: int) -> List[float]:
        """Generate uniform pattern."""
        return [1.0] * size

class OptimizerOrchestrator:
    """Coordinates the optimization process with multiple strategies."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def _evaluate_individual(self, individual: List[float]) -> Tuple[float, bool]:
        """Evaluate a single individual with error handling."""
        try:
            c2_score = AutoconvolutionComputation.calculate_c2(individual)
            return c2_score, True
        except Exception:
            return 0.0, False

    def _optimize_single_method(self, initial_values: List[float],
                               method: str = 'Nelder-Mead') -> Optional[List[float]]:
        """Optimize using a single method."""
        def objective(f_vals):
            return -AutoconvolutionComputation.calculate_c2(f_vals)

        try:
            result = minimize(
                objective,
                initial_values,
                method=method,
                options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6}
                if method == 'Nelder-Mead' else {'maxiter': 300}
            )

            if result.success:
                optimized_values = result.x
                optimized_values = np.maximum(optimized_values, 0.0)
                return optimized_values.tolist()
        except Exception:
            pass
        return None

    def _execute_strategy(self, strategy_func: Callable[[], List[float]]) -> Optional[List[float]]:
        """Execute a single strategy with optimization."""
        try:
            # Generate initial pattern
            current_solution = strategy_func()

            # Try optimization with multiple methods
            optimized_solutions = []

            # Try Nelder-Mead first
            nm_opt = self._optimize_single_method(current_solution, 'Nelder-Mead')
            if nm_opt is not None:
                optimized_solutions.append(nm_opt)

            # Try L-BFGS-B if available
            lbfgsb_opt = self._optimize_single_method(current_solution, 'L-BFGS-B')
            if lbfgsb_opt is not None:
                optimized_solutions.append(lbfgsb_opt)

            # Return best optimized solution
            if optimized_solutions:
                best_optimized = min(optimized_solutions,
                                   key=lambda x: -AutoconvolutionComputation.calculate_c2(x))
                return best_optimized

            # If nothing optimized, return original
            return current_solution

        except Exception:
            return None

    def _try_all_strategies(self) -> List[float]:
        """Try all optimization strategies and return best."""
        strategies = [
            StrategyManager.generate_gamma_pattern,
            StrategyManager.generate_exponential_pattern,
            StrategyManager.generate_pattern_with_peaks,
            StrategyManager.generate_multi_scale_gaussian_pattern
        ]

        best_solution = None
        best_c2 = -float('inf')

        # Execute each strategy with optimization
        for i, strategy_func in enumerate(strategies):
            try:
                solution = self._execute_strategy(strategy_func)
                if solution is not None:
                    c2_score, success = self._evaluate_individual(solution)
                    if success and c2_score > best_c2:
                        best_c2 = c2_score
                        best_solution = solution
            except Exception:
                continue

        # Fallback to uniform distribution if all strategies fail
        if best_solution is None:
            n_steps = 500
            best_solution = [1.0] * n_steps

        return best_solution

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        return self._try_all_strategies()

# Global optimizer instance
_optimizer = OptimizerOrchestrator()

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    return _optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")