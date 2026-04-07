# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import random
from typing import List, Tuple, Optional
import multiprocessing as mp
from functools import partial

@njit
def compute_autoconvolution_numba(f):
    """Compute autoconvolution g = f * f using numba JIT"""
    n = len(f)
    # Autoconvolution using discrete convolution
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]

    # Trim to center portion (length n-1)
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]
    return g_trimmed

@njit
def compute_c2_numba(f):
    """Compute C2 value for given step function f using numba JIT"""
    if len(f) < 2:
        return 0.0

    # Compute autoconvolution
    g = compute_autoconvolution_numba(f)

    if len(g) == 0:
        return 0.0

    # Compute norms
    norm_l2_sq = 0.0
    norm_l1 = 0.0
    norm_inf = 0.0

    for i in range(len(g)):
        abs_g = abs(g[i])
        norm_l2_sq += abs_g * abs_g
        norm_l1 += abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g

    # Avoid division by zero
    if norm_l1 < 1e-12 or norm_inf < 1e-12:
        return 0.0

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

class StepFunctionOptimizer:
    def __init__(self, max_time=90):
        self.max_time = max_time
        self.seed = 42
        self._set_random_seed()

    def _set_random_seed(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _create_initial_patterns(self, n: int) -> List[List[float]]:
        """Create diverse initial patterns for population"""
        patterns = []

        # Uniform pattern
        patterns.append([1.0] * n)

        # Alternating pattern
        pattern = []
        segment_length = max(1, n // 8)
        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                pattern.extend([1.0] * min(segment_length, n - i))
            else:
                pattern.extend([0.1] * min(segment_length, n - i))
        patterns.append(pattern)

        # Gaussian weighted pattern
        pattern = []
        segment_length = max(1, n // 8)
        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                pattern.extend([1.0] * min(segment_length, n - i))
            else:
                pattern.extend([0.1] * min(segment_length, n - i))

        # Apply Gaussian smoothing
        if len(pattern) > 0:
            pattern_arr = np.array(pattern)
            pattern_arr = np.clip(pattern_arr, 0, 10.0)
            kernel_size = min(5, len(pattern_arr)//4)
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
                kernel = kernel / np.sum(kernel)
                pattern_arr = np.convolve(pattern_arr, kernel, mode='same')
            pattern = np.maximum(pattern_arr, 0).tolist()
        patterns.append(pattern)

        # Mixed pattern
        half_n = n // 2
        pattern = [1.0] * half_n
        remaining = n - half_n
        if remaining > 0:
            alt_pattern = []
            segment_length = max(1, remaining // 8)
            for i in range(0, remaining, segment_length):
                if i // segment_length % 2 == 0:
                    alt_pattern.extend([1.0] * min(segment_length, remaining - i))
                else:
                    alt_pattern.extend([0.1] * min(segment_length, remaining - i))
            pattern.extend(alt_pattern)

        # Apply Gaussian smoothing to mixed pattern
        if len(pattern) > 0:
            pattern_arr = np.array(pattern)
            pattern_arr = np.clip(pattern_arr, 0, 10.0)
            kernel_size = min(5, len(pattern_arr)//4)
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
                kernel = kernel / np.sum(kernel)
                pattern_arr = np.convolve(pattern_arr, kernel, mode='same')
            pattern = np.maximum(pattern_arr, 0).tolist()
        patterns.append(pattern)

        # Advanced pattern
        pattern = []
        segment_length = max(1, n // 12)
        heights = [1.0, 0.7, 0.4, 0.2]
        for i in range(0, n, segment_length):
            height_idx = (i // segment_length) % len(heights)
            height = heights[height_idx]
            pattern.extend([height] * min(segment_length, n - i))

        # Apply Gaussian smoothing
        if len(pattern) > 0:
            pattern_arr = np.array(pattern)
            pattern_arr = np.clip(pattern_arr, 0, 10.0)
            kernel_size = min(7, max(3, len(pattern_arr) // 10))
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/4)**2))
                kernel = kernel / np.sum(kernel)
                pattern_arr = np.convolve(pattern_arr, kernel, mode='same')
            pattern = np.maximum(pattern_arr, 0).tolist()
        patterns.append(pattern)

        return patterns

    def _evaluate_population(self, population: List[List[float]]) -> List[Tuple[float, List[float]]]:
        """Evaluate all individuals in population efficiently"""
        results = []
        for individual in population:
            try:
                if len(individual) < 2:
                    results.append((0.0, individual))
                else:
                    # Convert to numpy array and ensure non-negativity
                    f_array = np.array(individual)
                    f_array = np.maximum(f_array, 0)

                    # Compute C2
                    c2_value = compute_c2_numba(f_array)
                    results.append((c2_value, individual))
            except Exception as e:
                warnings.warn(f"Evaluation failed: {str(e)}")
                results.append((0.0, individual))
        return results

    def _adaptive_parameters(self, n: int) -> Tuple[int, int]:
        """Dynamically determine optimization parameters based on problem size"""
        # Population size scales with problem size but with bounds
        popsize = min(max(10, n // 20), 30)

        # Iterations also scale but with bounds
        maxiter = min(max(20, n // 15), 60)

        return popsize, maxiter

    def _single_optimization_run(self, n: int, seed_offset: int,
                               initial_population: List[List[float]] = None) -> Tuple[Optional[float], Optional[List[float]]]:
        """Perform a single optimization run with given parameters"""
        try:
            # Set seed for reproducibility
            np.random.seed(self.seed + seed_offset)

            # Get adaptive parameters
            popsize, maxiter = self._adaptive_parameters(n)

            # Generate initial population if not provided
            if initial_population is None:
                initial_patterns = self._create_initial_patterns(n)
                initial_population = []
                for i, pattern in enumerate(initial_patterns):
                    # Add noise to break symmetry
                    noise = np.random.normal(0, 0.05, len(pattern))
                    individual = np.maximum(np.array(pattern) + noise, 0).tolist()
                    initial_population.append(individual)

            # Define bounds for each parameter (step height)
            bounds = [(0, 10) for _ in range(n)]

            # Helper function for differential evolution
            def objective_function(x):
                # x contains the step heights
                # Need to ensure non-negativity
                f = np.maximum(x, 0)
                c2 = compute_c2_numba(f)
                return -c2  # Negative because we want to maximize

            # Run differential evolution
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=self.seed + seed_offset,
                strategy='best1bin',
                init=initial_population,
                disp=False
            )

            if de_result.success:
                f_opt = np.maximum(de_result.x, 0)
                c2_value = -objective_function(f_opt)

                # If we have a good solution, also try local refinement
                if c2_value > 0.1:
                    try:
                        # Use L-BFGS for local refinement
                        lbfgs_result = minimize(
                            objective_function,
                            f_opt,
                            method='L-BFGS-B',
                            bounds=[(0, 10) for _ in range(n)],
                            options={'maxiter': min(50, max(20, n // 10)), 'disp': False}
                        )

                        if lbfgs_result.success:
                            f_refined = np.maximum(lbfgs_result.x, 0)
                            c2_refined = -objective_function(f_refined)

                            if c2_refined > c2_value:
                                c2_value = c2_refined
                                f_opt = f_refined.tolist()
                    except Exception:
                        pass  # Continue with original result if refinement fails

                return c2_value, f_opt.tolist()

        except Exception as e:
            warnings.warn(f"Optimization run failed: {str(e)}")

        return None, None

    def _multi_start_optimization(self, n: int) -> Tuple[Optional[float], Optional[List[float]]]:
        """Perform multi-start optimization with parallel execution"""
        current_best_c2 = 0.0
        current_best_f = None

        # Calculate number of parallel runs based on problem size
        num_runs = min(8, max(3, n // 200))

        # Set timeout for this optimization block
        start_time = time.time()
        timeout_per_run = min(15, max(5, self.max_time // (num_runs * 2)))

        # Create partial functions for parallel execution
        partial_func = partial(self._single_optimization_run, n)

        # Generate initial population for first run
        initial_patterns = self._create_initial_patterns(n)
        initial_population = []
        for i, pattern in enumerate(initial_patterns):
            # Add noise to break symmetry
            noise = np.random.normal(0, 0.05, len(pattern))
            individual = np.maximum(np.array(pattern) + noise, 0).tolist()
            initial_population.append(individual)

        # Execute in parallel using processes
        try:
            with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), num_runs)) as executor:
                futures = [executor.submit(partial_func, seed_offset=i,
                                         initial_population=initial_population if i == 0 else None)
                          for i in range(num_runs)]

                # Collect results
                for future in futures:
                    try:
                        c2, f = future.result(timeout=timeout_per_run + 5)
                        if c2 is not None and c2 > current_best_c2:
                            current_best_c2 = c2
                            current_best_f = f
                    except Exception:
                        continue

        except Exception:
            # Fallback to sequential if parallel fails
            for i in range(num_runs):
                try:
                    c2, f = self._single_optimization_run(n, i,
                                                        initial_population=initial_population if i == 0 else None)
                    if c2 is not None and c2 > current_best_c2:
                        current_best_c2 = c2
                        current_best_f = f
                except Exception:
                    continue

        return current_best_c2, current_best_f

    def _fallback_initialization(self, n: int) -> Tuple[float, List[float]]:
        """Fallback initialization strategies"""
        strategies = [
            self._uniform_initialization,
            self._alternating_initialization,
            self._gaussian_weighted_initialization,
            self._mixed_initialization,
            self._advanced_initialization
        ]

        best_strategy_c2 = 0.0
        best_strategy_f = None

        for strategy in strategies:
            try:
                f_strategy = strategy(n)
                c2_strategy = compute_c2_numba(np.array(f_strategy))
                if c2_strategy > best_strategy_c2:
                    best_strategy_c2 = c2_strategy
                    best_strategy_f = f_strategy
            except Exception as e:
                warnings.warn(f"Fallback strategy failed: {str(e)}")
                continue

        return best_strategy_c2, best_strategy_f

    def _uniform_initialization(self, n):
        """Initialize with uniform step heights"""
        return [1.0] * n

    def _alternating_initialization(self, n):
        """Initialize with alternating high/low segments"""
        f = []
        segment_length = max(1, n // 8)
        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                f.extend([1.0] * min(segment_length, n - i))
            else:
                f.extend([0.1] * min(segment_length, n - i))
        return f

    def _gaussian_weighted_initialization(self, n):
        """Initialize with Gaussian-weighted alternating pattern"""
        # Create alternating high/low segments
        f = []
        segment_length = max(1, n // 8)

        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                f.extend([1.0] * min(segment_length, n - i))
            else:
                f.extend([0.1] * min(segment_length, n - i))

        # Apply Gaussian weighting to smooth transitions
        if len(f) > 0:
            f = np.array(f)
            f = np.clip(f, 0, 10.0)

            # Apply Gaussian smoothing kernel
            kernel_size = min(5, len(f)//4)
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
                kernel = kernel / np.sum(kernel)
                f = np.convolve(f, kernel, mode='same')

            # Ensure all values are non-negative
            f = np.maximum(f, 0)

        return f.tolist()

    def _mixed_initialization(self, n):
        """Mix of different initialization strategies"""
        # Use uniform initialization for half the elements
        half_n = n // 2
        f = [1.0] * half_n

        # Add alternating pattern for the rest
        remaining = n - half_n
        if remaining > 0:
            alt_pattern = self._alternating_initialization(remaining)
            f.extend(alt_pattern)

        # Apply Gaussian weighting
        f = np.array(f)
        f = np.clip(f, 0, 10.0)

        # Apply Gaussian smoothing kernel
        kernel_size = min(5, len(f)//4)
        if kernel_size > 1:
            kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/3)**2))
            kernel = kernel / np.sum(kernel)
            f = np.convolve(f, kernel, mode='same')

        # Ensure all values are non-negative
        f = np.maximum(f, 0)

        return f.tolist()

    def _advanced_initialization(self, n):
        """
        Create advanced initial step function with multiple pattern combinations
        """
        # Create a more complex alternating pattern
        f = []
        segment_length = max(1, n // 12)

        # Alternate between multiple heights
        heights = [1.0, 0.7, 0.4, 0.2]
        for i in range(0, n, segment_length):
            height_idx = (i // segment_length) % len(heights)
            height = heights[height_idx]
            f.extend([height] * min(segment_length, n - i))

        # Apply Gaussian smoothing to create smoother transitions
        if len(f) > 0:
            f = np.array(f)
            f = np.clip(f, 0, 10.0)

            # Apply Gaussian smoothing kernel with variable size
            kernel_size = min(7, max(3, len(f) // 10))
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/4)**2))
                kernel = kernel / np.sum(kernel)
                f = np.convolve(f, kernel, mode='same')

            # Ensure all values are non-negative
            f = np.maximum(f, 0)

        return f.tolist()

    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value"""
        # Set seed for reproducibility
        self._set_random_seed()

        # Initialize tracking variables
        best_c2 = 0.0
        best_f = []

        # Try different configurations to find the best one
        configurations = [
            200,  # Smaller for faster exploration
            500,  # Medium configuration
            1000, # Large configuration
            2000, # Extra large configuration
        ]

        start_time = time.time()

        for n in configurations:
            if time.time() - start_time > self.max_time * 0.9:
                break

            try:
                # Perform multi-start optimization
                c2_value, f_opt = self._multi_start_optimization(n)

                if c2_value is not None and c2_value > best_c2:
                    best_c2 = c2_value
                    best_f = f_opt

            except Exception as e:
                warnings.warn(f"Configuration {n} failed: {str(e)}")
                continue

        # If nothing worked, fallback to sophisticated initialization
        if len(best_f) == 0:
            # Try multiple sizes with advanced initialization
            for n in [1000, 2000]:
                try:
                    c2_val, f_opt = self._fallback_initialization(n)
                    if c2_val > best_c2:
                        best_c2 = c2_val
                        best_f = f_opt
                except Exception as e:
                    warnings.warn(f"Fallback initialization {n} failed: {str(e)}")
                    continue

        # Final safety check - if still no good solution, use uniform distribution
        if len(best_f) == 0:
            n = 500
            best_f = [1.0] * n
            best_c2 = compute_c2_numba(np.array(best_f))

        return best_f

# Main execution function
def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimization."""
    optimizer = StepFunctionOptimizer(max_time=90)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")