# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import njit, prange
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import random

class ImprovedAdaptiveEvolutionaryOptimizer:
    def __init__(self, max_time=90):
        self.max_time = max_time
        self.seed = 42
        np.random.seed(self.seed)

    @staticmethod
    @njit(parallel=True)
    def compute_autoconvolution_numba_parallel(f):
        """Compute autoconvolution g = f * f using numba JIT with parallelization"""
        n = len(f)
        # Autoconvolution using discrete convolution
        g = np.zeros(2*n - 1)

        # Parallelized convolution loop
        for i in prange(n):
            for j in range(n):
                g[i + j] += f[i] * f[j]

        # Trim to center portion (length n-1)
        offset = (n - 1) // 2
        g_trimmed = g[offset:(2*n-1)-offset]
        return g_trimmed

    @staticmethod
    @njit
    def compute_c2_numba(f):
        """Compute C2 value for given step function f using numba JIT"""
        if len(f) < 2:
            return 0.0

        # Compute autoconvolution
        g = ImprovedAdaptiveEvolutionaryOptimizer.compute_autoconvolution_numba_parallel(f)

        if len(g) == 0:
            return 0.0

        # Compute norms using efficient accumulation
        norm_l2_sq = 0.0
        norm_l1 = 0.0
        norm_inf = 0.0

        # Iterate with explicit indexing for speed
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

    @staticmethod
    def enhanced_initialization(n):
        """
        Create enhanced initial step function with spectral pattern combinations
        Focuses on creating structures that promote favorable autoconvolution behavior
        """
        # Create a hybrid pattern based on mathematical principles with spectral components
        f = np.zeros(n)

        # Pattern 1: Multi-scale alternating pattern with spectral enhancement
        segment_sizes = [max(1, n // 20), max(1, n // 10), max(1, n // 5)]

        for i, seg_size in enumerate(segment_sizes):
            for j in range(0, n, seg_size):
                if (j // seg_size) % 2 == 0:
                    height = 1.0 + 0.5 * (i % 2)  # Varying heights
                else:
                    height = 0.1 + 0.2 * (i % 2)
                end_idx = min(j + seg_size, n)
                f[j:end_idx] = height

        # Pattern 2: Add spectral components for better convolution structure
        # Add sinusoidal modulation to encourage structured autoconvolution
        x = np.linspace(-1, 1, n)
        # Add a fundamental sinusoidal component
        fundamental_freq = 2.0 + np.random.random() * 3.0
        sin_component = np.sin(fundamental_freq * np.pi * x) * 0.2
        f += sin_component

        # Add a secondary harmonic component for richer spectrum
        second_freq = 4.0 + np.random.random() * 4.0
        sin_component2 = np.sin(second_freq * np.pi * x) * 0.15
        f += sin_component2

        # Pattern 3: Add sparse high-amplitude regions
        num_high_regions = max(1, n // 50)
        for _ in range(num_high_regions):
            pos = np.random.randint(0, n)
            width = max(1, n // 20)
            f[max(0, pos-width//2):min(n, pos+width//2)] += 0.5

        # Pattern 4: Apply smooth transition to reduce sharp edges
        if n > 10:
            kernel_size = min(7, max(3, n // 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/4)**2))
                kernel = kernel / np.sum(kernel)
                f = np.convolve(f, kernel, mode='same')

        # Ensure non-negativity and normalize
        f = np.clip(f, 0, None)
        if np.sum(f) > 0:
            f = f / np.sum(f) * n * 0.5

        return f.tolist()

    @staticmethod
    def evaluate_function(f):
        """Evaluate the function and return C2 value with error handling"""
        try:
            if len(f) < 2:
                return 0.0

            # Convert to numpy array for fast computation
            f_array = np.array(f, dtype=np.float64)

            # Ensure non-negativity
            f_array = np.maximum(f_array, 0)

            # Compute C2
            c2_value = ImprovedAdaptiveEvolutionaryOptimizer.compute_c2_numba(f_array)
            return c2_value

        except Exception as e:
            warnings.warn(f"Evaluation failed: {str(e)}")
            return 0.0

    def objective_function(self, x):
        """Objective function to minimize (negative C2)"""
        # x contains the step heights
        # Need to ensure non-negativity
        f = np.maximum(x, 0)
        c2 = self.evaluate_function(f)
        return -c2  # Negative because we want to maximize

    def adaptive_parameters(self, n):
        """Dynamically determine optimization parameters based on problem size"""
        # Population size scales with problem size but with bounds
        popsize = min(max(15, n // 15), 30)

        # Iterations also scale but with bounds
        maxiter = min(max(25, n // 12), 50)

        return popsize, maxiter

    def single_optimization_run(self, n, seed_offset):
        """Perform a single optimization run with given parameters"""
        try:
            # Set seed for reproducibility
            np.random.seed(self.seed + seed_offset)

            # Get adaptive parameters
            popsize, maxiter = self.adaptive_parameters(n)

            # Generate initial population with diverse strategies
            initial_population = []
            for i in range(min(20, popsize)):
                # Mix different initialization strategies for diversity
                if i % 4 == 0:
                    # Enhanced initialization
                    f_init = self.enhanced_initialization(n)
                elif i % 4 == 1:
                    # Gaussian peak pattern
                    f_init = self._gaussian_peak_initialization(n)
                elif i % 4 == 2:
                    # Alternating initialization
                    f_init = self._alternating_initialization(n)
                else:
                    # Uniform initialization with noise
                    f_init = [1.0] * n
                    noise = np.random.normal(0, 0.1, n)
                    f_init = np.maximum(np.array(f_init) + noise, 0)

                initial_population.append(f_init)

            # Define bounds for each parameter (step height)
            bounds = [(0, 10) for _ in range(n)]

            # Run differential evolution
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=self.seed + seed_offset,
                strategy='best1bin',
                init=initial_population,
                disp=False,
                workers=1  # Use single worker to avoid multiprocessing issues
            )

            if de_result.success:
                f_opt = np.maximum(de_result.x, 0)
                c2_value = -self.objective_function(f_opt)

                return c2_value, f_opt.tolist()

        except Exception as e:
            warnings.warn(f"Optimization run failed: {str(e)}")

        return None, None

    def _alternating_initialization(self, n):
        """Helper method for alternating pattern initialization"""
        f = []
        segment_length = max(1, n // 8)
        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                f.extend([1.0] * min(segment_length, n - i))
            else:
                f.extend([0.1] * min(segment_length, n - i))
        return f

    def _gaussian_peak_initialization(self, n):
        """Helper method for Gaussian peak initialization"""
        f = np.zeros(n)
        x = np.linspace(-1, 1, n)
        num_peaks = max(1, n // 30)
        for _ in range(num_peaks):
            center = np.random.uniform(-0.25, 0.25)
            width = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(0.5, 1.0)
            gauss_peak = amplitude * np.exp(-0.5 * ((x - center) / width)**2)
            f += gauss_peak
        f = np.clip(f, 0, None)
        if np.sum(f) > 0:
            f = f / np.sum(f)
        return f.tolist()

    def multi_start_optimization(self, n):
        """Perform multi-start optimization with reduced parallelism for stability"""
        current_best_c2 = 0.0
        current_best_f = None

        # Calculate number of parallel runs
        num_runs = min(6, max(2, n // 200))

        # Reduce number of runs to avoid resource overuse
        actual_runs = min(4, num_runs)

        # Set timeout for each run to respect overall time limit
        start_time = time.time()
        timeout_per_run = min(20, max(8, (self.max_time - (time.time() - start_time)) / (actual_runs * 2)))

        # Execute runs sequentially to maintain stability
        for i in range(actual_runs):
            try:
                c2, f = self.single_optimization_run(n, i)
                if c2 is not None and c2 > current_best_c2:
                    current_best_c2 = c2
                    current_best_f = f
            except Exception:
                continue

        return current_best_c2, current_best_f

    def construct_function(self) -> list[float]:
        """Main function to construct step-function with high C2 value"""
        # Initialize tracking variables
        best_c2 = 0.0
        best_f = []

        # Try different configurations to find the best one
        # Prioritize medium to large configurations for better results
        configurations = [
            500,   # Medium size - balance speed and quality
            1000,  # Large size for better optimization
            2000,  # Extra large for maximum potential
        ]

        start_time = time.time()

        for n in configurations:
            if time.time() - start_time > self.max_time * 0.9:
                break

            try:
                # Perform multi-start optimization
                c2_value, f_opt = self.multi_start_optimization(n)

                if c2_value > best_c2:
                    best_c2 = c2_value
                    best_f = f_opt

            except Exception as e:
                warnings.warn(f"Configuration {n} failed: {str(e)}")
                continue

        # If nothing worked, fallback to enhanced initialization
        if len(best_f) == 0:
            # Try multiple sizes with enhanced initialization
            for n in [1000, 2000]:
                try:
                    f_enhanced = self.enhanced_initialization(n)
                    c2_enhanced = self.evaluate_function(f_enhanced)

                    if c2_enhanced > best_c2:
                        best_c2 = c2_enhanced
                        best_f = f_enhanced

                except Exception as e:
                    warnings.warn(f"Fallback initialization {n} failed: {str(e)}")
                    continue

        # Final safety check - if still no good solution, use uniform distribution
        if len(best_f) == 0:
            n = 500
            best_f = [1.0] * n
            best_c2 = self.evaluate_function(best_f)

        return best_f

# Main execution function
def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimization."""
    optimizer = ImprovedAdaptiveEvolutionaryOptimizer(max_time=90)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")