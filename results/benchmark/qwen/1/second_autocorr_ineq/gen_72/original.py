# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time
import warnings

class StepFunctionOptimizer:
    def __init__(self, max_time=90, max_evaluations=1000):
        self.max_time = max_time
        self.max_evaluations = max_evaluations
        self.best_result = None
        self.seed = 42

    @staticmethod
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

    @staticmethod
    @njit
    def compute_c2_numba(f):
        """Compute C2 value for given step function f using numba JIT"""
        if len(f) < 2:
            return 0.0

        # Compute autoconvolution
        g = StepFunctionOptimizer.compute_autoconvolution_numba(f)

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

    @staticmethod
    def sophisticated_initialization(n):
        """
        Create sophisticated initial step function with alternating segments
        and Gaussian weighting to balance flatness and energy concentration
        """
        # Create alternating high/low segments
        f = []
        segment_length = max(1, n // 8)  # Variable segment size

        for i in range(0, n, segment_length):
            if i // segment_length % 2 == 0:
                # High segment
                f.extend([1.0] * min(segment_length, n - i))
            else:
                # Low segment
                f.extend([0.1] * min(segment_length, n - i))

        # Apply Gaussian weighting to smooth transitions
        if len(f) > 0:
            # Normalize to avoid extreme values
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

    def evaluate_function(self, f):
        """Evaluate the function and return C2 value with error handling"""
        try:
            if len(f) < 2:
                return 0.0

            # Convert to numpy array for fast computation
            f_array = np.array(f)

            # Ensure non-negativity
            f_array = np.maximum(f_array, 0)

            # Compute C2
            c2_value = self.compute_c2_numba(f_array)
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

    def multi_stage_optimization(self, n):
        """Perform multi-stage optimization with fallbacks"""
        current_best_c2 = 0.0
        current_best_f = None

        # Stage 1: Differential Evolution with sophisticated initialization
        try:
            # Generate initial population using sophisticated approach
            initial_population = []
            for _ in range(10):  # 10 initial individuals
                f_init = self.sophisticated_initialization(n)
                # Add some noise to break symmetry
                noise = np.random.normal(0, 0.1, len(f_init))
                f_noisy = np.maximum(np.array(f_init) + noise, 0)
                initial_population.append(f_noisy.tolist())

            # Define bounds for each parameter (step height)
            bounds = [(0, 10) for _ in range(n)]

            # Run differential evolution
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=30,
                popsize=10,
                seed=self.seed,
                strategy='best1bin',
                init=initial_population,
                disp=False
            )

            if de_result.success:
                f_opt = np.maximum(de_result.x, 0)
                c2_value = -self.objective_function(f_opt)

                if c2_value > current_best_c2:
                    current_best_c2 = c2_value
                    current_best_f = f_opt.tolist()

        except Exception as e:
            warnings.warn(f"Differential evolution failed: {str(e)}")

        # Stage 2: Local optimization (L-BFGS) if we have a good candidate
        if current_best_f is not None and current_best_c2 > 0.1:
            try:
                # Use L-BFGS for local refinement
                lbfgs_result = minimize(
                    self.objective_function,
                    current_best_f,
                    method='L-BFGS-B',
                    bounds=[(0, 10) for _ in range(n)],
                    options={'maxiter': 50, 'disp': False}
                )

                if lbfgs_result.success:
                    f_refined = np.maximum(lbfgs_result.x, 0)
                    c2_refined = -self.objective_function(f_refined)

                    if c2_refined > current_best_c2:
                        current_best_c2 = c2_refined
                        current_best_f = f_refined.tolist()

            except Exception as e:
                warnings.warn(f"L-BFGS refinement failed: {str(e)}")

        return current_best_c2, current_best_f

    def construct_function(self) -> list[float]:
        """Main function to construct step-function with high C2 value"""
        # Set seed for reproducibility
        np.random.seed(self.seed)

        # Initialize tracking variables
        best_c2 = 0.0
        best_f = []

        # Try different configurations to find the best one
        configurations = [
            100,  # Small configuration for quick testing
            500,  # Medium configuration
            1000, # Large configuration
        ]

        start_time = time.time()

        for n in configurations:
            if time.time() - start_time > self.max_time * 0.9:
                break

            try:
                # Perform multi-stage optimization
                c2_value, f_opt = self.multi_stage_optimization(n)

                if c2_value > best_c2:
                    best_c2 = c2_value
                    best_f = f_opt

            except Exception as e:
                warnings.warn(f"Configuration {n} failed: {str(e)}")
                continue

        # If nothing worked, fallback to sophisticated initialization
        if len(best_f) == 0:
            # Try multiple sizes with sophisticated initialization
            for n in [500, 1000, 2000]:
                try:
                    f_sophisticated = self.sophisticated_initialization(n)
                    c2_sophisticated = self.evaluate_function(f_sophisticated)

                    if c2_sophisticated > best_c2:
                        best_c2 = c2_sophisticated
                        best_f = f_sophisticated

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
    optimizer = StepFunctionOptimizer(max_time=90)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")