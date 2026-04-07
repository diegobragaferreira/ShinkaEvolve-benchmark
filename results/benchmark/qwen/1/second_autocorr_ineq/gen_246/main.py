# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time
import warnings
import jax
import jax.numpy as jnp
from jax import grad, jit

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
    def compute_autoconvolution_jax(f):
        """Compute autoconvolution using JAX for potential gradient computation"""
        f = jnp.array(f)
        # Autoconvolution using discrete convolution
        g = jnp.convolve(f, f, mode='full')
        # Trim to center portion (length n-1)
        n = len(f)
        offset = (n - 1) // 2
        g = g[offset:-offset]
        return g

    @staticmethod
    def compute_c2_jax(f):
        """Compute C2 value using JAX for automatic differentiation"""
        try:
            # Compute autoconvolution
            g = StepFunctionOptimizer.compute_autoconvolution_jax(f)

            # Compute norms using JAX operations
            g_abs = jnp.abs(g)
            norm_l2_sq = jnp.sum(g_abs**2)
            norm_l1 = jnp.sum(g_abs)
            norm_inf = jnp.max(g_abs)

            # Avoid division by zero
            eps = 1e-12
            norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
            norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

            c2 = norm_l2_sq / (norm_l1 * norm_inf)
            return c2
        except Exception:
            return 0.0

    @staticmethod
    @jit
    def compute_autoconvolution_norms_jax(f):
        """
        Compute autoconvolution norms using JAX operations for vectorized batch processing
        """
        # Compute autoconvolution
        g = StepFunctionOptimizer.compute_autoconvolution_jax(f)

        # Compute norms using JAX operations
        g_abs = jnp.abs(g)
        norm_l2_sq = jnp.sum(g_abs**2)
        norm_l1 = jnp.sum(g_abs)
        norm_inf = jnp.max(g_abs)

        # Avoid division by zero
        eps = 1e-12
        norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
        norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

        return norm_l2_sq, norm_l1, norm_inf

    @staticmethod
    @jit
    def compute_c2_batch_jax(f_batch):
        """
        Compute C2 values for a batch of functions using vectorized operations
        """
        # Vectorized computation of norms for all functions in batch
        norms = jax.vmap(StepFunctionOptimizer.compute_autoconvolution_norms_jax)(f_batch)
        norm_l2_sq, norm_l1, norm_inf = norms

        # Compute C2 values for all functions in batch
        eps = 1e-12
        norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
        norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

        c2_batch = norm_l2_sq / (norm_l1 * norm_inf)
        return c2_batch

    @staticmethod
    def multi_scale_initialization(n):
        """
        Create multi-scale initial step function combining different patterns
        to better explore the structure of optimal autoconvolution profiles
        """
        f = np.zeros(n)

        # Pattern 1: Base alternating structure
        segment_size = max(1, n // 12)
        for i in range(0, n, segment_size):
            if (i // segment_size) % 2 == 0:
                f[i:i+min(segment_size, n-i)] = 1.0
            else:
                f[i:i+min(segment_size, n-i)] = 0.2

        # Pattern 2: Superimposed Gaussian peaks for fine structure
        x = np.linspace(-1, 1, n)
        for _ in range(3):  # Add 3 Gaussian peaks
            center = np.random.uniform(-0.25, 0.25)
            width = np.random.uniform(0.05, 0.2)
            amplitude = np.random.uniform(0.3, 0.8)
            gauss_peak = amplitude * np.exp(-0.5 * ((x - center) / width)**2)
            f += gauss_peak

        # Pattern 3: Sparse high-amplitude regions
        num_peaks = max(1, n // 50)
        for _ in range(num_peaks):
            pos = np.random.randint(0, n)
            width = max(1, n // 20)
            f[max(0, pos-width//2):min(n, pos+width//2)] += 0.5

        # Apply smoothing
        if n > 10:
            # Apply multiple smoothing passes for different scales
            kernel_size = min(7, n // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size > 1:
                kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/4)**2))
                kernel = kernel / np.sum(kernel)
                f = np.convolve(f, kernel, mode='same')

        # Normalize and clip
        f = np.clip(f, 0, None)
        if np.sum(f) > 0:
            f = f / np.sum(f) * n * 0.5

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

    def adaptive_population_size(self, n):
        """Adaptively determine population size based on problem size"""
        # For small problems, use smaller populations for faster exploration
        # For large problems, use larger populations for better exploitation
        if n < 200:
            return min(10, max(5, n // 20))
        elif n < 1000:
            return min(20, max(10, n // 50))
        else:
            return min(30, max(15, n // 100))

    def adaptive_maxiter(self, n):
        """Adaptively determine max iterations based on problem size"""
        if n < 200:
            return min(20, max(10, n // 10))
        elif n < 1000:
            return min(30, max(20, n // 20))
        else:
            return min(40, max(25, n // 30))

    def multi_stage_optimization(self, n):
        """Perform multi-stage optimization with fallbacks and adaptive parameters"""
        current_best_c2 = 0.0
        current_best_f = None

        # Stage 1: Differential Evolution with multi-scale initialization
        try:
            # Determine adaptive parameters
            popsize = self.adaptive_population_size(n)
            maxiter = self.adaptive_maxiter(n)

            # Generate initial population using multi-scale approach
            initial_population = []
            for _ in range(min(20, popsize)):  # Larger initial population for better exploration
                f_init = self.multi_scale_initialization(n)
                # Add noise to break symmetry
                noise = np.random.normal(0, 0.05, len(f_init))  # Reduced noise for stability
                f_noisy = np.maximum(np.array(f_init) + noise, 0)
                initial_population.append(f_noisy.tolist())

            # Define bounds for each parameter (step height)
            bounds = [(0, 10) for _ in range(n)]

            # Run differential evolution with adaptive parameters
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
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

        # Stage 2: Multi-start local optimization using consistent JAX autodiff
        if current_best_f is not None and current_best_c2 > 0.1:
            try:
                # Use consistent JAX-based optimization instead of mixed approaches
                best_local_c2 = current_best_c2
                best_local_f = current_best_f

                # Try multiple refinements using vectorized JAX autodiff
                try:
                    # Prepare batch of perturbed solutions
                    batch_size = 5  # Number of simultaneous refinements
                    f_array = np.array(current_best_f)

                    # Create batch of perturbed solutions
                    perturbed_batch = []
                    for _ in range(batch_size):
                        perturbed = f_array * (1 + np.random.normal(0, 0.03, len(f_array)))
                        perturbed = np.maximum(perturbed, 0)
                        perturbed_batch.append(perturbed)

                    # Convert to JAX array for vectorized computation
                    f_batch = jnp.array(perturbed_batch, dtype=jnp.float32)

                    # Vectorized gradient computation
                    grad_fn = grad(StepFunctionOptimizer.compute_c2_jax)
                    grad_batch = jax.vmap(grad_fn)(f_batch)

                    # Vectorized gradient ascent
                    step_size = 0.01
                    refined_batch = f_batch + step_size * grad_batch
                    refined_batch = jnp.maximum(refined_batch, 0)

                    # Convert back to numpy for evaluation
                    refined_numpy = np.array(refined_batch)

                    # Evaluate all refined solutions in batch
                    c2_values = []
                    for refined_f in refined_numpy:
                        c2_val = self.evaluate_function(refined_f.tolist())
                        c2_values.append(c2_val)

                    # Find best among refined solutions
                    best_idx = np.argmax(c2_values)
                    best_c2_refined = c2_values[best_idx]
                    best_refined_f = refined_numpy[best_idx].tolist()

                    if best_c2_refined > best_local_c2:
                        best_local_c2 = best_c2_refined
                        best_local_f = best_refined_f

                except Exception as refinement_error:
                    warnings.warn(f"Vectorized JAX refinement failed: {str(refinement_error)}")
                    # Fallback to single refinement if vectorized fails
                    try:
                        # Perturb current best solution
                        perturbed = np.array(current_best_f) * (1 + np.random.normal(0, 0.03, len(current_best_f)))
                        perturbed = np.maximum(perturbed, 0)

                        # Refine using single JAX-based gradient ascent
                        @jit
                        def refine_step(f_vals):
                            f = jnp.array(f_vals, dtype=jnp.float32)
                            grad_fn = grad(StepFunctionOptimizer.compute_c2_jax)
                            grad_val = grad_fn(f)
                            step_size = 0.01
                            refined = f + step_size * grad_val
                            refined = jnp.maximum(refined, 0)
                            return np.array(refined)

                        refined_f = refine_step(perturbed)
                        c2_refined = self.evaluate_function(refined_f)

                        if c2_refined > best_local_c2:
                            best_local_c2 = c2_refined
                            best_local_f = refined_f.tolist()

                    except Exception:
                        pass  # Continue with original result if all refinements fail

                # Update with better local result if found
                if best_local_c2 > current_best_c2:
                    current_best_c2 = best_local_c2
                    current_best_f = best_local_f

            except Exception as e:
                warnings.warn(f"Local optimization failed: {str(e)}")

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
            200,  # Smaller for faster exploration
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