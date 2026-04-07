# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from typing import List
from numba import njit
import time

class AutoconvolutionOptimizer:
    """Optimization class for finding step functions that maximize C₂"""

    def __init__(self, n_steps: int = 500, max_time: float = 90.0):
        self.n_steps = n_steps
        self.max_time = max_time
        self.seed = 42

    @staticmethod
    @njit
    def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
        """
        Fast computation of autoconvolution norms using Numba JIT compilation
        """
        n = len(f_values)

        # Compute autoconvolution g = f * f using discrete convolution
        g_length = 2 * n - 1
        g = np.zeros(g_length)

        # Manual convolution loop for speed - optimized version
        # This computes the discrete convolution sum_{k=-∞}^{∞} f[i] * f[j] where i+j=k
        # But since f is zero outside [0,n-1], we know that g[k] = sum_{i+j=k} f[i]*f[j]
        # where both i and j are in [0,n-1]
        for i in range(n):
            for j in range(n):
                k = i + j
                if 0 <= k < g_length:
                    g[k] += f_values[i] * f_values[j]

        # Compute the norms using improved integration
        norm_g_2_squared = 0.0

        # Improved trapezoidal-like integration for L2 norm:
        # Instead of treating each pair as a single trapezoid, we can integrate
        # more accurately by considering the actual convolution result structure
        # For a step function convolution, we get piecewise linear segments
        # Integration of y^2 over piecewise linear segments gives us:
        # For segment from (x1,y1) to (x2,y2): ∫(a*x+b)^2 dx = (1/3)*[(a*x2+b)^3 - (a*x1+b)^3] / a
        # But in our case, since we have step functions, we can compute:
        # ∫ y^2 dx ≈ (1/3)(y1^2 + y1*y2 + y2^2) for each adjacent pair
        for i in range(g_length - 1):
            y1 = g[i]
            y2 = g[i + 1]
            # Using the trapezoidal-like formula that accounts for quadratic variation
            # This represents integration of y^2 over unit step from y1 to y2
            norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

        # ||g||₁ = sum(|g[i]|)
        norm_g_1 = 0.0
        for i in range(g_length):
            norm_g_1 += abs(g[i])

        # ||g||∞ = max(|g[i]|)
        norm_g_inf = 0.0
        for i in range(g_length):
            abs_g = abs(g[i])
            if abs_g > norm_g_inf:
                norm_g_inf = abs_g

        return norm_g_2_squared, norm_g_1, norm_g_inf

    def compute_autoconvolution_norms(self, f_values: List[float]) -> tuple:
        """
        Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
        """
        f = np.array(f_values)
        norm_g_2_squared, norm_g_1, norm_g_inf = self.compute_autoconvolution_norms_fast(f)
        return norm_g_2_squared, norm_g_1, norm_g_inf

    def evaluate_c2(self, f_values: List[float]) -> float:
        """
        Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
        """
        try:
            norm_g_2_squared, norm_g_1, norm_g_inf = self.compute_autoconvolution_norms(f_values)

            # Avoid division by zero
            if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
                return 0.0

            c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
            return c2
        except Exception:
            return 0.0

    def generate_initial_population(self, n_individuals: int) -> np.ndarray:
        """
        Generate diverse initial population for evolutionary algorithm
        """
        population = []

        # Create various types of initial configurations
        np.random.seed(self.seed)  # For reproducibility

        for i in range(n_individuals):
            # Type 1: Alternating regions with smooth transitions
            if i % 4 == 0:
                f = self._create_alternating_pattern()
            # Type 2: Gaussian-like distribution
            elif i % 4 == 1:
                f = self._create_gaussian_like()
            # Type 3: Uniform distribution with structure
            elif i % 4 == 2:
                f = self._create_structured_uniform()
            # Type 4: Peak-centered with noise
            else:
                f = self._create_peak_centered()

            population.append(f)

        return np.array(population)

    def _create_alternating_pattern(self) -> np.ndarray:
        """Create alternating high/low segments with smooth transitions"""
        f = np.zeros(self.n_steps)
        segment_size = max(1, self.n_steps // 8)
        for i in range(0, self.n_steps, segment_size):
            end_idx = min(i + segment_size, self.n_steps)
            if (i // segment_size) % 2 == 0:
                f[i:end_idx] = 0.7 + np.random.random(end_idx - i) * 0.3
            else:
                f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.1

        # Add smooth Gaussian smoothing
        x = np.linspace(-1, 1, self.n_steps)
        gaussian = np.exp(-0.5 * (x / 0.25)**2)
        f = f * gaussian * 0.6 + gaussian * 0.4

        # Ensure non-negativity and normalize
        f = np.clip(f, 0, None)
        if np.sum(f) > 0:
            f = f / np.sum(f)
        return f

    def _create_gaussian_like(self) -> np.ndarray:
        """Create a Gaussian-like distribution"""
        x = np.linspace(-1, 1, self.n_steps)
        sigma = 0.2 + np.random.random() * 0.3
        mu = np.random.random() * 0.5 - 0.25
        f = np.exp(-0.5 * ((x - mu) / sigma)**2)
        f = f / np.sum(f)
        return f

    def _create_structured_uniform(self) -> np.ndarray:
        """Create structured uniform distribution with some randomness"""
        f = np.random.random(self.n_steps)
        f = np.clip(f, 0, 1)
        f = f / np.sum(f)
        return f

    def _create_peak_centered(self) -> np.ndarray:
        """Create peak-centered distribution with noise"""
        f = np.zeros(self.n_steps)
        center = self.n_steps // 2
        width = max(1, self.n_steps // 8 + np.random.randint(-2, 3))
        f[max(0, center-width//2):min(self.n_steps, center+width//2)] = 1.0
        f += np.random.normal(0, 0.03, self.n_steps)
        f = np.clip(f, 0, None)
        f = f / np.sum(f)
        return f

    def evolutionary_optimization(self) -> List[float]:
        """
        Use evolutionary algorithm to optimize step function
        """
        # Define bounds for each parameter (step height)
        bounds = [(0, 1.0) for _ in range(self.n_steps)]

        def objective(x):
            # Return negative because we want to maximize C2
            return -self.evaluate_c2(x.tolist())

        # Use differential evolution for global optimization
        try:
            # Use a smaller population and fewer iterations to stay within time limits
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,  # Reduced iterations for faster execution
                popsize=12,   # Smaller population for speed
                seed=self.seed,
                disp=False
            )

            if result.success:
                optimized_f = np.maximum(result.x, 0)
                # Normalize to ensure good scaling
                if np.sum(optimized_f) > 0:
                    optimized_f = optimized_f / np.sum(optimized_f)
                return optimized_f.tolist()
        except Exception as e:
            print(f"Optimization failed: {e}")

        # Return default if optimization fails
        return [1.0/self.n_steps] * self.n_steps

    def sophisticated_initialization(self) -> List[float]:
        """
        Generate a sophisticated initial configuration based on mathematical intuition
        """
        # Create a step function that tries to balance flatness with sufficient mass
        # Based on mathematical insights: create a function that when convolved
        # produces a relatively flat profile but with enough energy to achieve high C2

        # Start with alternating high/low regions with smooth transitions
        f = np.zeros(self.n_steps)

        # First create a base alternating pattern with some randomness
        segment_size = max(1, self.n_steps // 10)
        for i in range(0, self.n_steps, segment_size):
            end_idx = min(i + segment_size, self.n_steps)
            if (i // segment_size) % 2 == 0:
                # High region
                f[i:end_idx] = 0.8 + np.random.random(end_idx - i) * 0.2
            else:
                # Low region
                f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.1

        # Add Gaussian-based smoothing for more natural transitions
        x = np.linspace(-1, 1, self.n_steps)
        gaussian = np.exp(-0.5 * (x / 0.25)**2)
        f = f * gaussian * 0.6 + gaussian * 0.4

        # Add some noise to break symmetry
        noise = np.random.normal(0, 0.02, self.n_steps)
        f = f + noise

        # Ensure non-negativity
        f = np.clip(f, 0, None)

        # Normalize
        if np.sum(f) > 0:
            f = f / np.sum(f)

        return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using improved methods
    """
    start_time = time.time()

    # Create optimizer instance
    optimizer = AutoconvolutionOptimizer(n_steps=500, max_time=90.0)

    try:
        # Try sophisticated initialization first
        initial_f = optimizer.sophisticated_initialization()
        c2_initial = optimizer.evaluate_c2(initial_f)

        # Run evolutionary optimization
        optimized_f = optimizer.evolutionary_optimization()
        c2_optimized = optimizer.evaluate_c2(optimized_f)

        # Return the better of the two
        if c2_optimized > c2_initial:
            return optimized_f
        else:
            return initial_f

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")