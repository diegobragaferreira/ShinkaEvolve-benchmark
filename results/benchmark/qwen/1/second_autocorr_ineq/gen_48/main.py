# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
from numba import jit
import time
from typing import List, Tuple, Callable

class AutoConvolutionOptimizer:
    """
    A modular optimizer for finding step functions that maximize C2 constant.
    """
    
    def __init__(self, 
                 max_iterations: int = 100,
                 population_size: int = 15,
                 num_initial_populations: int = 10,
                 time_limit_seconds: float = 90.0):
        """
        Initialize the optimizer with configurable parameters.
        
        Args:
            max_iterations: Maximum number of differential evolution iterations
            population_size: Population size for differential evolution
            num_initial_populations: Number of initial populations to generate
            time_limit_seconds: Time limit for optimization process
        """
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.num_initial_populations = num_initial_populations
        self.time_limit_seconds = time_limit_seconds
        self.seed = 42
        
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_norms_numba(f_vals: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute autoconvolution norms efficiently using numba acceleration.
        
        Returns:
            Tuple of (norm_g2_sq, norm_g1, norm_ginf)
        """
        n = len(f_vals)
        g = np.zeros(2 * n - 1)
        
        # Manual convolution loop for speed
        for i in range(n):
            for j in range(n):
                g[i + j] += f_vals[i] * f_vals[j]
        
        # Extract central portion
        half_len = n - 1
        g_center = g[half_len:-half_len] if len(g) > 2 * half_len else g[half_len:]
        
        # Compute norms
        norm_g2_sq = 0.0
        norm_g1 = 0.0
        norm_ginf = 0.0
        
        for i in range(len(g_center)):
            val = g_center[i]
            norm_g2_sq += val * val
            norm_g1 += abs(val)
            if abs(val) > norm_ginf:
                norm_ginf = abs(val)
                
        return norm_g2_sq, norm_g1, norm_ginf
    
    def compute_autoconvolution_norms(self, f_values: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C2 calculation.
        
        Args:
            f_values: Array of step function heights
            
        Returns:
            Tuple of (norm_g2_sq, norm_g1, norm_ginf)
        """
        # Ensure non-negative values
        f_vals = np.maximum(f_values, 0.0)
        
        # Use numba-accelerated computation
        try:
            return self.compute_autoconvolution_norms_numba(f_vals)
        except Exception:
            # Fallback to standard computation if numba fails
            return self._standard_compute_autoconvolution_norms(f_vals)
    
    def _standard_compute_autoconvolution_norms(self, f_vals: np.ndarray) -> Tuple[float, float, float]:
        """
        Standard implementation without numba for reliability.
        """
        # Compute autoconvolution using discrete convolution
        g = convolve(f_vals, f_vals, mode='full')
        
        # Keep only the valid convolution part (middle)
        half_len = len(f_vals) - 1
        g_valid = g[half_len:-half_len] if len(g) > 2 * half_len else g[half_len:]
        
        # Compute norms
        norm_g2_sq = np.sum(g_valid ** 2)
        norm_g1 = np.sum(np.abs(g_valid))
        norm_ginf = np.max(np.abs(g_valid))
        
        return norm_g2_sq, norm_g1, norm_ginf
    
    def evaluate_c2(self, f_values: List[float]) -> Tuple[float, float, float]:
        """
        Evaluate C2 = ||g||₂² / (||g||₁ · ||g||∞).
        
        Returns:
            Tuple of (c2_value, benchmark_ratio, eval_time)
        """
        start_time = time.time()
        
        # Convert to numpy array
        f_array = np.array(f_values, dtype=np.float64)
        
        try:
            # Compute norms
            norm_g2_sq, norm_g1, norm_ginf = self.compute_autoconvolution_norms(f_array)
            
            # Avoid division by zero
            if norm_g1 < 1e-15 or norm_ginf < 1e-15:
                c2 = 0.0
            else:
                c2 = norm_g2_sq / (norm_g1 * norm_ginf)
                
        except Exception:
            c2 = 0.0
            
        eval_time = time.time() - start_time
        benchmark_ratio = c2 / 0.962 if c2 > 0 else 0.0
        
        return c2, benchmark_ratio, eval_time
    
    def sophisticated_initialization(self, n_steps: int = None) -> List[float]:
        """
        Generate a sophisticated initial step function using multiple strategies.
        
        Args:
            n_steps: Number of steps in the function (random if None)
            
        Returns:
            List of step function heights
        """
        if n_steps is None:
            n_steps = np.random.randint(500, 5000)
            
        # Create an initial pattern based on mathematical intuition
        f_values = []
        
        # Strategy selection
        strategies = ['uniform', 'alternating', 'gaussian', 'mixed']
        strategy = np.random.choice(strategies)
        
        if strategy == 'uniform':
            # Simple uniform distribution
            f_values = [0.5] * n_steps
        elif strategy == 'alternating':
            # Alternating high/low values
            for i in range(n_steps):
                if i % 2 == 0:
                    f_values.append(np.random.uniform(0.7, 1.0))
                else:
                    f_values.append(np.random.uniform(0.0, 0.3))
        elif strategy == 'gaussian':
            # Create a bell-shaped pattern with Gaussian distribution
            x = np.linspace(-1, 1, n_steps)
            mu, sigma = 0, 0.3
            gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            # Scale so that maximum is around 1
            scale_factor = 1.0 / np.max(gauss)
            f_values = (gauss * scale_factor).tolist()
        else:  # mixed
            # Hybrid approach: alternating pattern with Gaussian smoothing
            temp_values = []
            for i in range(n_steps):
                if i % 2 == 0:
                    temp_values.append(np.random.uniform(0.7, 1.0))
                else:
                    temp_values.append(np.random.uniform(0.0, 0.3))
            
            # Apply Gaussian smoothing
            if n_steps >= 5:
                kernel_size = min(11, n_steps // 10)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma = kernel_size / 6.0
                x = np.arange(kernel_size) - kernel_size // 2
                gaussian_kernel = np.exp(-x**2 / (2 * sigma**2))
                gaussian_kernel /= np.sum(gaussian_kernel)
                temp_values = np.convolve(temp_values, gaussian_kernel, mode='same')
            
            f_values = temp_values.tolist()
        
        # Add some noise for exploration
        noise_level = 0.05
        f_values = [max(0, val + np.random.normal(0, noise_level)) for val in f_values]
        
        # Ensure non-negativity and reasonable scale
        f_values = np.maximum(f_values, 0)
        
        return f_values.tolist()
    
    def adaptive_evolutionary_optimization(self, initial_f_values: List[float]) -> List[float]:
        """
        Perform adaptive evolutionary optimization with multiple phases.
        
        Args:
            initial_f_values: Starting point for optimization
            
        Returns:
            Optimized step function heights
        """
        start_time = time.time()
        
        # Phase 1: Global exploration with differential evolution
        n_dimensions = len(initial_f_values)
        bounds = [(0, 3) for _ in range(n_dimensions)]  # Extended bounds for better exploration
        
        # Set bounds for DE
        max_iter_phase1 = min(self.max_iterations, int(self.time_limit_seconds * 0.7))
        
        def objective(f_vals_array):
            # Return negative because we minimize in scipy
            return -self.evaluate_c2(f_vals_array)[0]
        
        try:
            # Differential evolution for global optimization
            result_de = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter_phase1,
                popsize=self.population_size,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=self.seed,
                disp=False
            )
            
            # Get the best solution from DE
            best_solution = result_de.x
            
            # Phase 2: Local refinement with L-BFGS-B
            if result_de.success:
                # Refinement with local optimization
                refined_result = minimize(
                    objective,
                    best_solution,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max(10, int(self.time_limit_seconds * 0.2))}
                )
                
                if refined_result.success:
                    final_solution = refined_result.x
                else:
                    final_solution = best_solution
            else:
                final_solution = best_solution
                
        except Exception as e:
            # Fallback to original solution if optimization fails
            print(f"Optimization failed: {e}, falling back to initial solution")
            final_solution = np.array(initial_f_values)
        
        # Ensure non-negative values and reasonable scaling
        final_solution = np.maximum(final_solution, 0.0)
        
        # Normalize to maintain reasonable scale
        total = np.sum(final_solution)
        if total > 0:
            final_solution = final_solution / total * len(final_solution) * 0.5
        
        # Clip any remaining outliers
        final_solution = np.clip(final_solution, 0, 10)
        
        return final_solution.tolist()
    
    def optimize(self) -> Tuple[List[float], float, float, float]:
        """
        Main optimization routine that returns the optimized function and metrics.
        
        Returns:
            Tuple of (optimized_function, c2_value, benchmark_ratio, eval_time)
        """
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        
        # Generate initial sample
        initial_f_values = self.sophisticated_initialization()
        
        # Perform optimization
        try:
            optimized_f_values = self.adaptive_evolutionary_optimization(initial_f_values)
        except Exception as e:
            # Fallback to initial function if optimization fails
            print(f"Optimization failed: {e}")
            optimized_f_values = initial_f_values
        
        # Evaluate final result
        c2_value, benchmark_ratio, eval_time = self.evaluate_c2(optimized_f_values)
        
        return optimized_f_values, c2_value, benchmark_ratio, eval_time

def construct_function() -> List[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    
    Returns:
        List of step function heights that maximize C2
    """
    # Create optimizer instance with default parameters
    optimizer = AutoConvolutionOptimizer(
        max_iterations=100,
        population_size=15,
        num_initial_populations=10,
        time_limit_seconds=90.0
    )
    
    # Perform optimization
    optimized_function, _, _, _ = optimizer.optimize()
    
    return optimized_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
