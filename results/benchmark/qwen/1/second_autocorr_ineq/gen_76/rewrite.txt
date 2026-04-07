# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import random

class AdaptiveEvolutionaryOptimizer:
    def __init__(self, max_time=90):
        self.max_time = max_time
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
        g = AdaptiveEvolutionaryOptimizer.compute_autoconvolution_numba(f)
        
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
    def advanced_initialization(n):
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
    
    @staticmethod
    def evaluate_function(f):
        """Evaluate the function and return C2 value with error handling"""
        try:
            if len(f) < 2:
                return 0.0
            
            # Convert to numpy array for fast computation
            f_array = np.array(f)
            
            # Ensure non-negativity
            f_array = np.maximum(f_array, 0)
            
            # Compute C2
            c2_value = AdaptiveEvolutionaryOptimizer.compute_c2_numba(f_array)
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
        popsize = min(max(10, n // 20), 30)
        
        # Iterations also scale but with bounds
        maxiter = min(max(20, n // 15), 60)
        
        return popsize, maxiter
    
    def single_optimization_run(self, n, seed_offset, timeout=60):
        """Perform a single optimization run with given parameters"""
        try:
            # Set seed for reproducibility
            np.random.seed(self.seed + seed_offset)
            
            # Get adaptive parameters
            popsize, maxiter = self.adaptive_parameters(n)
            
            # Generate initial population with multiple strategies
            initial_population = []
            for i in range(min(15, popsize)):
                # Mix different initialization strategies
                if i % 3 == 0:
                    # Advanced initialization
                    f_init = self.advanced_initialization(n)
                elif i % 3 == 1:
                    # Alternating initialization
                    f_init = self._alternating_initialization(n)
                else:
                    # Uniform initialization with noise
                    f_init = [1.0] * n
                    noise = np.random.normal(0, 0.1, n)
                    f_init = np.maximum(np.array(f_init) + noise, 0)
                
                initial_population.append(f_init.tolist())
            
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
                disp=False
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
    
    def multi_start_optimization(self, n):
        """Perform multi-start optimization with parallel execution"""
        current_best_c2 = 0.0
        current_best_f = None
        
        # Calculate number of parallel runs based on problem size
        num_runs = min(8, max(3, n // 200))
        
        # Set timeout for this optimization block
        start_time = time.time()
        timeout_per_run = min(15, max(5, self.max_time // (num_runs * 2)))
        
        # Prepare arguments for parallel execution
        args_list = [(n, i, timeout_per_run) for i in range(num_runs)]
        
        # Execute in parallel
        try:
            with ThreadPoolExecutor(max_workers=min(4, num_runs)) as executor:
                futures = [executor.submit(self.single_optimization_run, *args) 
                          for args in args_list]
                
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
                c2, f = self.single_optimization_run(n, i, timeout_per_run)
                if c2 is not None and c2 > current_best_c2:
                    current_best_c2 = c2
                    current_best_f = f
        
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
            2000, # Extra large configuration
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
        
        # If nothing worked, fallback to sophisticated initialization
        if len(best_f) == 0:
            # Try multiple sizes with advanced initialization
            for n in [1000, 2000]:
                try:
                    f_advanced = self.advanced_initialization(n)
                    c2_advanced = self.evaluate_function(f_advanced)
                    
                    if c2_advanced > best_c2:
                        best_c2 = c2_advanced
                        best_f = f_advanced
                        
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
    optimizer = AdaptiveEvolutionaryOptimizer(max_time=90)
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")