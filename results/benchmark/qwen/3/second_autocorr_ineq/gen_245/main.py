# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy import signal
import time
import numba
from numba import jit
from typing import List, Tuple, Optional
import random
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DOMAIN = [-0.25, 0.25]
N_MIN, N_MAX = 100, 2000
MAX_TIME_SECONDS = 85
DEFAULT_RESOLUTIONS = [100, 200, 300, 500]
DEFAULT_POPULATION_SIZE = 20
DEFAULT_GENERATIONS = 50

class ConvolutionComputer:
    """Handles all convolution-related computations with numba optimization"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals):
        """Fast numba-based autoconvolution computation"""
        n = len(f_vals)
        g = np.zeros(2*n - 1)
        for i in range(n):
            for j in range(n):
                g[i + j] += f_vals[i] * f_vals[j]
        return g

    @staticmethod
    @jit(nopython=True)
    def compute_convolution_norms(g_vals):
        """Fast computation of convolution norms"""
        n = len(g_vals)
        l2_sq = 0.0
        l1 = 0.0
        l_inf = 0.0

        for i in range(n):
            val = g_vals[i]
            l2_sq += val * val
            l1 += abs(val)
            if abs(val) > l_inf:
                l_inf = abs(val)

        return l2_sq, l1, l_inf

class C2Calculator:
    """Computes C2 values with proper error handling"""
    
    @staticmethod
    def compute_c2(f_vals):
        """
        Compute C2 value using convolution approach
        Returns negative C2 for minimization purposes
        """
        try:
            # Ensure non-negative values
            f_vals = np.maximum(f_vals, 0)
            
            # Compute autoconvolution
            g_vals = ConvolutionComputer.compute_autoconvolution(f_vals)
            
            # Compute norms
            l2_sq, l1, l_inf = ConvolutionComputer.compute_convolution_norms(g_vals)
            
            # Avoid division by zero
            if l1 <= 1e-12 or l_inf <= 1e-12:
                return 1e10  # Large penalty for invalid results
                
            # Compute C2
            c2 = l2_sq / (l1 * l_inf)
            return -c2  # Negative for minimization
        except Exception as e:
            logger.error(f"C2 computation error: {e}")
            return 1e10

class Initializer:
    """Generates initial population and patterns"""
    
    @staticmethod
    def generate_sinc_initial_function(n):
        """
        Generate initial function based on sinc-like pattern 
        """
        x = np.linspace(-0.25, 0.25, n)
        
        # Create a pattern with low-frequency oscillations
        pattern = np.sinc(2 * x) * 0.5 + 0.5
        
        # Add some structured variation
        center = n // 2
        base = np.zeros(n)
        for i in range(n):
            dist = abs(i - center)
            base[i] = max(0, 1 - dist / (n/2))
        
        # Blend the patterns
        final = 0.6 * pattern + 0.4 * base
        
        # Ensure non-negativity
        final = np.maximum(final, 0)
        
        # Normalize to reasonable scale
        total = np.sum(final)
        if total > 0:
            final = final / total * 5.0
            
        return final.tolist()

    @staticmethod
    def generate_peaky_initial_function(n):
        """
        Generate peaky initial function
        """
        pattern = np.zeros(n)
        
        center = n // 2
        half_width = n // 3
        
        # Central peak
        for i in range(n):
            distance = abs(i - center)
            if distance < half_width:
                pattern[i] = 1.0 - (distance / half_width) ** 2
            else:
                pattern[i] = 0.0
                
        # Add some random variation
        for i in range(n):
            if np.random.random() < 0.05:
                pattern[i] *= np.random.uniform(0.7, 1.3)
                
        # Normalize
        total = np.sum(pattern)
        if total > 0:
            pattern = pattern / total * 10
            
        return pattern.tolist()

class Optimizer:
    """Main optimization controller"""
    
    def __init__(self, 
                 resolutions: List[int] = DEFAULT_RESOLUTIONS,
                 pop_size: int = DEFAULT_POPULATION_SIZE,
                 generations: int = DEFAULT_GENERATIONS):
        self.resolutions = resolutions
        self.pop_size = pop_size
        self.generations = generations
        self.best_solution: Optional[np.ndarray] = None
        self.best_c2 = -np.inf

    def _gradient_estimate(self, f_vals, epsilon=1e-4):
        """
        Estimate gradient using finite differences
        """
        n = len(f_vals)
        grad = np.zeros(n)
        base_c2 = -C2Calculator.compute_c2(f_vals)  # Negative for minimization
        
        for i in range(n):
            # Forward difference
            f_perturbed = f_vals.copy()
            f_perturbed[i] += epsilon
            perturbed_c2 = -C2Calculator.compute_c2(f_perturbed)
            
            grad[i] = (perturbed_c2 - base_c2) / epsilon
        
        return grad

    def _coarse_resolution_search(self, start_time: float, max_time_seconds: float):
        """Phase 1: Coarse resolution search"""
        for res in self.resolutions:
            if time.time() - start_time > max_time_seconds - 10:
                break
                
            try:
                f_vals = Initializer.generate_sinc_initial_function(res)
                
                # Simple gradient ascent to get in right ballpark
                current_f = np.array(f_vals)
                current_c2 = -C2Calculator.compute_c2(current_f)
                
                # Perform simple gradient-based updates
                for iter_num in range(100):
                    if time.time() - start_time > max_time_seconds - 10:
                        break
                        
                    grad = self._gradient_estimate(current_f)
                    step_size = 0.01
                    # Update along gradient direction (but stay non-negative)
                    new_f = current_f + step_size * grad
                    new_f = np.maximum(new_f, 0)
                    
                    new_c2 = -C2Calculator.compute_c2(new_f)
                    
                    if new_c2 > current_c2:
                        current_c2 = new_c2
                        current_f = new_f
                    else:
                        break
                        
                if current_c2 > self.best_c2:
                    self.best_c2 = current_c2
                    self.best_solution = current_f.copy()
                    
            except Exception as e:
                logger.warning(f"Error in coarse search at resolution {res}: {e}")
                continue

    def _fine_resolution_optimization(self, start_time: float, max_time_seconds: float):
        """Phase 2: Finer resolution optimization"""
        if (self.best_solution is None or 
            time.time() - start_time > max_time_seconds - 10):
            return
            
        try:
            # Increase resolution for fine-tuning
            fine_res = min(int(len(self.best_solution) * 1.5), N_MAX)
            
            # Create a refined version of the best solution
            coarse_solution = self.best_solution
            fine_solution = np.zeros(fine_res)
            
            # Interpolate from coarse to fine resolution
            coarse_x = np.linspace(0, len(coarse_solution) - 1, len(coarse_solution))
            fine_x = np.linspace(0, len(coarse_solution) - 1, fine_res)
            for i in range(fine_res):
                # Linear interpolation
                left_idx = int(fine_x[i])
                right_idx = min(left_idx + 1, len(coarse_solution) - 1)
                weight = fine_x[i] - left_idx
                
                fine_solution[i] = coarse_solution[left_idx] * (1 - weight) + \
                                  coarse_solution[right_idx] * weight
            
            # Fine-tune using scipy minimize with L-BFGS-B
            def obj_func(x):
                return C2Calculator.compute_c2(x)
                
            # Constraints to maintain non-negativity
            bounds = [(0, 100) for _ in range(fine_res)]
            
            # Optimize with bounds
            result = minimize(
                obj_func, 
                fine_solution, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'disp': False},
                tol=1e-6
            )
            
            if result.success:
                final_solution = result.x.copy()
                final_c2 = -obj_func(final_solution)
                
                if final_c2 > self.best_c2:
                    self.best_c2 = final_c2
                    self.best_solution = final_solution
                    
        except Exception as e:
            logger.warning(f"Error in fine resolution optimization: {e}")

    def _final_local_refinement(self, start_time: float, max_time_seconds: float):
        """Phase 3: Final local refinement"""
        if (self.best_solution is None or 
            time.time() - start_time > max_time_seconds - 5):
            return
            
        try:
            # Apply gradient-based local refinement
            current_solution = self.best_solution.copy()
            current_c2 = -C2Calculator.compute_c2(current_solution)
            
            # Gradient ascent with line search
            for iter_num in range(50):
                if time.time() - start_time > max_time_seconds - 5:
                    break
                    
                grad = self._gradient_estimate(current_solution, epsilon=1e-3)
                step_size = 0.01
                
                # Try different step sizes
                best_step_size = 0
                best_new_c2 = current_c2
                
                for test_step in [step_size, step_size*0.5, step_size*2]:
                    new_solution = current_solution + test_step * grad
                    new_solution = np.maximum(new_solution, 0)
                    new_c2 = -C2Calculator.compute_c2(new_solution)
                    
                    if new_c2 > best_new_c2:
                        best_new_c2 = new_c2
                        best_step_size = test_step
                        
                if best_step_size != 0:
                    current_solution = current_solution + best_step_size * grad
                    current_solution = np.maximum(current_solution, 0)
                    current_c2 = best_new_c2
                else:
                    break
                    
            self.best_solution = current_solution
            
        except Exception as e:
            logger.warning(f"Error in final local refinement: {e}")

    def optimize(self, max_time_seconds: float) -> List[float]:
        """
        Main optimization routine
        """
        start_time = time.time()
        
        # Phase 1: Coarse resolution search
        self._coarse_resolution_search(start_time, max_time_seconds)
        
        # Phase 2: Finer resolution optimization
        self._fine_resolution_optimization(start_time, max_time_seconds)
        
        # Phase 3: Final local refinement
        self._final_local_refinement(start_time, max_time_seconds)
        
        # Return result
        if self.best_solution is not None:
            # Final normalization and formatting
            total = np.sum(self.best_solution)
            if total > 0:
                self.best_solution = self.best_solution / total * 10
                
            return self.best_solution.tolist()
        else:
            # Fallback to simple pattern
            return Initializer.generate_sinc_initial_function(100)

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    Uses modularized optimization approach.
    """
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    # Execute the optimization process
    result = optimizer.optimize(MAX_TIME_SECONDS)
    
    # Ensure we don't exceed time limit
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME_SECONDS:
        logger.warning("Time limit exceeded, returning fallback solution")
        return [0.5] * 100
        
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")