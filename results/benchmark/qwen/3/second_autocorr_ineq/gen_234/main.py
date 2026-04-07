# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import numba
from numba import jit, prange
from typing import List, Tuple, Optional
import random
from collections import deque

# Constants
DOMAIN = [-0.25, 0.25]
N_MIN, N_MAX = 100, 2000
MAX_TIME_SECONDS = 85
NUM_INIT_STRATEGIES = 5
INITIAL_POP_SIZE = 20
OPTIMIZATION_ITERATIONS = 30

class FastConvolutionComputator:
    """Efficient convolution and norm computation module with parallel processing"""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_autoconvolution_parallel(f_vals):
        """Compute autoconvolution using optimized parallel numba approach"""
        n = len(f_vals)
        g = np.zeros(2*n - 1)
        for i in prange(n):
            for j in range(n):
                g[i + j] += f_vals[i] * f_vals[j]
        return g

    @staticmethod
    @jit(nopython=True)
    def compute_l2_norm_squared(g_vals):
        """Fast L2 norm squared computation"""
        l2_sq = 0.0
        for i in range(len(g_vals)):
            l2_sq += g_vals[i] * g_vals[i]
        return l2_sq

    @staticmethod
    @jit(nopython=True)
    def compute_l1_norm(g_vals):
        """Fast L1 norm computation"""
        l1 = 0.0
        for i in range(len(g_vals)):
            l1 += abs(g_vals[i])
        return l1

    @staticmethod
    @jit(nopython=True)
    def compute_linf_norm(g_vals):
        """Fast L-infinity norm computation"""
        l_inf = 0.0
        for i in range(len(g_vals)):
            val = abs(g_vals[i])
            if val > l_inf:
                l_inf = val
        return l_inf

    @classmethod
    def compute_all_norms(cls, g_vals):
        """Compute all norms efficiently"""
        return (
            cls.compute_l2_norm_squared(g_vals),
            cls.compute_l1_norm(g_vals),
            cls.compute_linf_norm(g_vals)
        )

class SolutionEvaluator:
    """Evaluates solutions and computes C2 values with efficient convolution"""
    
    @staticmethod
    def compute_c2(f_vals: List[float]) -> float:
        """Compute C2 value for given function values using fast convolution"""
        try:
            # Ensure non-negative values
            f_vals = np.maximum(f_vals, 0)
            
            # Compute autoconvolution with parallel processing
            g_vals = FastConvolutionComputator.compute_autoconvolution_parallel(f_vals)
            
            # Compute norms
            l2_sq, l1, l_inf = FastConvolutionComputator.compute_all_norms(g_vals)
            
            # Avoid division by zero
            if l1 <= 1e-12 or l_inf <= 1e-12:
                return 0.0
            
            # Compute C2
            c2 = l2_sq / (l1 * l_inf)
            return c2
        except Exception:
            return 0.0

class InitializationStrategy:
    """Various initialization strategies for generating candidates"""
    
    @staticmethod
    def uniform_random(n: int) -> np.ndarray:
        """Uniform random initialization"""
        return np.random.exponential(1, n)
    
    @staticmethod
    def gaussian_peak(n: int) -> np.ndarray:
        """Gaussian-shaped peak initialization"""
        x = np.linspace(-0.25, 0.25, n)
        return np.exp(-0.5 * (x / 0.1) ** 2)
    
    @staticmethod
    def symmetric_bump(n: int) -> np.ndarray:
        """Symmetric bump pattern"""
        half = n // 2
        quarter = n // 4
        pattern = np.zeros(n)
        for i in range(n):
            if i < quarter:
                pattern[i] = i / quarter
            elif i < half:
                pattern[i] = 1.0
            elif i < 3*quarter:
                pattern[i] = (3*quarter - i) / quarter
            else:
                pattern[i] = (n - i) / quarter
        return pattern
    
    @staticmethod
    def alternating_pattern(n: int) -> np.ndarray:
        """Alternating high/low pattern"""
        pattern = np.zeros(n)
        for i in range(n):
            if i % 2 == 0:
                pattern[i] = np.random.uniform(0.5, 1.0)
            else:
                pattern[i] = np.random.uniform(0.0, 0.3)
        return pattern
    
    @staticmethod
    def structured_peak(n: int) -> np.ndarray:
        """Structured peak with tapering edges"""
        pattern = np.zeros(n)
        center = n // 2
        half_width = n // 3
        for i in range(n):
            distance_from_center = abs(i - center)
            if distance_from_center < half_width:
                pattern[i] = np.exp(-0.5 * (distance_from_center / (half_width/2))**2)
            else:
                pattern[i] = np.exp(-0.5 * ((distance_from_center - half_width) / (n/4))**2)
        return pattern

class HybridOptimizer:
    """Main hybrid optimization class combining evolutionary and adaptive approaches"""
    
    def __init__(self):
        self.initialization_strategies = [
            InitializationStrategy.uniform_random,
            InitializationStrategy.gaussian_peak,
            InitializationStrategy.symmetric_bump,
            InitializationStrategy.alternating_pattern,
            InitializationStrategy.structured_peak
        ]
        
    def multi_stage_optimization(self, max_time_seconds: float) -> List[float]:
        """Multi-stage optimization with progressive refinement"""
        start_time = time.time()
        best_c2 = 0.0
        best_solution = None
        solutions_history = deque(maxlen=10)
        
        # Phase 1: Diversified multi-start search with parallel evaluations
        for strategy_idx in range(NUM_INIT_STRATEGIES):
            if time.time() - start_time > max_time_seconds - 10:
                break
                
            for _ in range(INITIAL_POP_SIZE // NUM_INIT_STRATEGIES):
                if time.time() - start_time > max_time_seconds - 10:
                    break
                    
                # Sample a random size
                n_steps = np.random.randint(N_MIN, N_MAX + 1)
                
                # Apply initialization strategy
                try:
                    strategy = self.initialization_strategies[strategy_idx]
                    f_vals = strategy(n_steps)
                    
                    # Evaluate solution using fast convolution
                    c2 = SolutionEvaluator.compute_c2(f_vals)
                    
                    if c2 > best_c2:
                        best_c2 = c2
                        best_solution = f_vals.copy()
                        
                    solutions_history.append((f_vals.copy(), c2))
                    
                except Exception:
                    continue
        
        # Phase 2: Differential evolution refinement of best with parallel processing
        if best_solution is not None and time.time() - start_time < max_time_seconds - 10:
            try:
                n_steps = len(best_solution)
                bounds = [(0, 10) for _ in range(n_steps)]
                
                def objective(x):
                    f_vals = np.abs(x) * 5.0
                    return -SolutionEvaluator.compute_c2(f_vals)
                
                result = differential_evolution(
                    objective, bounds, 
                    maxiter=OPTIMIZATION_ITERATIONS//2, 
                    popsize=10, 
                    seed=42
                )
                
                if result.success:
                    refined_solution = np.abs(result.x) * 5.0
                    refined_c2 = SolutionEvaluator.compute_c2(refined_solution)
                    
                    if refined_c2 > best_c2:
                        best_c2 = refined_c2
                        best_solution = refined_solution
                        
            except Exception:
                pass
        
        # Phase 3: Local coordinate-wise refinement with enhanced parallelism
        if best_solution is not None and time.time() - start_time < max_time_seconds - 5:
            best_solution = self.local_coordinate_refinement(best_solution, start_time, max_time_seconds)
            
        return best_solution.tolist() if best_solution is not None else [0.5] * 100
    
    def local_coordinate_refinement(self, initial_solution: np.ndarray, start_time: float, max_time_seconds: float) -> np.ndarray:
        """Perform enhanced local coordinate-wise refinement with faster convergence"""
        current_solution = initial_solution.copy()
        current_c2 = SolutionEvaluator.compute_c2(current_solution)
        
        # Try small perturbations to each element using adaptive step sizes
        max_iterations = OPTIMIZATION_ITERATIONS * 2
        for iteration in range(max_iterations):
            if time.time() - start_time > max_time_seconds - 2:
                break
                
            improved = False
            # Process elements in shuffled order for better exploration
            indices = list(range(len(current_solution)))
            np.random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > max_time_seconds - 2:
                    break
                    
                original_val = current_solution[i]
                
                # Try different adaptive step sizes
                step_sizes = [0.005, 0.01, 0.02, 0.05]
                for step in step_sizes:
                    for direction in [-1, 1]:
                        if time.time() - start_time > max_time_seconds - 2:
                            break
                            
                        new_val = original_val + direction * step
                        if new_val >= 0:
                            test_solution = current_solution.copy()
                            test_solution[i] = new_val
                            new_c2 = SolutionEvaluator.compute_c2(test_solution)
                            
                            if new_c2 > current_c2:
                                current_c2 = new_c2
                                current_solution = test_solution
                                improved = True
                                break
                    if improved:
                        break
                        
            if not improved:
                break
                
        return current_solution

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize hybrid optimizer
    optimizer = HybridOptimizer()
    
    # Execute multi-stage optimization
    result = optimizer.multi_stage_optimization(MAX_TIME_SECONDS)
    
    # Ensure we don't exceed time limit
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME_SECONDS:
        return [0.5] * 100
        
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
