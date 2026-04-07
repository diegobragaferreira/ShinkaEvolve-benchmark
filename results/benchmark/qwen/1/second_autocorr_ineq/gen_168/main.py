# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit, prange
import numba
from typing import List, Tuple, Optional

class AutoconvolutionComputer:
    """Handles all autoconvolution computations with optimized Numba implementations"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals: np.ndarray) -> np.ndarray:
        """Compute autoconvolution using fast Numba implementation"""
        n = len(f_vals)
        
        # Convolution result has length 2*n-1
        g_len = 2 * n - 1
        g = np.zeros(g_len)

        # Compute convolution manually for efficiency
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_len:
                    g[idx] += f_vals[i] * f_vals[j]
                    
        return g
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_optimized(f_vals: np.ndarray) -> np.ndarray:
        """Optimized autoconvolution for larger arrays"""
        n = len(f_vals)
        
        if n <= 1000:
            # Direct computation for small arrays
            g_len = 2 * n - 1
            g = np.zeros(g_len)
            
            for i in range(n):
                for j in range(n):
                    idx = i + j
                    if 0 <= idx < g_len:
                        g[idx] += f_vals[i] * f_vals[j]
        else:
            # Special optimized approach for larger arrays
            g_len = 2 * n - 1
            g = np.zeros(g_len)
            
            # Pre-compute sums to reduce redundant calculations
            for i in range(n):
                temp_sum = 0.0
                for j in range(n):
                    temp_sum += f_vals[i] * f_vals[j]
                # Distribute to appropriate indices
                for k in range(n):
                    if i + k < g_len:
                        g[i + k] += temp_sum
                        
        return g

class NormComputer:
    """Computes various norms for autoconvolution results"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms(g_vals: np.ndarray) -> Tuple[float, float, float]:
        """Compute L1, L2^2, and L-infinity norms efficiently"""
        if len(g_vals) == 0:
            return 0.0, 0.0, 0.0
            
        # L1 norm (sum of absolute values)
        l1_norm = 0.0
        for i in range(len(g_vals)):
            l1_norm += abs(g_vals[i])
        
        # L2^2 norm (sum of squares)
        l2_sq_norm = 0.0
        for i in range(len(g_vals)):
            l2_sq_norm += g_vals[i] * g_vals[i]
        
        # L-infinity norm (maximum absolute value)
        linf_norm = 0.0
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            if abs_val > linf_norm:
                linf_norm = abs_val
                
        return l1_norm, l2_sq_norm, linf_norm
    
    @staticmethod
    @jit(nopython=True)
    def compute_c2_norms(g_vals: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute all norms needed for C2 calculation with proper integration"""
        if len(g_vals) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        # For L1 norm (sum of absolute values)
        g_l1 = 0.0
        for i in range(len(g_vals)):
            g_l1 += abs(g_vals[i])

        # For infinity norm (max absolute value)
        g_max = 0.0
        for i in range(len(g_vals)):
            if abs(g_vals[i]) > g_max:
                g_max = abs(g_vals[i])

        # Compute L2^2 norm using trapezoidal integration
        g_l2_sq = 0.0
        if len(g_vals) >= 2:
            g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
            for i in range(1, len(g_vals)-1):
                g_l2_sq += 2 * g_vals[i] * g_vals[i]
            # Step width for convolution domain
            h = 0.5 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
            g_l2_sq *= h / 2.0

        return g_l2_sq, g_l1, g_max, g_l2_sq / (g_l1 * g_max) if (g_l1 > 1e-15 and g_max > 1e-15) else 0.0

class OptimizerEngine:
    """Main optimization engine with adaptive strategies"""
    
    def __init__(self):
        self.autoconv_computer = AutoconvolutionComputer()
        self.norm_computer = NormComputer()
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function to minimize (negative C2)"""
        try:
            # Clip negative values
            f_vals = np.clip(params, 0, None)

            # Compute autoconvolution
            g_vals = self.autoconv_computer.compute_autoconvolution(f_vals)

            # Compute C2
            _, _, _, c2 = self.norm_computer.compute_c2_norms(g_vals)

            # Return negative because we're minimizing
            return -c2
        except Exception as e:
            return 1e10  # Large penalty for invalid results
    
    def sophisticated_initialization(self, dim: int) -> np.ndarray:
        """Create a sophisticated initial step function using multiple strategies"""
        # Generate points using Sobol sequence for better space-filling
        try:
            sampler = qmc.Sobol(d=dim, seed=42)
            points = sampler.random(n=100)
        except:
            # Fallback to regular random if Sobol fails
            points = np.random.random((100, dim))

        # Create a pattern that has shown success in previous implementations
        init_params = []
        for i in range(dim):
            # Create structured pattern: alternating high/low with sinusoidal modulation
            pattern_val = 0.5 + 0.3 * np.sin(i * 0.7)
            # Add variation from Sobol sampling
            variation = points[i % 100][0] * 0.2 if i < 100 else np.random.random() * 0.2
            init_params.append(max(0, pattern_val + variation - 0.1))

        return np.array(init_params)
    
    def evolutionary_optimization(self, initial_dim: int) -> np.ndarray:
        """Perform evolutionary optimization with adaptive parameters"""
        # Start with good initialization
        x0 = self.sophisticated_initialization(initial_dim)

        # Set bounds for optimization
        bounds = [(0, 10)] * len(x0)

        # Parameters for differential evolution
        de_params = {
            'mutation': (0.5, 1),
            'recombination': 0.7,
            'popsize': 15,
            'maxiter': 100,
            'seed': 42,
            'tol': 1e-6,
            'init': 'latinhypercube',
            'disp': False
        }

        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            **de_params
        )

        return result.x

class MultiStartOptimizer:
    """Handles multi-start optimization with various configurations"""
    
    def __init__(self):
        self.engine = OptimizerEngine()
    
    def run_multistart_optimization(self, max_time_seconds: float = 85.0) -> List[float]:
        """Run optimization with multiple starting configurations"""
        start_time = time.time()
        best_c2 = -np.inf
        best_params = None
        
        # Try multiple optimizations with different configurations
        configs = [
            (500, 42),
            (700, 123),
            (900, 456),
            (1100, 789)
        ]
        
        # Also try a few random configurations
        for _ in range(3):
            dim = np.random.randint(500, 1200)
            seed = np.random.randint(1000, 9999)
            configs.append((dim, seed))
        
        for dim, seed in configs:
            if time.time() - start_time > max_time_seconds - 2:  # Leave buffer for cleanup
                break

            try:
                np.random.seed(seed)
                params = self.engine.evolutionary_optimization(dim)

                # Compute actual C2 value
                f_vals = np.clip(params, 0, None)
                if len(f_vals) > 0:
                    # Use optimized autoconvolution
                    g_vals = self.engine.autoconv_computer.compute_autoconvolution_optimized(f_vals)
                    _, _, _, c2 = self.engine.norm_computer.compute_c2_norms(g_vals)

                    if c2 > best_c2:
                        best_c2 = c2
                        best_params = params.copy()
            except Exception as e:
                continue

        # If no valid parameters found, return default
        if best_params is None:
            return [0.5] * 100

        # Final check and conversion to list
        final_f_vals = np.clip(best_params, 0, None)
        return final_f_vals.tolist()

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value."""
    optimizer = MultiStartOptimizer()
    
    try:
        # Run multi-start optimization with time constraint
        f_values = optimizer.run_multistart_optimization(max_time_seconds=85.0)
        return f_values
    except Exception as e:
        # Return a fallback solution in case of any failure
        return [0.5] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")