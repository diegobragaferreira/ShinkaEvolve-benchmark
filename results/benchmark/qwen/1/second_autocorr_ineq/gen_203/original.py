# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import differential_evolution
from typing import List, Tuple, Optional
import random

# Core Calculation Module
class AutoconvolutionCalculator:
    """Computes autoconvolutions and C2 norms efficiently using Numba JIT compilation"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals: np.ndarray) -> np.ndarray:
        """Efficiently compute autoconvolution with manual Numba loop"""
        n = len(f_vals)
        g = np.zeros(2 * n - 1)
        
        # Manual convolution loop for efficiency - core optimization
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < len(g):
                    g[idx] += f_vals[i] * f_vals[j]
        
        return g
    
    @staticmethod
    @jit(nopython=True)
    def compute_norms(g_vals: np.ndarray) -> Tuple[float, float, float]:
        """Compute L1, L2^2, and L-infinity norms efficiently"""
        l1_norm = 0.0
        l2_sq_norm = 0.0
        linf_norm = 0.0
        
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            l1_norm += abs_val
            l2_sq_norm += g_vals[i] * g_vals[i]
            if abs_val > linf_norm:
                linf_norm = abs_val
        
        return l1_norm, l2_sq_norm, linf_norm
    
    @classmethod
    def compute_c2(cls, f_vals: np.ndarray) -> float:
        """Compute C2 value with numerical safety checks"""
        try:
            # Compute autoconvolution
            g_vals = cls.compute_autoconvolution(f_vals)
            
            # Compute norms
            l1, l2_sq, linf = cls.compute_norms(g_vals)
            
            # Avoid division by zero - critical safety check
            if l1 <= 1e-15 or linf <= 1e-15:
                return 0.0
            
            # Return C2 value
            return l2_sq / (l1 * linf)
        except Exception:
            return 0.0

# Initialization Module
class Initializer:
    """Creates diverse initial step function configurations"""
    
    @staticmethod
    def create_bell_shaped_pattern(n_steps: int) -> np.ndarray:
        """Create a bell-shaped pattern emphasizing edges"""
        x = np.linspace(0, 1, n_steps)
        # Gaussian-like shape with emphasis on edges
        pattern = (1.0 + 0.8 * np.exp(-15 * (x - 0.5)**2) - 
                  0.3 * np.exp(-5 * x**2) - 0.3 * np.exp(-5 * (1-x)**2))
        return np.clip(pattern, 0, np.inf)
    
    @staticmethod
    def create_alternating_pattern(n_steps: int) -> np.ndarray:
        """Create alternating high/low pattern"""
        pattern = []
        for i in range(n_steps):
            if i % 2 == 0:
                pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
            else:
                pattern.append(max(0.0, 0.1 + np.random.normal(0, 0.05)))
        return np.array(pattern)
    
    @staticmethod
    def create_peak_centered_pattern(n_steps: int) -> np.ndarray:
        """Create peak-centered pattern with tapering edges"""
        pattern = np.zeros(n_steps)
        center = n_steps // 2
        width = max(1, n_steps // 6 + np.random.randint(-1, 2))
        
        # Create a central peak
        pattern[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0
        
        # Add tapering to edges
        for i in range(center - width//2):
            pattern[i] *= (i / (center - width//2))
        for i in range(center + width//2, n_steps):
            pattern[i] *= ((n_steps - i) / (width//2 + 1))
        
        # Add some noise
        noise = np.random.normal(0, 0.05, n_steps)
        pattern = pattern + noise
        return np.clip(pattern, 0, np.inf)
    
    @staticmethod
    def create_smooth_transition_pattern(n_steps: int) -> np.ndarray:
        """Create smooth transition pattern"""
        pattern = np.zeros(n_steps)
        # Create smooth ramp with some random variation
        for i in range(n_steps):
            x = i / (n_steps - 1) if n_steps > 1 else 0.5
            pattern[i] = 0.5 + 0.5 * np.sin(np.pi * x) + np.random.normal(0, 0.1)
        return np.clip(pattern, 0, np.inf)
    
    @classmethod
    def create_multi_scale_initialization(cls, n_steps: int) -> np.ndarray:
        """Create diverse initial solution using multiple strategies"""
        strategies = [
            cls.create_bell_shaped_pattern,
            cls.create_alternating_pattern,
            cls.create_peak_centered_pattern,
            cls.create_smooth_transition_pattern
        ]
        
        # Choose a strategy randomly
        strategy = np.random.choice(strategies)
        pattern = strategy(n_steps)
        return pattern / np.sum(pattern) * n_steps

# Optimization Engine
class Optimizer:
    """Main optimization controller coordinating all components"""
    
    def __init__(self):
        self.best_solution = None
        self.best_c2 = -float('inf')
        self.max_time_seconds = 90.0
    
    def evaluate_function(self, f_vals: List[float]) -> float:
        """Primary evaluation function with comprehensive error handling"""
        try:
            # Ensure non-negative values with fast list comprehension  
            f_vals = np.array([max(0.0, x) for x in f_vals])
            
            # Handle edge cases immediately
            if len(f_vals) == 0:
                return 0.0
            
            # Compute C2 value using optimized calculator
            c2 = AutoconvolutionCalculator.compute_c2(f_vals)
            
            # Ensure finite values
            if np.isnan(c2) or np.isinf(c2):
                return 0.0
                
            return c2
        except Exception:
            return 0.0
    
    def evolutionary_optimization(self, initial_solution: List[float]) -> List[float]:
        """Evolutionary optimization with bounded search space"""
        try:
            # Define bounds for differential evolution
            bounds = [(0.0, 10.0)] * len(initial_solution)
            
            # Run differential evolution with fixed parameters for consistency
            result = differential_evolution(
                lambda x: -self.evaluate_function(x),  # Negative for maximization
                bounds,
                maxiter=50,  # Reduced iterations for speed
                popsize=15,   # Standard population size
                seed=42,
                strategy='best1bin',
                tol=1e-6,
                recombination=0.7,
                disp=False
            )
            
            # Return best solution found if successful
            if result.success:
                return result.x.tolist()
                
        except Exception:
            # Fall back gracefully if optimization fails
            pass
        
        # Return original solution if optimization fails
        return initial_solution
    
    def multi_scale_search(self) -> List[float]:
        """Multi-scale search with diverse initializations"""
        best_solution = None
        best_c2 = -float('inf')
        start_time = time.time()
        
        # Try several different initializations
        for attempt in range(8):  # Increased attempts for better exploration
            # Early termination check
            if time.time() - start_time > self.max_time_seconds * 0.95:
                break
                
            # Create diverse initial solution
            n_steps = np.random.randint(100, 1000)  # Broader range for diversity
            initial_solution = Initializer.create_multi_scale_initialization(n_steps)
            
            # Optimize this initialization
            optimized_solution = self.evolutionary_optimization(initial_solution.tolist())
            
            # Evaluate result
            c2 = self.evaluate_function(optimized_solution)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = optimized_solution
        
        return best_solution if best_solution is not None else [1.0] * 100

# Main Controller
def construct_function() -> List[float]:
    """
    Main entry point for constructing step-function with high C2 value.
    Uses modular optimization approach with diverse initialization strategies.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    try:
        # Use multi-scale search approach
        best_solution = optimizer.multi_scale_search()
        
        # Final evaluation
        final_c2 = optimizer.evaluate_function(best_solution)
        
        end_time = time.time()
        eval_time = end_time - time.time()  # This was incorrectly set to start_time previously
        
        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {final_c2:.6f}")
        
        return best_solution
        
    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * 100
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")