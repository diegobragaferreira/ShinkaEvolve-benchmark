# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, prange
import time
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class OptimizationStrategy(Enum):
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    MULTI_SCALE = "multi_scale"

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    max_time_seconds: float = 85.0
    min_steps: int = 100
    max_steps: int = 1000
    population_size: int = 15
    max_iterations: int = 100
    seed: int = 42
    tolerance: float = 1e-6
    recombination_rate: float = 0.7

class AutoconvolutionComputation:
    """Handles computation of autoconvolution and related norms efficiently"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals: np.ndarray) -> np.ndarray:
        """Efficiently compute autoconvolution using Numba JIT compilation"""
        n = len(f_vals)
        g = np.zeros(2 * n - 1)
        
        # Manual convolution loop for efficiency
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
        # L1 norm (sum of absolute values normalized)
        l1_norm = 0.0
        # L2^2 norm (sum of squares with trapezoidal integration)
        l2_sq_norm = 0.0
        # L-infinity norm (maximum absolute value)
        linf_norm = 0.0
        
        # Process all elements
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            l1_norm += abs_val

        # Compute L2^2 with trapezoidal integration (piecewise linear)
        if len(g_vals) >= 2:
            # Trapezoidal rule for integral of g^2 with unit spacing
            # Approximate integral of g(t)^2 dt ≈ sum of (y_i^2 + y_i*y_{i+1})/3 for adjacent pairs
            for i in range(len(g_vals) - 1):
                y1 = g_vals[i]
                y2 = g_vals[i + 1]
                l2_sq_norm += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
        else:
            l2_sq_norm = g_vals[0] * g_vals[0] if len(g_vals) > 0 else 0.0
        
        # L-infinity norm
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            if abs_val > linf_norm:
                linf_norm = abs_val

        # Normalize L1
        if len(g_vals) > 0:
            l1_norm = l1_norm / (len(g_vals) + 1)

        return l1_norm, l2_sq_norm, linf_norm
    
    @classmethod
    def compute_c2(cls, f_vals: np.ndarray) -> float:
        """Compute C2 value using optimized functions"""
        # Compute autoconvolution
        g_vals = cls.compute_autoconvolution(f_vals)
        
        # Compute norms
        l1, l2_sq, linf = cls.compute_norms(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return 0.0
        
        # Return C2 value
        return l2_sq / (l1 * linf)

class StepFunctionInitializer:
    """Handles creation of initial step function configurations"""
    
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

class Optimizer:
    """Main optimization manager coordinating different strategies"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_solution = None
        self.best_c2 = -float('inf')
        
    def evaluate_function(self, f_vals: List[float]) -> float:
        """Evaluate a step function and return C2 value"""
        try:
            # Ensure non-negative values
            f_vals = np.array([max(0.0, x) for x in f_vals])
            
            # Handle edge cases
            if len(f_vals) == 0:
                return 0.0
            
            # Compute C2 value using optimized calculator
            c2 = AutoconvolutionComputation.compute_c2(f_vals)
            
            # Ensure finite values
            if np.isnan(c2) or np.isinf(c2):
                return 0.0
                
            return c2
        except Exception:
            return 0.0
    
    def evolutionary_optimization(self, initial_solution: List[float]) -> List[float]:
        """Use evolutionary algorithm with differential evolution for optimization"""
        # Define bounds for differential evolution
        bounds = [(0.0, 10.0)] * len(initial_solution)
        
        # Run differential evolution
        result = differential_evolution(
            lambda x: -self.evaluate_function(x),  # Negative because we want to maximize
            bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            seed=self.config.seed,
            strategy='best1bin',
            tol=self.config.tolerance,
            recombination=self.config.recombination_rate,
            disp=False
        )
        
        # Return best solution found
        return result.x.tolist()
    
    def local_refinement(self, solution: List[float], max_iter: int = 30) -> List[float]:
        """Apply local refinement to improve solution"""
        f = np.array(solution)
        n_steps = len(f)
        
        # Simple gradient-like approach with small perturbations
        for iteration in range(max_iter):
            current_c2 = self.evaluate_function(f.tolist())
            
            # Try small perturbations
            best_f = f.copy()
            best_c2 = current_c2
            
            # Perturb multiple times
            for _ in range(20):
                perturbed_f = f.copy()
                # Apply small random changes with controlled variance
                idx = np.random.randint(0, n_steps)
                delta = np.random.normal(0, 0.01)
                perturbed_f[idx] = max(0, perturbed_f[idx] + delta)
                
                # Normalize
                if np.sum(perturbed_f) > 0:
                    perturbed_f = perturbed_f / np.sum(perturbed_f)
                
                new_c2 = self.evaluate_function(perturbed_f.tolist())
                
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_f = perturbed_f
            
            f = best_f
            
            # Early stopping if improvement is minimal
            if abs(best_c2 - current_c2) < 1e-8:
                break
                
        return f.tolist()
    
    def multi_scale_optimization(self) -> List[float]:
        """Perform multi-scale optimization using multiple initialization strategies"""
        # Initialize with multiple random samples
        best_solution = None
        best_c2 = -float('inf')
        
        # Try several different initializations
        for attempt in range(8):
            # Create diverse initial solution
            n_steps = np.random.randint(self.config.min_steps, self.config.max_steps)
            initial_solution = StepFunctionInitializer.create_multi_scale_initialization(n_steps)
            
            # Optimize this initialization
            optimized_solution = self.evolutionary_optimization(initial_solution.tolist())
            
            # Evaluate result
            c2 = self.evaluate_function(optimized_solution)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = optimized_solution
        
        # Apply local refinement to the best result
        if best_solution is not None:
            refined_solution = self.local_refinement(best_solution)
            refined_c2 = self.evaluate_function(refined_solution)
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_solution
        
        return best_solution if best_solution is not None else [1.0] * 100

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value.
    Uses modular optimization approach with clear separation of concerns.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize configuration
    config = OptimizationConfig()
    
    # Initialize optimizer
    optimizer = Optimizer(config)
    
    # Set start time
    start_time = time.time()
    
    try:
        # Use multi-scale optimization approach
        best_solution = optimizer.multi_scale_optimization()
        
        # Ensure solution is valid and normalized
        best_solution = np.array(best_solution)
        best_solution = np.clip(best_solution, 0, np.inf)
        if np.sum(best_solution) > 0:
            best_solution = best_solution / np.sum(best_solution)
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        print(f"Eval time: {eval_time:.4f}s")
        print(f"Best C2 found: {optimizer.evaluate_function(best_solution.tolist()):.6f}")
        
        return best_solution.tolist()
        
    except Exception as e:
        # Fallback to simple approach if optimization fails
        print(f"Optimization failed with error: {e}. Using fallback.")
        fallback_solution = [1.0] * 100
        return fallback_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")