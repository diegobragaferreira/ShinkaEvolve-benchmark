# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import minimize
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time
from collections import deque
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class AutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with optimized numerical methods"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the autoconvolution g = f*f and its norms efficiently.
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        if not f_values or len(f_values) < 2:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4] with equal spacing
        n = len(f_values)
        
        # Step size in x domain [-1/4, 1/4]
        dx = 0.5 / (n - 1) if n > 1 else 0.5

        # Compute autoconvolution using numpy's convolution
        g = signal.convolve(f_values, f_values, mode='full')
        
        # Extract the central portion representing the actual convolution on [-1/2, 1/2]
        # For two functions of length n on [-1/4, 1/4], convolution produces 2*n-1 points
        center_start = len(g) // 2 - (n - 1)
        center_end = center_start + (2 * n - 1)
        g = g[center_start:center_end]

        # Compute the three norms
        # ||g||∞ = max of |g|
        norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0

        # ||g||₁ = sum of |g| * dx
        norm_1 = np.sum(np.abs(g)) * dx if len(g) > 1 else 0.0

        # ||g||₂² = ∫ g² dx using trapezoidal-like integration
        if len(g) <= 1:
            norm_2_squared = 0.0
        else:
            # Use piecewise linear integration for g^2
            # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
            norm_2_squared = 0.0
            for i in range(len(g)-1):
                y1, y2 = g[i], g[i+1]
                norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)

        return norm_2_squared, norm_1, norm_inf

    @classmethod
    def compute_c2(cls, f_values: List[float]) -> float:
        """Compute the C2 value for given step function."""
        norm_2_squared, norm_1, norm_inf = cls.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class GradientBasedStepFunctionOptimizer:
    """Optimizes step function using gradient-based methods instead of evolutionary algorithms"""
    
    def __init__(self):
        self.n_steps = 0
        self.step_sizes = []
        self.x_coords = []
        self.initial_solution = []
        self.bounds = []
        
    def _initialize_step_function(self, n_points: int = 500) -> List[float]:
        """
        Create initial step function based on theoretical insights:
        - Start with a symmetric structure to balance autoconvolution
        - Use logarithmic spacing of step heights to encourage flatness
        - Ensure non-negative values throughout
        """
        # Determine step count based on problem size
        self.n_steps = max(50, min(2000, n_points))
        
        # Create coordinate system [-1/4, 1/4]
        self.x_coords = np.linspace(-0.25, 0.25, self.n_steps)
        dx = 0.5 / (self.n_steps - 1) if self.n_steps > 1 else 0.5
        
        # Create logarithmically spaced step heights to encourage flat autoconvolution
        # Start with a base shape that promotes good C2 values
        base_heights = np.ones(self.n_steps)
        
        # Apply logarithmic decay from center outward to create flatter autoconvolution
        center_idx = self.n_steps // 2
        distances_from_center = np.abs(np.arange(self.n_steps) - center_idx)
        log_decay = np.exp(-distances_from_center / (self.n_steps / 4))
        
        # Apply some randomness but maintain structure
        noise = np.random.normal(0, 0.1, self.n_steps)
        base_heights = 0.5 + 0.5 * log_decay + noise
        
        # Clip to ensure non-negativity and reasonable bounds
        base_heights = np.clip(base_heights, 0, 3.0)
        
        # Normalize to avoid overly large values that could cause numerical issues
        if np.max(base_heights) > 1e-6:
            base_heights = base_heights / np.max(base_heights) * 2.0
        
        # Store bounds for optimization
        self.bounds = [(0.0, 10.0) for _ in range(self.n_steps)]
        self.initial_solution = base_heights.tolist()
        
        return self.initial_solution.copy()
    
    def _objective_function(self, step_heights: np.ndarray) -> float:
        """
        Objective function to maximize C2 value.
        Returns negative C2 since we're minimizing.
        """
        try:
            # Ensure non-negativity
            step_heights = np.maximum(step_heights, 0.0)
            
            # Compute autoconvolution norms
            norm_2_squared, norm_1, norm_inf = AutoconvolutionEvaluator.compute_autoconvolution_norms(step_heights.tolist())
            
            # Avoid numerical issues
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return 1e10  # Penalize invalid solutions heavily
            
            # Return negative C2 (since we minimize)
            c2 = norm_2_squared / (norm_1 * norm_inf)
            return -c2  # Negative because we maximize C2 but scipy minimizes
            
        except Exception as e:
            # Return large penalty for numerical errors
            return 1e10
    
    def _objective_gradient(self, step_heights: np.ndarray) -> np.ndarray:
        """
        Compute approximate gradient of objective function.
        Uses finite differences for gradient estimation.
        """
        epsilon = 1e-6
        grad = np.zeros_like(step_heights)
        
        # Compute directional derivatives
        for i in range(len(step_heights)):
            # Perturb dimension i
            step_h = step_heights.copy()
            step_h[i] += epsilon
            
            # Clamp to bounds to avoid out-of-bounds errors
            step_h = np.clip(step_h, 0, 10.0)
            
            # Compute function values
            f_plus = self._objective_function(step_h)
            f_original = self._objective_function(step_heights)
            
            # Approximate gradient
            grad[i] = (f_plus - f_original) / epsilon
            
        return grad
    
    def _smooth_solution(self, solution: List[float], smoothing_factor: float = 0.3) -> List[float]:
        """
        Apply gentle smoothing to the solution to encourage regularity.
        """
        if len(solution) < 3:
            return solution
            
        # Apply moving average filter
        smoothed = np.convolve(solution, np.ones(5)/5, mode='same')
        
        # Blend with original
        result = smoothing_factor * smoothed + (1 - smoothing_factor) * np.array(solution)
        
        return result.tolist()
    
    def optimize(self, max_time_seconds: int = 85) -> List[float]:
        """
        Main optimization routine using gradient-based optimization.
        """
        start_time = time.time()
        
        # Initialize with smart starting point
        initial_solution = self._initialize_step_function()
        
        # Set up optimization parameters
        options = {
            'maxiter': 500,  # Limit iterations to keep time under control
            'ftol': 1e-8,
            'gtol': 1e-6,
        }
        
        # Run optimization
        try:
            # Use L-BFGS-B which respects bounds
            result = minimize(
                fun=self._objective_function,
                x0=np.array(initial_solution),
                method='L-BFGS-B',
                bounds=self.bounds,
                options=options,
                callback=self._callback_function if max_time_seconds > 5 else None
            )
            
            # Extract optimized solution
            optimized_solution = result.x
            
            # Apply smoothing for better numerical behavior
            final_solution = self._smooth_solution(optimized_solution.tolist())
            
            # Ensure non-negativity
            final_solution = [max(0.0, x) for x in final_solution]
            
            # Validate the result
            if self._objective_function(np.array(final_solution)) < 1e5:
                return final_solution
            else:
                # Fall back to initial solution if optimization failed
                return initial_solution
                
        except Exception as e:
            # Fall back to initial solution if optimization fails
            print(f"Optimization error: {e}")
            return initial_solution
    
    def _callback_function(self, xk):
        """Simple callback to check time limits (not actually used in scipy)"""
        pass

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        optimizer = GradientBasedStepFunctionOptimizer()
        f_values = optimizer.optimize(max_time_seconds=85)
        return f_values
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple approach
        n_points = np.random.randint(100, 1000)
        f_values = [np.random.random() for _ in range(n_points)]
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")