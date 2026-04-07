# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time
from numba import jit, prange
import numba
import jax.numpy as jnp
from jax import jit as jax_jit, grad
import jax
from functools import partial
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS
MAX_TIME_SECONDS = 85

class AutoConvolutionEvaluator:
    """Modular class for evaluating autoconvolution and C2 values with optimized computation"""
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution(f_vals):
        """Compute autoconvolution using optimized Numba implementation with better memory access"""
        n = len(f_vals)
        # Convolution result has length 2*n-1
        g_len = 2 * n - 1
        g = np.zeros(g_len)

        # Compute convolution manually with optimized loops
        # Use more efficient memory access pattern
        for i in range(n):
            f_i = f_vals[i]
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_len:
                    g[idx] += f_i * f_vals[j]

        return g

    @staticmethod
    @jit(nopython=True)
    def compute_c2_norms(g_vals):
        """Compute C2 norms efficiently with better numerical accuracy"""
        if len(g_vals) == 0:
            return 0.0, 0.0, 0.0

        # Compute norms using trapezoidal-like integration for L2^2
        g_l2_sq = 0.0
        g_l1 = 0.0
        g_max = 0.0

        # For L1 norm (sum of absolute values)
        for i in range(len(g_vals)):
            g_l1 += abs(g_vals[i])

        # For infinity norm (max absolute value)
        for i in range(len(g_vals)):
            if abs(g_vals[i]) > g_max:
                g_max = abs(g_vals[i])

        # Compute L2^2 norm correctly using trapezoidal-like integration
        if len(g_vals) >= 2:
            # For piecewise linear integration: integrate over intervals
            # Each interval contributes (h/3)(y1^2 + y1*y2 + y2^2) where h=1 in convolution domain
            for i in range(len(g_vals) - 1):
                y1 = g_vals[i]
                y2 = g_vals[i + 1]
                g_l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

        return g_l2_sq, g_l1, g_max

    @classmethod
    def evaluate_c2(cls, f_vals):
        """Evaluate C2 value with numerical stability and optimized computation"""
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        if len(f_vals) == 0:
            return 0.0

        # Compute autoconvolution
        g_vals = cls.compute_autoconvolution(f_vals)
        
        # Compute norms
        l2_sq, l1, l_inf = cls.compute_c2_norms(g_vals)
        
        # Compute C2 with numerical stability
        epsilon = 1e-15
        if l1 > epsilon and l_inf > epsilon:
            c2 = l2_sq / (l1 * l_inf)
        else:
            c2 = 0.0
            
        return c2

class OptimizerFactory:
    """Factory class for creating different optimization strategies"""
    
    @staticmethod
    def create_sophisticated_initialization(n_steps: int) -> np.ndarray:
        """Create a sophisticated initial population with alternating pattern and Gaussian smoothing"""
        # Create alternating high/low regions with some randomness
        initial = np.zeros(n_steps)

        # Divide into segments for structured alternation
        segment_size = max(1, n_steps // 10)

        for i in range(n_steps):
            segment_idx = i // segment_size
            if segment_idx % 2 == 0:
                # High value region (with some randomness)
                base_val = 1.0 + np.random.random() * 0.5
            else:
                # Low value region (with some randomness)
                base_val = 0.1 + np.random.random() * 0.3

            # Apply smoothing to avoid sharp transitions
            if i >= 2:
                # Smooth with previous values
                smooth_factor = 0.7
                base_val = smooth_factor * base_val + (1-smooth_factor) * 0.5 * (initial[i-1] + initial[i-2])

            initial[i] = max(0, base_val)

        return initial

    @staticmethod
    def create_harmonic_initialization(n_steps: int) -> np.ndarray:
        """Generate initial step function using harmonic patterns that are known to work well"""
        # Create a combination of harmonics that produce good autoconvolution properties
        initial = np.zeros(n_steps)
        
        # Base pattern: cosine wave with multiple frequencies
        freqs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Different frequencies
        amplitudes = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
        
        for i in range(n_steps):
            # Sum of cosines with different frequencies and amplitudes
            pattern_sum = 0.0
            for freq, amp in zip(freqs, amplitudes):
                pattern_sum += amp * np.cos(i * freq)
            initial[i] = max(0, 0.3 + 0.5 * pattern_sum)
        
        # Add some random variation
        noise = np.random.normal(0, 0.05, n_steps)
        initial += noise
        
        # Ensure non-negative
        initial = np.maximum(initial, 0)
        
        # Apply some smoothing to reduce sharp transitions
        if n_steps > 10:
            smoothed = initial.copy()
            for i in range(1, n_steps-1):
                smoothed[i] = 0.3 * initial[i-1] + 0.4 * initial[i] + 0.3 * initial[i+1]
            initial = smoothed
        
        return initial

    @staticmethod
    def create_structure_aware_initialization(n_steps: int) -> np.ndarray:
        """Create initial population that is aware of the structure that tends to produce better results"""
        # Create a structure that combines multiple peaks and valleys to encourage good autoconvolution
        initial = np.zeros(n_steps)
        
        # Create multiple regions with different patterns
        region_size = max(1, n_steps // 8)
        
        for i in range(0, n_steps, region_size):
            region_end = min(i + region_size, n_steps)
            
            # Within each region, create alternating high/low pattern
            for j in range(i, region_end):
                # Create some periodicity in the pattern
                pattern_pos = j % (region_size // 4) if region_size > 4 else 0
                if pattern_pos < (region_size // 4):
                    # High values
                    initial[j] = 0.8 + 0.2 * np.random.random()
                else:
                    # Low values  
                    initial[j] = 0.1 + 0.1 * np.random.random()
        
        # Add some noise to avoid perfect patterns that might get stuck
        noise = np.random.normal(0, 0.05, n_steps)
        initial += noise
        initial = np.maximum(initial, 0)
        
        return initial

    @staticmethod
    def create_multi_initializations(n_steps: int, count: int = 3) -> List[np.ndarray]:
        """Create multiple diverse initializations"""
        initializations = []
        
        # Structure-aware pattern
        np.random.seed(42)
        initializations.append(OptimizerFactory.create_structure_aware_initialization(n_steps))
        
        # Harmonic pattern
        np.random.seed(123)
        initializations.append(OptimizerFactory.create_harmonic_initialization(n_steps))
        
        # Sophisticated pattern
        np.random.seed(234)
        initializations.append(OptimizerFactory.create_sophisticated_initialization(n_steps))
        
        # Random structured patterns
        for i in range(count-3):
            np.random.seed(345 + i)
            pattern = np.random.random(n_steps) * 0.5 + 0.25
            initializations.append(pattern)
            
        return initializations

class GradientOptimizer:
    """Module for gradient-based optimization using JAX"""
    
    @staticmethod
    @jax_jit
    def evaluate_c2_jax(f_vals):
        """JAX-based C2 computation for gradient-based optimization"""
        try:
            # Ensure non-negative values
            f_vals = jnp.maximum(f_vals, 0.0)

            # Compute autoconvolution using JAX's convolution
            g_vals = jnp.convolve(f_vals, f_vals, mode='full')

            # Compute norms using JAX operations
            # L2^2 norm using piecewise linear integration
            l2_squared = 0.0
            if len(g_vals) >= 2:
                for i in range(len(g_vals)-1):
                    y1 = g_vals[i]
                    y2 = g_vals[i+1]
                    l2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

            # L1 norm (piecewise constant approximation)
            l1 = jnp.sum(jnp.abs(g_vals)) / (len(g_vals) + 1) if len(g_vals) + 1 > 0 else 0.0

            # L-infinity norm
            l_inf = jnp.max(jnp.abs(g_vals))

            # Avoid division by zero
            l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
            l_inf_safe = jnp.where(l_inf <= 1e-15, 1e-15, l_inf)

            # Compute C2
            c2 = l2_squared / (l1_safe * l_inf_safe)
            return c2
        except:
            return 0.0

    @staticmethod
    @partial(jax_jit, static_argnums=(0,))
    def compute_gradients_jax(f_vals, num_steps):
        """Compute gradients of C2 w.r.t. input using JAX"""
        # Create a wrapper for jax.grad
        def c2_wrapper(f_vals_vec):
            f_vals = f_vals_vec.reshape(num_steps)
            return GradientOptimizer.evaluate_c2_jax(f_vals)
        
        # Compute gradients
        grad_fn = jax.grad(c2_wrapper)
        gradients = grad_fn(jnp.array(f_vals))
        return gradients

    @staticmethod
    def adaptive_gradient_optimization(initial_params: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Adaptive gradient-based optimization approach with improved stability"""
        # Convert to JAX array for gradient computation
        x0 = jnp.array(initial_params)
        
        # Adaptive learning rate and iterations
        learning_rate = 0.01
        
        # Track best solution
        best_x = x0
        best_c2 = GradientOptimizer.evaluate_c2_jax(x0)
        
        # Store history for convergence monitoring
        history = [best_c2]
        
        for iteration in range(max_iter):
            # Compute gradients
            try:
                grads = GradientOptimizer.compute_gradients_jax(x0, len(initial_params))
                
                # Update parameters with gradient ascent (since we want to maximize C2)
                x_new = x0 + learning_rate * grads
                
                # Project back to feasible space [0, 1]
                x_new = jnp.clip(x_new, 0.0, 1.0)
                
                # Evaluate new solution
                new_c2 = GradientOptimizer.evaluate_c2_jax(x_new)
                
                # Accept improvement
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_x = x_new
                    history.append(best_c2)
                    
                # Update for next iteration
                x0 = x_new
                
                # Reduce learning rate over time
                learning_rate *= 0.995
                
            except Exception as e:
                # If gradient computation fails, fall back to differential evolution
                break
                
        return np.array(best_x)

class HybridOptimizer:
    """Main hybrid optimizer that orchestrates the optimization workflow"""
    
    def __init__(self, n_steps: int = 1000):
        self.n_steps = n_steps
    
    def optimize(self) -> List[float]:
        """Main optimization loop with modular components and improved strategies"""
        # Phase 1: Global search with multi-start and better initialization
        best_c2 = -np.inf
        best_params = None
        
        # Create diverse initializations with higher variety
        initializations = OptimizerFactory.create_multi_initializations(self.n_steps, 5)
        
        # Run evolutionary optimization on each initialization
        start_time = time.time()
        for i, x0 in enumerate(initializations):
            if time.time() - start_time > MAX_TIME_SECONDS * 0.9:  # Leave buffer
                break
                
            try:
                # Use differential evolution for global search
                bounds = [(0.0, 1.0) for _ in range(self.n_steps)]
                
                # Adaptive population size based on dimensionality
                popsize = max(15, min(30, self.n_steps // 50))
                
                result = differential_evolution(
                    lambda x: -AutoConvolutionEvaluator.evaluate_c2(x),
                    bounds,
                    x0=x0,
                    seed=i + 42,
                    maxiter=100,
                    popsize=popsize,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False,
                    tol=1e-6
                )
                
                c2 = -result.fun
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = result.x.copy()
                    
            except Exception:
                continue
        
        # Phase 2: Local refinement using gradient-based optimization
        if best_params is not None:
            try:
                refined_params = GradientOptimizer.adaptive_gradient_optimization(best_params)
                refined_c2 = AutoConvolutionEvaluator.evaluate_c2(refined_params)
                
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_params = refined_params
            except Exception:
                pass
        
        # Return best solution
        return best_params.tolist() if best_params is not None else [0.5] * self.n_steps

def construct_function() -> List[float]:
    """Main function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Create and run hybrid optimizer with optimized parameters
    optimizer = HybridOptimizer(n_steps=1000)
    f_values = optimizer.optimize()
    
    end_time = time.time()
    
    # Final verification of C2 value
    try:
        final_c2 = AutoConvolutionEvaluator.evaluate_c2(np.array(f_values))
    except:
        final_c2 = 0.0
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"C2 achieved: {final_c2}")
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")