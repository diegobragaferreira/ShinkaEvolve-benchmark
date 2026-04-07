# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple, Optional

class AutoconvolutionEvaluator:
    """Handles computation of autoconvolution and C2 norms with numerical stability"""
    
    def __init__(self, domain: Tuple[float, float] = (-0.25, 0.25), n_bins: int = 1000):
        self.domain = domain
        self.n_bins = n_bins
        self.step_width = (domain[1] - domain[0]) / n_bins
    
    @staticmethod
    @jit(nopython=True)
    def compute_autoconvolution_fast(f_vals):
        """Fast autoconvolution computation using Numba with optimized memory access"""
        n = len(f_vals)
        if n == 0:
            return np.array([])
        
        # Convolution result has length 2*n-1
        g_len = 2 * n - 1
        g = np.zeros(g_len, dtype=np.float64)
        
        # Optimized convolution using contiguous memory access
        for i in range(n):
            f_i = f_vals[i]
            # Pre-compute the index range to avoid repeated calculations
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_len:
                    g[idx] += f_i * f_vals[j]
        
        # Trim to center portion (length n-1) - this is the actual autoconvolution
        offset = (n - 1) // 2
        # Create a view of the trimmed array to reduce copying
        g_trimmed = g[offset:(2*n-1)-offset]
        return g_trimmed

    @staticmethod
    @jit(nopython=True)
    def compute_c2_fast(g_vals, step_width: float):
        """Fast C2 computation with numerical stability and optimized integration"""
        if len(g_vals) == 0:
            return 0.0
        
        # Compute norms with optimized accumulation
        g_l2_sq = 0.0
        g_l1 = 0.0
        g_max = 0.0
        
        # For L2 norm squared using piecewise quadratic integration (more accurate trapezoidal-like)
        # Using (h/3)(y1² + y1y2 + y2²) for each adjacent pair
        for i in range(len(g_vals) - 1):
            val1 = g_vals[i]
            val2 = g_vals[i+1]
            g_l2_sq += (step_width/3) * (val1*val1 + val1*val2 + val2*val2)
        
        # For L1 norm (sum of absolute values) - optimized loop
        for i in range(len(g_vals)):
            g_l1 += abs(g_vals[i])
        
        # For infinity norm (max absolute value) - optimized loop
        for i in range(len(g_vals)):
            abs_val = abs(g_vals[i])
            if abs_val > g_max:
                g_max = abs_val
        
        # Compute C2 with robust division and better numerical handling
        safe_l1 = max(g_l1, 1e-15)
        safe_max = max(g_max, 1e-15)
        c2 = g_l2_sq / (safe_l1 * safe_max)
        
        return c2

    def evaluate(self, f_vals: List[float]) -> float:
        """Evaluate C2 for given step function values with improved error handling"""
        try:
            # Clip negative values efficiently
            f_vals = np.clip(f_vals, 0, None)
            
            # Compute autoconvolution
            g_vals = self.compute_autoconvolution_fast(f_vals)
            
            # Compute C2
            c2 = self.compute_c2_fast(g_vals, self.step_width)
            
            return float(c2)
        except Exception as e:
            # Return minimum value on failure
            return 0.0

class MultiScaleInitializer:
    """Generates sophisticated initial configurations for optimization with enhanced patterns"""
    
    @staticmethod
    def generate(dim: int, seed: int = 42) -> List[float]:
        """Generate initial function with enhanced multi-scale patterns for better exploration"""
        np.random.seed(seed)
        
        # Create a combination of different patterns with better mathematical foundation
        init_params = np.zeros(dim, dtype=np.float64)
        
        # Scale 1: Enhanced Gaussian pattern with sharper central peak
        center = dim // 2
        sigma = dim / 10  # Narrower for sharper peak
        for i in range(dim):
            init_params[i] += 2.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)
        
        # Scale 2: Multi-frequency sinusoidal modulation with varying amplitudes
        for i in range(dim):
            # Multiple frequency components
            freq1 = 4 * np.pi * i / (dim / 4)
            freq2 = 8 * np.pi * i / (dim / 2)
            init_params[i] += 0.6 * np.sin(freq1) + 0.4 * np.cos(freq2)
        
        # Scale 3: Controlled random component with low amplitude
        rand_component = np.random.random(dim, dtype=np.float64) * 0.2
        init_params += rand_component
        
        # Scale 4: Structured alternating pattern with enhanced contrast
        # Create strong alternating high-low regions
        segment_size = max(1, dim // 15)
        for i in range(0, dim, segment_size):
            end_idx = min(i + segment_size, dim)
            if (i // segment_size) % 2 == 0:
                # High regions with moderate variation
                init_params[i:end_idx] += 0.3 * np.random.random(end_idx - i)
            else:
                # Low regions
                init_params[i:end_idx] += 0.05 * np.random.random(end_idx - i)
        
        # Scale 5: Add boundary effects for better convolution behavior
        # Create gradual tapering at edges
        edge_width = min(50, dim // 10)
        for i in range(edge_width):
            taper_factor = i / edge_width
            init_params[i] *= (1 - taper_factor * 0.5)
            init_params[dim - 1 - i] *= (1 - taper_factor * 0.5)
        
        # Ensure non-negative values
        init_params = np.maximum(init_params, 0)
        
        # Normalize to reasonable range with better scaling
        max_val = np.max(init_params)
        if max_val > 0:
            init_params = init_params / max_val * 1.8
        
        return init_params.tolist()

class EvolutionaryOptimizer:
    """Handles evolutionary optimization with advanced adaptive strategies"""
    
    def __init__(self, evaluator: AutoconvolutionEvaluator, max_time: float = 80):
        self.evaluator = evaluator
        self.max_time = max_time
    
    def adaptive_differential_evolution(self, dim: int) -> List[float]:
        """Run differential evolution with advanced adaptive population sizing and strategies"""
        start_time = time.time()
        
        # Adaptive population sizing based on problem dimension and expected performance
        base_popsize = max(10, dim // 12)
        popsize = min(35, base_popsize + np.random.randint(-3, 4))
        
        # Create bounds for parameters (0 to 10 for reasonable values)
        bounds = [(0, 10) for _ in range(dim)]
        
        # Advanced strategy selection for DE
        strategies = ['best1bin', 'best2bin', 'rand1bin']
        selected_strategy = random.choice(strategies)
        
        # Initial optimization with adaptive population
        try:
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=min(100, 2000//popsize),
                popsize=popsize,
                seed=42,
                strategy=selected_strategy,
                disp=False
            )
            
            if not result.success:
                raise Exception("Differential evolution failed")
            
            best_x = result.x
            best_c2 = -self._objective_function(best_x)
            
            # Adaptive refinement if initial progress is promising
            if best_c2 > 0.90:  # If already quite good, refine further
                # Try increased population size for better exploitation
                refined_popsize = min(40, max(popsize, 25) + 5)
                try:
                    result = differential_evolution(
                        self._objective_function,
                        bounds,
                        maxiter=min(60, 2000//refined_popsize),
                        popsize=refined_popsize,
                        seed=42,
                        strategy=selected_strategy,
                        disp=False
                    )
                    
                    if result.success:
                        final_x = result.x
                        final_c2 = -self._objective_function(final_x)
                        best_x = final_x if final_c2 > best_c2 else best_x
                        best_c2 = max(best_c2, final_c2)
                except:
                    pass
                    
            # Return the optimized parameters
            return best_x.tolist()
            
        except Exception as e:
            # Fallback to sophisticated initialization if something goes wrong
            return MultiScaleInitializer.generate(dim, 42)
    
    def _objective_function(self, params):
        """Objective function to minimize (negative C2) with enhanced error handling"""
        try:
            # Clip negative values efficiently
            f_vals = np.clip(params, 0, None)
            
            # Compute C2 value efficiently
            c2 = self.evaluator.evaluate(f_vals)
            
            # Return negative because we're minimizing
            return -float(c2)
        except Exception as e:
            # Large penalty for invalid results
            return 1e10

class GradientRefiner:
    """Applies advanced gradient-based refinement strategies using JAX"""
    
    def __init__(self, evaluator: AutoconvolutionEvaluator):
        self.evaluator = evaluator
    
    def advanced_refinement(self, initial_params: List[float], max_iterations: int = 150) -> List[float]:
        """Apply advanced refinement with JAX gradient-based optimization"""
        # Convert to jax array for efficient gradient computations
        params = jnp.array(initial_params, dtype=jnp.float64)
        
        # Define gradient-aware objective using the evaluator for consistency
        def jax_objective(params):
            f_vals_np = np.array(params, dtype=np.float64)
            f_vals_np = np.clip(f_vals_np, 0, None)
            c2 = self.evaluator.evaluate(f_vals_np)
            return -c2
        
        # Use JAX automatic differentiation for precise gradients
        grad_fn = jax.grad(jax_objective)
        
        # Apply gradient-based refinement with adaptive parameters
        learning_rate = 0.1
        momentum = 0.92  # Slightly higher momentum for stable convergence
        velocity = jnp.zeros_like(params, dtype=jnp.float64)
        
        # Run optimization with adaptive learning rate adjustments
        for i in range(max_iterations):
            # Compute gradient
            grad_val = grad_fn(params)
            
            # Update with momentum
            velocity = momentum * velocity - learning_rate * grad_val
            params = params + velocity
            
            # Clip to non-negative values
            params = jnp.maximum(params, 0)
            
            # Adaptive learning rate adjustment
            if i > 50 and i % 20 == 0:
                current_c2 = -jax_objective(params)
                if current_c2 > 0.92:  # If approaching good solution, reduce learning rate
                    learning_rate *= 0.92
            
            # Early exit if time limit approached
            if time.time() - start_time > 85:
                break
        
        return np.array(params).tolist()

class StochasticPerturber:
    """Adds advanced stochastic perturbation to prevent premature convergence"""
    
    @staticmethod
    def perturb(params: List[float], strength: float = 0.05) -> List[float]:
        """Add advanced stochastic perturbation to parameters"""
        # Add small random noise with controlled variance
        noise = np.random.normal(0, strength, len(params))
        perturbed = np.array(params, dtype=np.float64) + noise
        
        # Ensure non-negativity
        perturbed = np.clip(perturbed, 0, None)
        
        # Preserve total mass by normalizing
        total_mass = np.sum(params)
        if total_mass > 0:
            perturbed = perturbed / np.sum(perturbed) * total_mass
        
        return perturbed.tolist()

class ModularizedOptimizer:
    """Main optimization controller that orchestrates all components with performance enhancements"""
    
    def __init__(self, n_steps: int = 500, max_time: float = 90.0):
        self.n_steps = n_steps
        self.max_time = max_time
        self.evaluator = AutoconvolutionEvaluator()
        self.initializer = MultiScaleInitializer()
        self.evolutionary_optimizer = EvolutionaryOptimizer(self.evaluator, max_time)
        self.gradient_refiner = GradientRefiner(self.evaluator)
        self.perturber = StochasticPerturber()
    
    def optimize(self) -> List[float]:
        """Main optimization pipeline with enhanced performance strategies"""
        global start_time
        start_time = time.time()
        
        best_c2 = -np.inf
        best_params = None
        
        # Enhanced multi-start approach with optimized dimensions and strategies
        # Try more varied dimensions to cover different regimes
        dimensions = [250, 350, 500, 700, 900, 1100]
        strategy_configs = [
            {"popsize": 12, "iterations": 50, "refinement": True},
            {"popsize": 15, "iterations": 60, "refinement": True},
            {"popsize": 18, "iterations": 70, "refinement": False}
        ]
        
        for dim_idx, dim in enumerate(dimensions):
            if time.time() - start_time > 85:
                break
                
            try:
                # Strategy 1: Adaptive differential evolution with multiple configurations
                for config in strategy_configs:
                    if time.time() - start_time > 85:
                        break
                        
                    # Adjust parameters based on dimension and iteration
                    effective_popsize = min(25, max(10, dim // 10))
                    max_iter = min(100, 2000 // effective_popsize)
                    
                    # Use adaptive population sizing based on dimension
                    params = self.evolutionary_optimizer.adaptive_differential_evolution(dim)
                    
                    # Apply gradient refinement if requested
                    if config["refinement"]:
                        refined_params = self.gradient_refiner.advanced_refinement(params, max_iter // 2)
                    else:
                        refined_params = params
                    
                    # Stochastic perturbation to escape local optima
                    perturbed_params = self.perturber.perturb(refined_params, 0.03)
                    
                    # Compute actual C2 value
                    c2 = self.evaluator.evaluate(perturbed_params)
                    
                    if c2 > best_c2:
                        best_c2 = c2
                        best_params = perturbed_params.copy()
                    
            except Exception as e:
                continue
        
        # If no valid parameters found, return default
        if best_params is None:
            return [0.5] * 100
        
        # Final check and conversion to list
        return best_params

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    optimizer = ModularizedOptimizer(n_steps=500, max_time=90.0)
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")