# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from scipy.optimize import differential_evolution
import random

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    if n == 0:
        return np.array([])

    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    # Using a more optimized approach with proper indexing
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_i * f_vals[j]

    # Trim to center portion (length n-1) - this is the actual autoconvolution
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]

    return g_trimmed

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared (trapezoidal integration)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        # Trapezoidal rule: (h/2)*(y1 + y2) but we square for L2 norm
        # Using piecewise quadratic approximation instead (more accurate)
        h = STEP_WIDTH
        g_l2_sq += (h/3) * (val1*val1 + val1*val2 + val2*val2)

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute C2 with robust division
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

class OptimizationPipeline:
    """Manages the complete optimization process in a modular fashion"""
    
    def __init__(self):
        self.best_c2 = -np.inf
        self.best_params = None
        self.start_time = None
        
    def initialize_parameters(self, dim, strategy="multiscale"):
        """Generate initial parameters using specified strategy"""
        if strategy == "multiscale":
            return self._generate_multi_scale_initialization(dim)
        else:
            return self._generate_simple_initialization(dim)
    
    def _generate_multi_scale_initialization(self, dim):
        """Generate initial function with multiple scales for better exploration"""
        # Create a combination of different patterns
        init_params = np.zeros(dim)
        
        # Scale 1: Centered Gaussian pattern
        center = dim // 2
        sigma = dim / 6
        for i in range(dim):
            init_params[i] += 1.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)
        
        # Scale 2: Sinusoidal modulation
        for i in range(dim):
            init_params[i] += 0.3 * np.sin(2 * np.pi * i / (dim / 4))
        
        # Scale 3: Random component
        np.random.seed(42)
        rand_component = np.random.random(dim) * 0.2
        init_params += rand_component
        
        # Ensure non-negative values
        init_params = np.maximum(init_params, 0)
        
        # Normalize to reasonable range
        max_val = np.max(init_params)
        if max_val > 0:
            init_params = init_params / max_val * 1.5
        
        return init_params.tolist()
    
    def _generate_simple_initialization(self, dim):
        """Simple initialization with uniform distribution"""
        return [1.0] * dim
    
    def optimize_with_de(self, params, dim):
        """Run differential evolution optimization"""
        bounds = [(0, 10) for _ in range(dim)]
        
        try:
            # Adaptive population sizing
            popsize = min(20, max(10, dim // 10))
            
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=min(100, 2000//popsize),
                popsize=popsize,
                seed=42,
                strategy='best1bin',
                disp=False
            )
            
            if result.success:
                return result.x
            else:
                return params
                
        except Exception:
            return params
    
    def objective_function(self, params):
        """Objective function to minimize (negative C2)"""
        try:
            # Clip negative values
            f_vals = np.clip(params, 0, None)
            
            # Compute autoconvolution
            g_vals = compute_autoconvolution_numba(f_vals)
            
            # Compute C2
            c2 = compute_c2_numba(g_vals)
            
            # Return negative because we're minimizing
            return -c2
        except Exception as e:
            return 1e10  # Large penalty for invalid results
    
    def refine_with_gradient(self, params):
        """Apply gradient-based refinement"""
        try:
            # Convert to jax array for gradient computations
            params_array = jnp.array(params)
            
            # Define gradient-aware objective
            def jax_objective(params):
                f_vals_np = np.array(params)
                f_vals_np = np.clip(f_vals_np, 0, None)
                g_vals = compute_autoconvolution_numba(f_vals_np)
                c2 = compute_c2_numba(g_vals)
                return -c2
            
            # Use automatic differentiation with jax
            grad_fn = jax.grad(jax_objective)
            
            # Apply gradient-based refinement with adaptive learning rate
            learning_rate = 0.05
            momentum = 0.9
            velocity = jnp.zeros_like(params_array)
            
            # Run optimization for a few iterations
            for i in range(100):  # Reduced iterations for speed
                # Compute gradient
                grad_val = grad_fn(params_array)
                
                # Update with momentum
                velocity = momentum * velocity - learning_rate * grad_val
                params_array = params_array + velocity
                
                # Clip to non-negative
                params_array = jnp.maximum(params_array, 0)
                
                # Early stopping condition
                if i > 50 and i % 20 == 0:
                    current_c2 = -jax_objective(params_array)
                    if current_c2 > 0.95:  # If close to good solution, slow down
                        learning_rate *= 0.9
                
                # Time check
                if time.time() - self.start_time > 85:
                    break
            
            return np.array(params_array)
            
        except Exception:
            return params
    
    def add_stochastic_perturbation(self, params, perturbation_strength=0.03):
        """Add stochastic perturbation to prevent premature convergence"""
        try:
            # Add small random noise to parameters
            noise = np.random.normal(0, perturbation_strength, len(params))
            perturbed = params + noise
            
            # Ensure non-negativity and normalize
            perturbed = np.clip(perturbed, 0, None)
            
            # Normalize to preserve relative proportions
            max_val = np.max(perturbed)
            if max_val > 0:
                perturbed = perturbed / max_val * np.max(params)
            
            return perturbed
        except Exception:
            return params
    
    def evaluate_solution(self, params):
        """Evaluate the quality of a solution"""
        try:
            f_vals = np.clip(params, 0, None)
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)
                return c2
        except Exception:
            return 0.0
        return 0.0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    pipeline = OptimizationPipeline()
    pipeline.start_time = time.time()
    
    # Multi-start approach with different dimensions and strategies
    dimensions = [300, 500, 700, 1000]
    
    for dim in dimensions:
        if time.time() - pipeline.start_time > 85:
            break
            
        try:
            # Strategy 1: Generate initial parameters
            np.random.seed(42)
            params = pipeline.initialize_parameters(dim, "multiscale")
            
            # Strategy 2: Optimize with differential evolution
            optimized_params = pipeline.optimize_with_de(params, dim)
            
            # Strategy 3: Refinement with gradient-based optimization
            refined_params = pipeline.refine_with_gradient(optimized_params)
            
            # Strategy 4: Stochastic perturbation to escape local optima
            perturbed_params = pipeline.add_stochastic_perturbation(refined_params, 0.03)
            
            # Strategy 5: Evaluate and compare
            c2 = pipeline.evaluate_solution(perturbed_params)
            
            if c2 > pipeline.best_c2:
                pipeline.best_c2 = c2
                pipeline.best_params = perturbed_params.copy()
                
        except Exception as e:
            continue
    
    # If no valid parameters found, return default
    if pipeline.best_params is None:
        return [0.5] * 100
    
    # Final check and conversion to list
    final_f_vals = np.clip(pipeline.best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")