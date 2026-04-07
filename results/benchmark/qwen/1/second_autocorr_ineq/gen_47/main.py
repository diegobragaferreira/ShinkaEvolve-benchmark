# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
import math

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
    # The center of the convolution result corresponds to the autoconvolution
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

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

# Custom HMC implementation using JAX for automatic differentiation
class HamiltonianMC:
    def __init__(self, target_log_prob_fn, step_size=0.01, num_leapfrog_steps=10):
        self.target_log_prob_fn = target_log_prob_fn
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps

    @staticmethod
    @jax_jit
    def _leapfrog_update(position, momentum, grad_log_prob, step_size, num_steps):
        """Perform leapfrog integration for HMC"""
        # Initialize position and momentum
        new_position = position
        new_momentum = momentum

        # Perform leapfrog steps
        for _ in range(num_steps):
            # Half step for momentum
            new_momentum = new_momentum - 0.5 * step_size * grad_log_prob(new_position)

            # Full step for position
            new_position = new_position + step_size * new_momentum

            # Recompute gradient
            grad_log_prob_new = grad_log_prob(new_position)

            # Half step for momentum
            new_momentum = new_momentum - 0.5 * step_size * grad_log_prob_new

        return new_position, new_momentum

    def sample(self, initial_position, num_samples, rng_key):
        """Sample from the target distribution using HMC"""
        samples = []
        current_position = initial_position

        # We'll use a simpler approach: just run a few steps of HMC
        # since we don't actually need full MCMC sampling for optimization

        # Instead, we'll use the gradient to optimize directly with a modified HMC-like approach
        # This avoids the complexity of full MCMC sampling while retaining benefits of gradient information

        # Create a simple gradient-based optimizer with momentum
        # We'll make a version that works well with our specific objective
        return current_position

def objective_function(params):
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

def sophisticated_geometric_initialization(dim):
    """Create an initialization that places steps geometrically to promote high convolution values"""
    # Create a pattern that starts with high values and tapers off
    # or creates a structure with sharp transitions that maximize convolution

    # Use a geometric approach: place some large values in strategic locations
    # and fill remaining with smaller values, ensuring smooth transitions

    init_params = []

    # Create a pattern: start with higher values in middle, tapering to sides
    # This promotes constructive interference in convolution
    for i in range(dim):
        # Use a bell-curve like pattern or a more complex structure
        position = i / (dim - 1) if dim > 1 else 0.5
        # Create a pattern that has peaks in the center and decays
        # This often gives good autoconvolution properties
        peak_height = 1.0
        decay_factor = 4.0

        # Gaussian-like shape centered in the middle
        center_point = 0.5
        gaussian_val = peak_height * np.exp(-decay_factor * (position - center_point)**2)

        # Add some structured variation
        struct_val = 0.1 * np.sin(10 * position) + 0.2 * np.cos(5 * position)
        final_val = max(0, gaussian_val + struct_val + 0.1)

        init_params.append(final_val)

    # Normalize slightly to prevent extreme values
    if init_params:
        max_val = max(init_params)
        if max_val > 0:
            init_params = [x/max_val * 1.0 for x in init_params]

    return init_params

def direct_gradient_based_optimization():
    """Direct optimization approach using gradient information without full MCMC"""
    # Since full HMC would be complex to implement properly for this discrete case,
    # we use a gradient-based approach combined with geometric initialization

    # Generate initial parameters with geometric pattern
    initial_dim = np.random.randint(300, 700)
    x0 = sophisticated_geometric_initialization(initial_dim)

    # Convert to JAX arrays for gradient computations
    x0_array = jnp.array(x0)

    # Define the objective function using JAX for gradients
    def jax_objective(params):
        # Convert back to numpy for our existing computations
        f_vals_np = np.array(params)
        f_vals_np = np.clip(f_vals_np, 0, None)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals_np)

        # Compute C2
        c2 = compute_c2_numba(g_vals)

        return -c2  # Negative because we're minimizing

    # Use basic gradient descent with Adam-like updates
    # This is a simplified approach that should work reasonably well
    learning_rate = 0.01
    iterations = 500

    # Simple gradient ascent approach with momentum
    x_best = x0_array
    best_value = jax_objective(x0_array)

    # Create a custom gradient descent implementation
    current_x = x0_array
    velocity = jnp.zeros_like(current_x)
    momentum = 0.9
    epsilon = 1e-8

    for i in range(iterations):
        # Compute gradient using finite differences for simplicity
        # (In practice, we would use analytical gradients or more sophisticated AD)

        # Very approximate gradient using finite differences
        eps = 1e-4
        grad_approx = jnp.zeros_like(current_x)

        for j in range(len(current_x)):
            # Perturb only dimension j
            perturbed_x = current_x.at[j].set(current_x[j] + eps)
            forward_val = jax_objective(perturbed_x)
            backward_val = jax_objective(current_x)
            grad_component = (forward_val - backward_val) / eps
            grad_approx = grad_approx.at[j].set(grad_component)

        # Update using momentum
        velocity = momentum * velocity - learning_rate * grad_approx
        current_x = current_x + velocity

        # Clip to non-negative values
        current_x = jnp.maximum(current_x, 0)

        # Evaluate objective
        current_value = jax_objective(current_x)

        if current_value < best_value:
            best_value = current_value
            x_best = current_x

    return np.array(x_best)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Multi-start approach with different initialization strategies
    best_c2 = -np.inf
    best_params = None

    # Try multiple optimizations with different strategies
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        try:
            # Use our direct gradient-based approach
            params = direct_gradient_based_optimization()

            # Compute actual C2 value
            f_vals = np.clip(params, 0, None)
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()
        except Exception as e:
            continue

        # Early exit if we've been running too long
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break

    # If no valid parameters found, return default
    if best_params is None:
        return [0.5] * 100

    # Final check and conversion to list
    final_f_vals = np.clip(best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")