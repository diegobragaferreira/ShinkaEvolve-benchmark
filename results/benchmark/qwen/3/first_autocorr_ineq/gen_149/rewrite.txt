# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.signal import convolve as jax_convolve
import time
import random

# Use jax for automatic differentiation and fast computations
jax.config.update("jax_enable_x64", True)

def compute_c1_constant_jax(sequence):
    """Compute C1 constant using JAX for automatic differentiation."""
    a = jnp.array(sequence)
    n = len(sequence)
    
    # Use JAX convolution for efficiency and differentiability
    b = jax_convolve(a, a, 'full')
    max_conv = jnp.max(b)
    sum_a = jnp.sum(a)
    
    # Avoid division by zero or very small values
    sum_a = jnp.where(sum_a < 1e-10, 1e-10, sum_a)
    
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def compute_inv_c1_jax(sequence):
    """Compute 1/C1 constant for maximization."""
    c1 = compute_c1_constant_jax(sequence)
    return 1.0 / c1

# Gradient computation using JAX
compute_inv_c1_grad = grad(compute_inv_c1_jax)

@jit
def update_sequence_jit(sequence, grad_vals, learning_rate, momentum=0.9):
    """Update sequence using gradient ascent with momentum."""
    # Clip gradients to prevent exploding gradients
    grad_vals = jnp.clip(grad_vals, -100.0, 100.0)
    # Update with momentum
    new_sequence = sequence + learning_rate * grad_vals
    # Ensure non-negativity
    new_sequence = jnp.maximum(new_sequence, 0.0)
    return new_sequence

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence using gradient ascent."""
    np.random.seed(42)
    random.seed(42)
    
    # Generate initial sequence
    n = random.randint(100, 1000)
    sequence = np.random.uniform(0, 1000, n)
    
    # Convert to JAX array for computation
    sequence = jnp.array(sequence)
    
    # Optimization parameters
    learning_rate = 0.01
    max_iter = 1000
    tolerance = 1e-6
    momentum = 0.9
    
    # Store best sequence
    best_sequence = sequence
    best_inv_c1 = compute_inv_c1_jax(sequence)
    
    # Initialize velocity for momentum
    velocity = jnp.zeros_like(sequence)
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > 180:
            break
            
        # Compute gradient
        grad_vals = compute_inv_c1_grad(sequence)
        
        # Update with momentum
        velocity = momentum * velocity + (1 - momentum) * grad_vals
        sequence = sequence + learning_rate * velocity
        
        # Ensure non-negativity and clipping
        sequence = jnp.maximum(sequence, 0.0)
        sequence = jnp.clip(sequence, 0, 1000)
        
        # Evaluate current sequence
        current_inv_c1 = compute_inv_c1_jax(sequence)
        
        # Update best if current is better
        if current_inv_c1 > best_inv_c1:
            best_inv_c1 = current_inv_c1
            best_sequence = sequence
        
        # Early stopping if improvement is negligible
        if i > 10 and abs(current_inv_c1 - compute_inv_c1_jax(sequence - learning_rate * grad_vals)) < tolerance:
            break
            
    # Return the best sequence as a list
    return best_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")