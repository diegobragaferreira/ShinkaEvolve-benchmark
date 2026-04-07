# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.signal import convolve as jax_convolve
import time

# Set random seed for reproducibility
np.random.seed(42)
jax.config.update('jax_enable_x64', True)

@jax.jit
def compute_convolution(seq):
    """Compute the autoconvolution using JAX."""
    seq = jnp.array(seq)
    conv = jax_convolve(seq, seq, mode='full')
    return conv

@jax.jit
def compute_c1(seq):
    """Compute the C1 constant from the sequence."""
    seq = jnp.array(seq)
    n = len(seq)
    conv = compute_convolution(seq)
    max_conv = jnp.max(conv)
    sum_seq = jnp.sum(seq)
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

@jax.jit
def inv_c1_fn(seq):
    """Compute the inverse of C1 (the objective to maximize)."""
    seq = jnp.array(seq)
    n = len(seq)
    conv = compute_convolution(seq)
    max_conv = jnp.max(conv)
    sum_seq = jnp.sum(seq)
    inv_c1 = (sum_seq ** 2) / (2 * n * max_conv)
    return inv_c1

@jax.jit
def gradient_inv_c1(seq):
    """Compute the gradient of inverse C1."""
    seq = jnp.array(seq)
    n = len(seq)
    conv = compute_convolution(seq)
    max_conv = jnp.max(conv)
    sum_seq = jnp.sum(seq)
    
    # Compute gradient numerically with autodiff
    grad_fn = grad(inv_c1_fn)
    return grad_fn(seq)

def update_sequence_with_momentum(current_seq, grad_seq, momentum=0.9, lr=0.01):
    """Update sequence using momentum-based gradient descent."""
    new_seq = current_seq - lr * grad_seq
    # Ensure non-negativity
    new_seq = jnp.maximum(new_seq, 0.0)
    return new_seq

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Optimize the sequence using gradient ascent."""
    if len(sequence) < 1:
        return None
        
    # Clip sequence values to [0, 1000]
    sequence = np.clip(sequence, 0, 1000)
    
    # Convert to JAX array
    seq_array = jnp.array(sequence, dtype=jnp.float64)
    
    # Compute gradient and update
    try:
        grad_inv_c1_val = gradient_inv_c1(seq_array)
        updated_seq = update_sequence_with_momentum(seq_array, grad_inv_c1_val, lr=0.01)
        
        # Convert back to Python list
        updated_seq = np.array(updated_seq)
        
        # Ensure sum is not too small
        if np.sum(updated_seq) < 0.01:
            return None
            
        return updated_seq.tolist()
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initialize with random sequence
    n = np.random.randint(100, 1000)
    sequence = np.random.rand(n).tolist()
    
    # Apply gradient ascent for several iterations
    for _ in range(50):  # Limit iterations to stay within time budget
        next_seq = get_good_direction_to_move_into(sequence)
        if next_seq is not None:
            sequence = next_seq
        else:
            # If gradient update fails, add noise and continue
            sequence = [(x + np.random.rand() * 0.1) % 1000 for x in sequence]
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
