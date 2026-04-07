# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.signal import fftconvolve

# Set random seed for reproducibility
np.random.seed(42)
jax.config.update('jax_enable_x64', True)

@jit
def convolve_jax(a, b):
    """JAX-based convolution using FFT for efficiency"""
    n = len(a)
    # Use scipy's fftconvolve for compatibility with fft-based approach
    conv = fftconvolve(a, b, mode='full')
    return conv[:2*n - 1]

@jit
def compute_c1_jax(sequence):
    """Compute C1 constant using JAX for automatic differentiation"""
    if len(sequence) == 0:
        return jnp.inf
    
    # Compute autoconvolution using JAX-compatible FFT
    autoconv = convolve_jax(sequence, sequence)
    
    # Extract max value
    max_b = jnp.max(autoconv)
    
    # Sum of sequence
    sum_a = jnp.sum(sequence)
    
    # Avoid division by zero
    sum_a = jnp.where(sum_a < 1e-10, 1e-10, sum_a)
    
    # Compute C1
    c1 = 2 * len(sequence) * max_b / (sum_a ** 2)
    
    return c1

@jit
def inv_c1_jax(sequence):
    """Compute inverse of C1 constant for maximization"""
    c1 = compute_c1_jax(sequence)
    return 1.0 / jnp.where(c1 < 1e-10, 1e10, c1)

@jit
def update_sequence(sequence, learning_rate=0.01):
    """Update sequence using gradient ascent on inverse C1"""
    # Compute gradient w.r.t. sequence
    grad_inv_c1 = grad(inv_c1_jax)(sequence)
    
    # Update sequence using gradient ascent
    new_sequence = sequence + learning_rate * grad_inv_c1
    
    # Ensure non-negative values (projection onto feasible set)
    new_sequence = jnp.maximum(new_sequence, 0.0)
    
    # Normalize to prevent overflow (optional)
    sum_seq = jnp.sum(new_sequence)
    new_sequence = jnp.where(sum_seq < 1e-10, 
                            new_sequence, 
                            new_sequence * jnp.sum(sequence) / sum_seq)
    
    return new_sequence

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """
    Returns the direction to move into the sequence using gradient-based optimization.
    """
    # Convert to JAX array
    sequence = jnp.array(sequence, dtype=jnp.float64)
    
    # Ensure sequence has a meaningful sum
    sum_seq = jnp.sum(sequence)
    sequence = jnp.where(sum_seq < 1e-10, 
                         jnp.ones_like(sequence), 
                         sequence)
    
    # Perform several gradient updates
    num_updates = 50
    for _ in range(num_updates):
        sequence = update_sequence(sequence, learning_rate=0.01)
    
    # Return as Python list
    return sequence.tolist()

def search_for_best_sequence() -> list[float]:
    """
    Function to search for the best coefficient sequence.
    Uses a random starting point and improves it using gradient descent.
    """
    # Generate random sequence of length between 100 and 1000
    n = np.random.randint(100, 1000)
    best_sequence = np.random.rand(n).tolist()
    
    # Improve the sequence using gradient-based optimization
    improved_sequence = get_good_direction_to_move_into(best_sequence)
    
    return improved_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
