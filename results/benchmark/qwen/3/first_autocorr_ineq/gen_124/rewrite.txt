# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy.signal import fftconvolve
import time
from functools import partial

# Set up JAX for GPU acceleration if available
jax.config.update('jax_enable_x64', True)

def convolve_fft_jax(seq):
    """Compute convolution using FFT in JAX for better performance."""
    n = len(seq)
    # Pad to next power of 2 for efficient FFT
    padded_len = 1 << (n - 1).bit_length()
    padded_seq = jnp.pad(seq, (0, padded_len - n), 'constant')

    # FFT-based convolution
    fft_seq = jnp.fft.fft(padded_seq)
    conv_fft = fft_seq * jnp.conjugate(fft_seq)
    conv_result = jnp.fft.ifft(conv_fft).real[:2*n-1]

    return conv_result

@jit
def compute_autocorrelation_constant_jax(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence using JAX.
    """
    if len(sequence) == 0:
        return jnp.inf, 0.0

    a = jnp.array(sequence, dtype=jnp.float64)
    n = len(a)

    # Compute convolution using FFT for efficiency
    conv = convolve_fft_jax(a)
    max_b = jnp.max(conv)
    sum_a = jnp.sum(a)

    # Avoid division by zero or near-zero sums
    sum_a = jnp.where(sum_a < 0.01, 0.01, sum_a)

    C1 = 2 * n * max_b / (sum_a ** 2)
    inv_C1 = 1 / C1

    return C1, inv_C1

@partial(jit, static_argnums=(1,))
def compute_objective_and_grad(sequence, n_steps):
    """Compute objective and its gradient for optimization."""
    # Ensure sequence has correct shape
    sequence = jnp.reshape(sequence, (n_steps,))
    
    # Compute negative inverse C1 (since we maximize 1/C1)
    _, inv_C1 = compute_autocorrelation_constant_jax(sequence)
    objective = -inv_C1
    
    # Compute gradient
    grad_func = grad(lambda x: compute_autocorrelation_constant_jax(x)[1])
    grad_val = grad_func(sequence)
    
    return objective, grad_val

def optimize_sequence(initial_sequence, max_iter=100):
    """Optimize sequence using gradient descent with adaptive learning rates."""
    n_steps = len(initial_sequence)
    sequence = jnp.array(initial_sequence, dtype=jnp.float64)
    
    # Initialize Adam optimizer parameters
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    m = jnp.zeros_like(sequence)  # First moment
    v = jnp.zeros_like(sequence)  # Second moment
    
    best_sequence = sequence
    best_inv_C1 = -float('inf')
    
    for i in range(max_iter):
        # Compute objective and gradient
        obj, grad_val = compute_objective_and_grad(sequence, n_steps)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad_val
        v = beta2 * v + (1 - beta2) * grad_val ** 2
        
        # Bias correction
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        
        # Update sequence
        sequence = sequence + learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Clip values to [0, 1000] for practicality
        sequence = jnp.clip(sequence, 0, 1000)
        
        # Track best solution
        current_inv_C1 = -obj
        best_sequence = jnp.where(current_inv_C1 > best_inv_C1, sequence, best_sequence)
        best_inv_C1 = jnp.maximum(current_inv_C1, best_inv_C1)
        
        # Early stopping for convergence
        if i > 10 and jnp.allclose(sequence, best_sequence, atol=1e-6):
            break
            
    return np.array(best_sequence)

def search_for_best_sequence():
    """Main function to find the best coefficient sequence using gradient descent."""
    # Initialize with deterministic seed for reproducibility
    np.random.seed(42)
    
    # Try multiple initialization strategies
    best_sequence = []
    best_inv_C1 = -float('inf')
    
    for _ in range(10):
        # Different initialization strategies
        strategy = np.random.choice(['random', 'geometric', 'spike'])

        if strategy == 'random':
            n = np.random.randint(100, 1000)
            sequence = np.random.uniform(0.1, 1.0, n)
        elif strategy == 'geometric':
            n = np.random.randint(100, 1000)
            sequence = np.array([0.9 ** i for i in range(n)])
        else:  # spike
            n = np.random.randint(100, 1000)
            sequence = np.zeros(n)
            spike_idx = np.random.randint(0, n)
            sequence[spike_idx] = 1.0

        # Optimize the sequence
        optimized_sequence = optimize_sequence(sequence, max_iter=50)
        
        # Evaluate the optimized sequence
        _, inv_C1 = compute_autocorrelation_constant_jax(optimized_sequence)
        
        # Keep the best solution
        if inv_C1 > best_inv_C1:
            best_inv_C1 = inv_C1
            best_sequence = optimized_sequence
    
    # Final optimization with the best found sequence
    if best_sequence.size > 0:
        final_sequence = optimize_sequence(best_sequence, max_iter=100)
        _, inv_C1 = compute_autocorrelation_constant_jax(final_sequence)
        if inv_C1 > best_inv_C1:
            return final_sequence.tolist()
    
    # Return the best sequence found or default
    if best_sequence.size == 0:
        return [1.0] * 100
    return best_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")