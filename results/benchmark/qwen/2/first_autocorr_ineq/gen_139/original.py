# EVOLVE-BLOCK-START

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit
def compute_convolution_jax(seq):
    """Compute convolution using JAX for efficient gradient calculations."""
    n = seq.shape[0]
    # Pad to size 2*n-1 for full convolution
    padded_seq = jnp.pad(seq, (0, n-1), mode='constant')
    # FFT convolution
    fft_seq = jnp.fft.fft(padded_seq)
    conv_result = jnp.fft.ifft(fft_seq * jnp.conj(fft_seq)).real
    # Return only the relevant part (first 2*n-1 elements)
    return conv_result[:2*n-1]

@jit
def compute_autocorrelation_constant_jax(sequence):
    """Compute the autocorrelation constant C1 for a sequence using JAX."""
    n = sequence.shape[0]
    if n == 0:
        return jnp.inf
    
    # Skip if sum is too small to avoid numerical issues
    sum_a = jnp.sum(sequence)
    if sum_a < 0.01:
        return jnp.inf  # Reject invalid sequences
    
    # Use JAX FFT for fast convolution
    conv_result = compute_convolution_jax(sequence)
    max_autocorr = jnp.max(conv_result)
    
    # Calculate C1 = 2n * max(b) / (sum(a))^2
    c1 = (2 * n * max_autocorr) / (sum_a ** 2)
    return c1

@jit
def compute_inverse_c1_jax(sequence):
    """Compute the inverse of C1 (our objective to maximize) using JAX."""
    c1 = compute_autocorrelation_constant_jax(sequence)
    return jnp.where(jnp.isinf(c1), 0.0, 1.0 / c1)

@jit
def gradient_of_inverse_c1(sequence):
    """Compute the gradient of 1/C1 with respect to sequence elements."""
    # Using JAX's automatic differentiation
    grad_func = grad(compute_inverse_c1_jax)
    return grad_func(sequence)

def compute_autocorrelation_constant(sequence):
    """Wrapper for numpy-based computation."""
    n = len(sequence)
    if n == 0:
        return float('inf')
    
    a = np.array(sequence, dtype=float)
    
    # Skip if sum is too small to avoid numerical issues
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')  # Reject invalid sequences
    
    # Use FFT for fast convolution
    padded_length = 2 * n - 1
    a_padded = np.pad(a, (0, padded_length - n), mode='constant')
    a_fft = np.fft.fft(a_padded)
    autocorr_fft = a_fft * np.conj(a_fft)
    autocorr = np.fft.ifft(autocorr_fft).real
    
    # Take only the first n elements for autocorrelation
    autocorr = autocorr[:n]
    
    # Find maximum in the autocorrelation
    max_autocorr = np.max(autocorr)
    
    # Calculate C1 = 2n * max(b) / (sum(a))^2
    c1 = (2 * n * max_autocorr) / (sum_a ** 2)
    return c1

def compute_inverse_c1(sequence):
    """Compute the inverse of C1 (our objective to maximize)."""
    try:
        c1 = compute_autocorrelation_constant(sequence)
        if c1 == float('inf'):
            return 0.0  # Penalty for invalid sequences
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_random_sequence(length=None, min_length=100, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)

    # Generate random heights between 0 and 1000
    sequence = [random.uniform(0, 1000) for _ in range(length)]

    # Ensure at least one element is positive
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0

    return sequence

def gradient_ascent_step(sequence, step_size=0.01):
    """Perform a gradient ascent step to improve the sequence."""
    sequence_array = np.array(sequence, dtype=float)
    
    # Compute gradient of inverse C1
    try:
        grad_vals = gradient_of_inverse_c1(sequence_array)
        # Update sequence with gradient ascent
        updated_sequence = sequence_array + step_size * grad_vals
        # Ensure non-negativity
        updated_sequence = np.maximum(updated_sequence, 0.0)
        # Normalize if necessary
        sum_updated = np.sum(updated_sequence)
        if sum_updated > 0.01:
            updated_sequence = updated_sequence / sum_updated
        else:
            # Re-normalize with a minimal correction
            updated_sequence = updated_sequence + 0.01
            updated_sequence = updated_sequence / np.sum(updated_sequence)
        return updated_sequence.tolist()
    except Exception:
        # Fallback to simple random perturbation if gradient computation fails
        new_sequence = sequence.copy()
        for i in range(len(new_sequence)):
            if random.random() < 0.1:
                new_sequence[i] = max(0, new_sequence[i] + random.uniform(-10, 10))
        return new_sequence

def search_for_best_sequence():
    """Main search function using gradient ascent optimization."""
    # Start with a random sequence
    initial_sequence = generate_random_sequence()
    
    current_sequence = initial_sequence.copy()
    best_sequence = current_sequence.copy()
    best_inv_c1 = compute_inverse_c1(current_sequence)
    
    start_time = time.time()
    max_time_seconds = 180
    
    iteration = 0
    while time.time() - start_time < max_time_seconds:
        iteration += 1
        # Perform gradient ascent step
        new_sequence = gradient_ascent_step(current_sequence, step_size=0.01)
        
        # Evaluate the new sequence
        new_inv_c1 = compute_inverse_c1(new_sequence)
        
        # Accept the new sequence if it improves the objective
        if new_inv_c1 > best_inv_c1:
            best_inv_c1 = new_inv_c1
            best_sequence = new_sequence.copy()
            
        current_sequence = new_sequence
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
