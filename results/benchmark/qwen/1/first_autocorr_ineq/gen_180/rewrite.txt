# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
import time
import jax
from jax import grad, jit, vmap
from jax.experimental.optimizers import adam
import jax.numpy as jnp
from functools import partial

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant_jax(sequence):
    """Compute the autocorrelation constant C1 using JAX for automatic differentiation."""
    n = len(sequence)
    if n == 0:
        return float('inf')
        
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return float('inf')
    
    # Use FFT for fast convolution
    padded_seq = np.pad(sequence, (0, n-1), 'constant', constant_values=0)
    conv_result = np.real(np.fft.fft(padded_seq) * np.conj(np.fft.fft(padded_seq)))
    
    max_conv = np.max(conv_result[:2*n-1])
    
    # Compute C₁ = 2n * max(b) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    
    return c1

# JIT-compiled gradient function for faster evaluation
@jit
def compute_inv_c1_grad(sequence):
    """Computes the gradient of 1/C1 with respect to the sequence elements."""
    # Convert to JAX array for automatic differentiation
    seq = jnp.array(sequence, dtype=jnp.float32)
    n = len(sequence)
    
    # Compute convolution using FFT
    padded_seq = jnp.pad(seq, (0, n-1), 'constant', constant_values=0)
    conv_result = jnp.real(jnp.fft.fft(padded_seq) * jnp.conj(jnp.fft.fft(padded_seq)))
    
    # Extract valid convolution part
    conv_valid = conv_result[:2*n-1]
    max_conv = jnp.max(conv_valid)
    sum_seq = jnp.sum(seq)
    
    # Avoid division by zero
    sum_sq = jnp.maximum(sum_seq ** 2, 1e-10)
    
    # Compute 1/C1 = (sum(a))^2 / (2n * max(conv))
    inv_c1 = sum_sq / (2 * n * max_conv)
    
    # Return gradient of 1/C1
    return grad(lambda s: jnp.sum(s)**2 / (2 * n * jnp.max(jnp.convolve(s, s, mode='full')[:2*n-1])))(seq)

def compute_fitness_jax(sequence):
    """Evaluate fitness (inverse of C₁) using JAX."""
    c1 = compute_autocorrelation_constant_jax(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def generate_initial_sequence(length):
    """Generate a structured initial sequence."""
    # Use a combination of step-like and harmonic patterns
    sequence = []
    for i in range(length):
        # Step pattern with decreasing heights
        step_height = max(0.01, 100 * np.exp(-i * 0.01))
        # Add harmonic modulation
        harmonic = 10 * np.sin(i * 0.1) + 5 * np.cos(i * 0.05)
        sequence.append(max(0.01, step_height + harmonic))
    
    # Normalize to ensure reasonable magnitude
    total = sum(sequence)
    if total > 0:
        sequence = [x * 100 / total for x in sequence]
    return sequence

def gradient_ascent_optimization(initial_sequence, max_iter=1000, learning_rate=0.01):
    """Optimize sequence using gradient ascent with adaptive learning rate."""
    sequence = np.array(initial_sequence, dtype=np.float32)
    n = len(sequence)
    
    # Initialize optimizer state
    opt_init, opt_update, get_params = adam(learning_rate)
    opt_state = opt_init(sequence)
    
    best_sequence = sequence.copy()
    best_fitness = compute_fitness_jax(best_sequence)
    
    for i in range(max_iter):
        # Get current parameters
        current_params = get_params(opt_state)
        
        # Compute fitness and gradient
        current_fitness = compute_fitness_jax(current_params)
        
        # If current is better, update best
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_sequence = current_params.copy()
        
        # Compute gradient
        try:
            # Use gradient descent in opposite direction since we want to maximize fitness
            grad_value = compute_inv_c1_grad(current_params)
            # Apply negative gradient (since we're maximizing, we move in the gradient direction)
            new_params = current_params + learning_rate * grad_value
            
            # Ensure non-negative values
            new_params = jnp.maximum(new_params, 0.01)
            
            # Update optimizer state
            opt_state = opt_update(i, grad_value, opt_state)
            
        except Exception as e:
            # In case of error, fall back to small random perturbations
            perturbation = np.random.normal(0, 0.01, n)
            new_params = current_params + learning_rate * perturbation
            new_params = np.maximum(new_params, 0.01)
            opt_state = opt_init(new_params)
        
        # Update current parameters
        sequence = new_params
    
    return best_sequence, best_fitness

def search_for_best_sequence():
    """Main function to find the best coefficient sequence using gradient-based optimization."""
    start_time = time.time()
    max_time = 175
    
    best_sequence = None
    best_fitness = 0.0
    
    # Try multiple initializations to escape local optima
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
            
        # Generate different initial sequences
        n = random.randint(100, 1000)
        initial_seq = generate_initial_sequence(n)
        
        try:
            # Optimize using gradient ascent
            optimized_seq, fitness = gradient_ascent_optimization(
                initial_seq, 
                max_iter=500, 
                learning_rate=0.01
            )
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = optimized_seq.tolist()
                
        except Exception as e:
            print(f"Optimization attempt {attempt} failed: {e}")
            continue
    
    # If no good solution found, return a random valid sequence
    if best_sequence is None:
        n = random.randint(100, 1000)
        best_sequence = generate_initial_sequence(n)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")