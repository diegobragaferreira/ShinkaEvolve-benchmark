# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit
import math
from joblib import Parallel, delayed
from scipy import optimize
import jax
from jax import grad, jit, vmap
from jax import numpy as jnp
import jax.numpy as jnp
from functools import partial

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def fast_convolve_jit(a, b):
    """Fast convolution using Numba JIT compilation."""
    n = len(a)
    m = len(b)
    result = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
    return result

class FastAutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""

    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        if seq_tuple in self._cache:
            self._cache_hits += 1
            return self._cache[seq_tuple]

        self._cache_misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        n = len(sequence)

        # Use FFT-based convolution for efficiency O(n log n)
        if n > 500:
            try:
                conv = fftconvolve(sequence, sequence, mode='full')
            except Exception:
                # Fallback to JIT for large sequences if FFT fails
                conv = fast_convolve_jit(sequence, sequence)
        else:
            conv = fast_convolve_jit(sequence, sequence)
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache[seq_tuple] = result
        return result

# Global evaluator instance
_evaluator = FastAutocorrelationEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def generate_mathematical_step_function(n: int) -> List[float]:
    """
    Generate a step function with mathematical properties designed to minimize
    autocorrelation peaks. Uses a refined pattern that combines geometric decay
    with periodic modulation to reduce convolution maxima.
    """
    num_steps = max(4, min(30, n // 10))  # Adjusted for better coverage

    # Generate step positions with more complex pattern to avoid regularity
    step_positions = []
    step_width = n / num_steps

    for i in range(num_steps):
        # Base position with periodic modulation for irregularity
        base_pos = i * step_width
        modulate = 3 * np.sin(i * 0.7) * np.cos(i * 0.4)
        actual_start = max(0, min(n-1, int(base_pos + modulate)))
        step_positions.append(actual_start)

    # Clean up duplicates and ensure enough unique positions
    step_positions = sorted(set(step_positions))
    while len(step_positions) < num_steps:
        new_pos = random.randint(0, n-1)
        if new_pos not in step_positions:
            step_positions.append(new_pos)
    step_positions = sorted(step_positions[:num_steps])

    # Generate step heights using a combination of geometric decay and frequency modulation
    step_heights = []
    base_height = 100.0
    decay_rate = 0.85

    for i in range(len(step_positions)):
        # Apply geometric decay
        height_base = base_height * (decay_rate ** i)

        # Add frequency modulation for more irregularity
        freq_mod = 1 + 0.1 * np.sin(i * 0.9) * np.cos(i * 0.6)

        # Add controlled randomness to prevent perfect symmetry
        noise = random.uniform(0.85, 1.15)

        height = max(0.01, height_base * freq_mod * noise)
        step_heights.append(height)

    # Construct final sequence
    sequence = [0.0] * n
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        # Determine end position
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = n

        # Ensure bounds
        pos = max(0, min(n-1, pos))
        end_pos = max(pos+1, min(n, end_pos))

        # Set step values
        if end_pos > pos:
            sequence[pos:end_pos] = [height] * (end_pos - pos)

    return sequence

def generate_random_valid_sequence(length_range=(50, 500), method='mixed') -> List[float]:
    """Generate a random valid sequence within specified length range."""
    n = random.randint(*length_range)

    if method == 'step':
        # Generate step function with varied heights
        num_steps = max(2, min(20, n // 10))
        step_positions = sorted(random.sample(range(n), num_steps))
        step_heights = [random.uniform(0.1, 100.0) for _ in range(num_steps)]

        sequence = [0.0] * n
        for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
            if i < len(step_positions) - 1:
                end_pos = step_positions[i+1]
            else:
                end_pos = n
            sequence[pos:end_pos] = [height] * (end_pos - pos)

    elif method == 'mathematical_step':
        # Use our enhanced mathematical step function generator
        return generate_mathematical_step_function(n)

    elif method == 'gaussian':
        # Generate Gaussian-like distribution
        sequence = [random.gauss(50.0, 20.0) for _ in range(n)]
        sequence = [max(0.01, x) for x in sequence]

    else:  # default 'uniform'
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]

    return sequence

def compute_inv_c1_jax(sequence_array):
    """Compute 1/C1 using JAX for automatic differentiation."""
    n = len(sequence_array)
    # Compute convolution using FFT
    conv = fftconvolve(sequence_array, sequence_array, mode='full')
    max_conv = np.max(conv)
    sum_seq = np.sum(sequence_array)
    
    if sum_seq == 0:
        return 0.0
    
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return 1.0 / c1 if c1 > 0 else 0.0

def project_onto_feasible_region(x, lower_bound=0.01, upper_bound=1000.0):
    """Project x onto feasible region [lower_bound, upper_bound]"""
    return jnp.clip(x, lower_bound, upper_bound)

def gradient_ascent_with_projection(init_seq, learning_rate=0.01, max_iter=500):
    """Perform gradient ascent with projection to solve for optimal sequence."""
    # Convert to JAX array
    x = jnp.array(init_seq, dtype=jnp.float32)
    
    # Define loss (negative of 1/C1 to maximize 1/C1)
    def loss_fn(x):
        return -compute_inv_c1_jax(np.array(x))
    
    # Compute gradient
    grad_loss = grad(loss_fn)
    
    # Perform gradient ascent
    for i in range(max_iter):
        # Compute gradient
        grad_val = grad_loss(x)
        
        # Update with projection
        x_new = x + learning_rate * grad_val
        x_projected = project_onto_feasible_region(x_new)
        
        # Check for convergence
        if jnp.allclose(x, x_projected, atol=1e-4):
            break
            
        x = x_projected
    
    return np.array(x)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()
    
    # Initialize with a well-constructed mathematical step function
    n = random.randint(100, 1000)
    initial_sequence = generate_mathematical_step_function(n)
    
    # Apply gradient ascent to refine the sequence
    refined_sequence = gradient_ascent_with_projection(initial_sequence, learning_rate=0.01, max_iter=200)
    
    # Validate and finalize
    _, inv_c1 = compute_autocorrelation_constant(refined_sequence)
    if inv_c1 < 0.01:
        # Fallback to random if invalid
        refined_sequence = generate_random_valid_sequence()
    
    # Ensure minimal positive sum
    if np.sum(refined_sequence) < 0.01:
        refined_sequence = [max(0.01, x) for x in refined_sequence]
    
    # Normalize to avoid extreme values
    sum_seq = np.sum(refined_sequence)
    if sum_seq > 0.01:
        refined_sequence = [x / sum_seq * 100 for x in refined_sequence]
        
    return refined_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")