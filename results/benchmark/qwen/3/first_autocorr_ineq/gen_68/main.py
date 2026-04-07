# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import List, Optional
import time
import jax
from jax import grad, jit, numpy as jnp
from jax.config import config

# Enable 64-bit precision for better numerical stability
config.update("jax_enable_x64", True)

def compute_convolution_with_penalty(sequence):
    """Compute convolution with a penalty term for high peaks"""
    n = len(sequence)
    if n == 0:
        return float('inf'), float('inf')
    
    # Convert to jax array for automatic differentiation
    seq = jnp.array(sequence, dtype=jnp.float64)
    
    # Manual convolution for better control
    conv = jnp.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < 2 * n - 1:
                conv = conv.at[k].add(seq[i] * seq[j])
    
    max_conv = jnp.max(conv)
    sum_sq = jnp.sum(seq)**2
    
    return max_conv, sum_sq

@jit
def autocorrelation_penalty_loss(sequence, penalty_weight=1.0):
    """Loss function combining convolution max and sum squared with penalty"""
    max_conv, sum_sq = compute_convolution_with_penalty(sequence)
    # Prevent division by zero
    sum_sq = jnp.maximum(sum_sq, 1e-12)
    
    # C1 = 2n * max_conv / sum_sq
    # We want to minimize C1, so we maximize 1/C1 = sum_sq / (2n * max_conv)
    # But we also want to penalize large max_conv values
    loss = penalty_weight * max_conv - sum_sq / (2 * len(sequence))
    return loss

@jit
def compute_autocorrelation_gradient(sequence):
    """Compute gradient of the autocorrelation penalty loss"""
    return grad(autocorrelation_penalty_loss)(jnp.array(sequence, dtype=jnp.float64))

def get_good_direction_to_move_into(
    sequence: List[float],
) -> Optional[List[float]]:
    """Returns the direction to move into the sequence using gradient descent approach."""
    n = len(sequence)
    if n < 10:
        return None
        
    # Convert to numpy array
    seq_array = np.array(sequence, dtype=np.float64)
    sum_sequence = np.sum(seq_array)
    
    if sum_sequence < 1e-10:
        return None
        
    # Normalize sequence to avoid numerical issues
    normalized_sequence = seq_array * np.sqrt(2 * n) / sum_sequence
    
    # Compute gradients using JAX for faster computation
    try:
        grad_vals = compute_autocorrelation_gradient(normalized_sequence)
    except Exception:
        return None
        
    # Ensure gradients are finite
    if not np.all(np.isfinite(grad_vals)):
        return None
        
    # Adaptive step size based on gradient magnitude
    grad_norm = np.linalg.norm(grad_vals)
    if grad_norm < 1e-10:
        step_size = 0.01
    else:
        # Dynamic step size: smaller when gradients are large, bigger when small
        step_size = 0.01 / (1.0 + grad_norm) 
        
    # Update using gradient descent with momentum
    momentum_factor = 0.9
    # Initialize or accumulate momentum
    if not hasattr(get_good_direction_to_move_into, 'momentum'):
        get_good_direction_to_move_into.momentum = np.zeros_like(grad_vals)
    
    get_good_direction_to_move_into.momentum = (
        momentum_factor * get_good_direction_to_move_into.momentum 
        + (1 - momentum_factor) * grad_vals
    )
    
    # Apply update
    new_sequence = normalized_sequence - step_size * get_good_direction_to_move_into.momentum
    
    # Ensure all values are non-negative
    new_sequence = np.maximum(new_sequence, 0)
    
    # Apply small random noise for exploration
    noise_factor = 0.001
    noise = np.random.normal(0, noise_factor, len(new_sequence))
    new_sequence = new_sequence + noise
    
    # Clip values to reasonable bounds
    new_sequence = np.clip(new_sequence, 0, 1000)
    
    return new_sequence.tolist()

def search_for_best_sequence() -> List[float]:
    """Search for the best coefficient sequence using gradient-based optimization."""
    # Initialize with diverse strategies
    n = random.randint(100, 500)
    best_sequence = [random.uniform(0.1, 1.0) for _ in range(n)]
    
    # Track best performance
    best_c1 = float('inf')
    last_improvement = 0
    max_iter_without_improvement = 50
    
    # Run optimization loop
    for iteration in range(100):
        try:
            # Get direction to move into
            new_sequence = get_good_direction_to_move_into(best_sequence)
            
            if new_sequence is not None:
                # Validate and compute C1
                seq_array = np.array(new_sequence)
                sum_sq = np.sum(seq_array)**2
                if sum_sq > 1e-10:
                    # Compute convolution manually for accuracy
                    n = len(seq_array)
                    conv = np.zeros(2 * n - 1)
                    for i in range(n):
                        for j in range(n):
                            k = i + j
                            if 0 <= k < 2 * n - 1:
                                conv[k] += seq_array[i] * seq_array[j]
                    
                    max_conv = np.max(conv)
                    c1 = 2 * n * max_conv / sum_sq
                    
                    if c1 < best_c1:
                        best_c1 = c1
                        best_sequence = new_sequence[:]
                        last_improvement = iteration
                        
        except Exception:
            pass
            
        # Early stopping if no improvement
        if iteration - last_improvement > max_iter_without_improvement:
            break
            
        # Occasionally random perturbation to escape local minima
        if random.random() < 0.1 and iteration > 10:
            idx = random.randint(0, len(best_sequence) - 1)
            best_sequence[idx] = max(0, best_sequence[idx] + random.uniform(-0.5, 0.5))
    
    # Final normalization
    total = sum(best_sequence)
    if total > 0.01:
        best_sequence = [x / total for x in best_sequence]
    else:
        best_sequence = [1.0 for _ in best_sequence]
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
