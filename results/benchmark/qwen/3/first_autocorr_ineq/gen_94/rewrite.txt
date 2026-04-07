# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time
import random
from typing import List, Tuple
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

@jit
def _objective_and_grad(params: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes negative of 1/C1 and its gradient."""
    # Apply sigmoid to ensure non-negativity and boundedness
    a = jnp.exp(params) / (1.0 + jnp.exp(params)) * 1000.0
    a = a.at[0].set(jnp.maximum(0.01, a[0]))  # Ensure minimum value
    
    # Convolution
    if n > 100:
        b = convolve_fft(np.array(a), np.array(a))
    else:
        b = convolve_direct(np.array(a), np.array(a))
    
    max_conv = jnp.max(b)
    sum_a = jnp.sum(a)
    
    # Avoid division by zero
    cond = sum_a > 0.01
    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = jnp.where(cond, 1.0 / c1, 0.0)
    
    # Return negative because we minimize
    return -inv_c1, grad(lambda p: -jnp.where(cond, 1.0 / (2 * n * jnp.max(convolve_fft(np.array(jnp.exp(p)/(1+jnp.exp(p))*1000.0), np.array(jnp.exp(p)/(1+jnp.exp(p))*1000.0))) / (jnp.sum(jnp.exp(p)/(1+jnp.exp(p))*1000.0)**2), 0.0))(params)

def gradient_optimization_step(initial_params: jnp.ndarray, n: int, 
                              max_iterations: int = 100) -> jnp.ndarray:
    """Performs gradient-based optimization on the transformed parameters."""
    # Define loss function
    def loss_func(params):
        a = jnp.exp(params) / (1.0 + jnp.exp(params)) * 1000.0
        a = a.at[0].set(jnp.maximum(0.01, a[0]))
        
        if n > 100:
            b = convolve_fft(np.array(a), np.array(a))
        else:
            b = convolve_direct(np.array(a), np.array(a))
            
        max_conv = jnp.max(b)
        sum_a = jnp.sum(a)
        
        cond = sum_a > 0.01
        c1 = 2 * n * max_conv / (sum_a ** 2)
        inv_c1 = jnp.where(cond, 1.0 / c1, 0.0)
        return -inv_c1
    
    # Optimize using L-BFGS
    try:
        result = minimize(loss_func, initial_params, method="l-bfgs", 
                          options={"maxiter": max_iterations})
        optimized_params = result.x
    except:
        optimized_params = initial_params
    
    return optimized_params

def initialize_sequence(n: int) -> jnp.ndarray:
    """Initialize sequence with specific strategies."""
    # Varying initialization schemes
    init_type = random.random()
    
    if init_type < 0.3:
        # Sparse initialization
        params = jnp.zeros(n)
        idx = random.randint(0, n-1)
        params = params.at[idx].set(random.random() * 5)
    elif init_type < 0.6:
        # Uniform initialization
        params = jnp.array([random.random() * 2 for _ in range(n)])
    else:
        # Gaussian-like with small values
        params = jnp.array([random.gauss(0, 0.5) for _ in range(n)])
    
    return params

def gradient_autocorrelation_optimizer(
    max_time_seconds: int = 180,
    num_candidates: int = 50,
    max_iterations_per_candidate: int = 100
) -> List[float]:
    """Gradient-based optimization approach for maximizing 1/C1."""
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = -float('inf')
    
    # Generate multiple initial sequences
    candidates = []
    for _ in range(num_candidates):
        n = random.randint(100, 1000)
        params = initialize_sequence(n)
        candidates.append((params, n))
    
    # Sequential optimization of candidates
    for params, n in candidates:
        if time.time() - start_time > max_time_seconds:
            break
            
        try:
            # Optimize using gradient-based method
            optimized_params = gradient_optimization_step(params, n, max_iterations_per_candidate)
            
            # Convert back to actual sequence
            sequence = jnp.exp(optimized_params) / (1.0 + jnp.exp(optimized_params)) * 1000.0
            sequence = sequence.at[0].set(jnp.maximum(0.01, sequence[0]))
            
            # Convert to list and compute C1
            seq_list = sequence.tolist()
            _, inv_c1 = compute_c1_constant(seq_list)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = seq_list
                
        except Exception as e:
            continue
    
    # Return best sequence or default
    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    random.seed(42)
    np.random.seed(42)
    
    return gradient_autocorrelation_optimizer()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")