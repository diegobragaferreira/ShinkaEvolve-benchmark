# EVOLVE-BLOCK-START
import numpy as np
from scipy import signal, optimize
from scipy.fft import fft, ifft
import random
from typing import List, Optional
import time

# Constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
FFT_THRESHOLD = 100  # Use FFT for sequences longer than this
CONVERGENCE_TOLERANCE = 1e-6
MAX_STAGNANT_ITERATIONS = 50

def autocorrelation_constant(sequence: List[float]) -> float:
    """
    Calculates C₁ = 2n * max(b) / (sum(a))^2 where b = a * a (autoconvolution).
    Returns the inverse 1/C₁ which we want to maximize.
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    
    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0
    
    # Compute autoconvolution using FFT for efficiency
    if n > FFT_THRESHOLD:
        # Use FFT for fast convolution
        padded_len = 2 * n - 1
        seq_fft = fft(sequence, padded_len)
        conv_fft = seq_fft * seq_fft.conj()  # Element-wise multiplication
        autoconv = ifft(conv_fft).real
        max_conv = max(autoconv)
    else:
        # Direct convolution for small sequences
        autoconv = signal.convolve(sequence, sequence, mode='full')
        max_conv = max(autoconv)
    
    # Calculate C₁
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def compute_gradient_approximation(sequence: List[float], epsilon_base: float = 1e-4) -> List[float]:
    """
    Approximate gradient using finite differences with adaptive epsilon.
    Epsilon is scaled adaptively based on the magnitude of the sequence elements.
    """
    n = len(sequence)
    grad = []
    for i in range(n):
        # Determine adaptive epsilon based on element magnitude
        elem_mag = abs(sequence[i])
        if elem_mag < 1e-6:
            epsilon = epsilon_base
        else:
            epsilon = epsilon_base * elem_mag
        
        # Perturb dimension i
        perturbed_plus = sequence[:]
        perturbed_minus = sequence[:]
        perturbed_plus[i] += epsilon
        perturbed_minus[i] -= epsilon
        
        # Ensure non-negativity
        perturbed_plus[i] = max(0, perturbed_plus[i])
        perturbed_minus[i] = max(0, perturbed_minus[i])
        
        # Evaluate function
        f_plus = autocorrelation_constant(perturbed_plus)
        f_minus = autocorrelation_constant(perturbed_minus)
        
        grad_i = (f_plus - f_minus) / (2 * epsilon)
        grad.append(grad_i)
    
    return grad

def adaptive_step_size(prev_inv_c1: float, iteration: int) -> float:
    """
    Adaptive step size based on iteration count with exponential decay.
    """
    # Base step size with exponential decay
    base_step = 0.01 * (0.95 ** iteration)
    return max(base_step, 1e-6)

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build convolution constraint matrix efficiently
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    # Use a more robust solver configuration
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub,
                                options={'maxiter': 1000, 'tol': 1e-8})
    except:
        return None

    if result.success:
        g_sequence = result.x
        # Clip negative values that might arise from numerical errors
        g_sequence = np.clip(g_sequence, 0, None)
        return g_sequence
    else:
        return None

def get_good_direction_to_move_into(sequence: List[float]) -> Optional[List[float]]:
    """
    Returns the direction to move into the sequence using enhanced gradient ascent with adaptive step size.
    """
    n = len(sequence)
    if n == 0:
        return None
    
    # Check if we have enough elements
    if n < MIN_SEQ_LENGTH:
        # Expand sequence
        extended_seq = sequence + [0.0] * (MIN_SEQ_LENGTH - n)
        sequence = extended_seq
    
    # Perform gradient ascent with adaptive step size
    current_inv_c1 = autocorrelation_constant(sequence)
    prev_grad_norm = 1.0
    stagnant_count = 0
    
    # Maximum number of gradient steps
    max_steps = 1000
    for step in range(max_steps):
        if time.time() > start_time + MAX_TIME_SECONDS - 2:
            break
            
        # Compute gradient
        try:
            gradient = compute_gradient_approximation(sequence)
        except Exception:
            return None
            
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-10:
            break
            
        # Compute adaptive step size
        step_size = adaptive_step_size(current_inv_c1, step)
        
        # Update sequence
        new_sequence = []
        for i in range(len(sequence)):
            new_val = sequence[i] + step_size * gradient[i]
            new_sequence.append(max(0, new_val))
        
        # Evaluate new sequence
        new_inv_c1 = autocorrelation_constant(new_sequence)
        
        if new_inv_c1 > current_inv_c1:
            sequence = new_sequence
            current_inv_c1 = new_inv_c1
            stagnant_count = 0
        else:
            stagnant_count += 1
            if stagnant_count > MAX_STAGNANT_ITERATIONS:
                break
            
        prev_grad_norm = grad_norm
    
    return sequence

def initialize_good_sequence():
    """Initialize sequence with known good patterns."""
    # Try some known good patterns that often perform well
    patterns = [
        # Simple uniform pattern
        [1.0] * 100,
        # Alternating pattern
        [1.0, 0.0] * 50,
        # Exponential decay
        [1.0 / (i + 1) for i in range(100)],
        # Gaussian-like decay
        [np.exp(-i**2 / 200.0) for i in range(100)]
    ]

    # Choose randomly from patterns
    pattern = random.choice(patterns)
    # Add some noise to avoid local optima
    noise_level = 0.1
    noisy_pattern = [max(0.0, x + random.uniform(-noise_level, noise_level) * x)
                     for x in pattern]
    return noisy_pattern

def search_for_best_sequence() -> List[float]:
    """
    Function to search for the best coefficient sequence.
    """
    global start_time
    start_time = time.time()
    
    # Initialize with a good sequence to speed up early convergence
    sequence = initialize_good_sequence()
    
    # Improve using gradient ascent
    improved_sequence = get_good_direction_to_move_into(sequence)
    
    if improved_sequence is not None and len(improved_sequence) > 0:
        return improved_sequence
    else:
        # Fallback to uniform sequence
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")