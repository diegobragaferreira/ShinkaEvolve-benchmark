# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def convolve_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Use scipy's fftconvolve for better numerical stability
    conv = fftconvolve(seq, seq, mode='full')
    return conv[:2*n - 1]

def compute_c1_value(seq):
    """Compute the C1 constant from the sequence."""
    n = len(seq)
    if n == 0:
        return float('inf')

    # Use FFT for efficiency when possible
    if n > 100:
        conv = convolve_fft(seq)
    else:
        conv = np.convolve(seq, seq, mode='full')

    max_conv = np.max(conv)
    sum_seq = np.sum(seq)

    if sum_seq < 1e-10:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def compute_gradient_c1(seq):
    """Approximate gradient of C1 with respect to sequence elements."""
    n = len(seq)
    if n == 0:
        return np.array([])

    # Normalize for easier gradient calculation
    norm_sum = np.sum(seq)
    if norm_sum < 1e-10:
        return np.zeros(n)

    # Use FFT for convolution
    conv = convolve_fft(seq, seq)
    max_conv_idx = np.argmax(conv)
    max_conv_val = conv[max_conv_idx]

    # For simplicity, estimate gradient assuming uniform contribution
    # This is a rough approximation for demonstration purposes
    grad = np.zeros(n)
    for i in range(n):
        # Simple derivative approximation
        eps = 1e-6
        seq_plus = seq.copy()
        seq_plus[i] += eps
        seq_minus = seq.copy()
        seq_minus[i] -= eps
        c1_plus = compute_c1_value(seq_plus)
        c1_minus = compute_c1_value(seq_minus)
        grad[i] = (c1_plus - c1_minus) / (2 * eps)

    return grad

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Compute a new sequence by gradient-based update."""
    try:
        n = len(sequence)
        if n < 10:
            return None

        # Normalize to avoid numerical issues
        sum_seq = np.sum(sequence)
        if sum_seq < 1e-10:
            return None

        # Convert to numpy array for easier manipulation
        seq_np = np.array(sequence)

        # Estimate the gradient direction
        grad = compute_gradient_c1(seq_np)

        # Avoid division by zero in gradient direction
        if np.allclose(grad, 0):
            # If gradient is nearly zero, do a small random perturbation
            perturbed_seq = seq_np + np.random.normal(0, 1e-3, n)
            return np.maximum(perturbed_seq, 0).tolist()

        # Gradient descent with adaptive step size
        # Step size is inversely proportional to magnitude of gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            step_size = 0.01
        else:
            step_size = min(0.05, 1.0 / grad_norm)

        # Update sequence in the direction that decreases C1 (negative gradient)
        updated_seq = seq_np - step_size * grad

        # Ensure non-negativity
        updated_seq = np.maximum(updated_seq, 0)

        # Re-normalize to maintain similar magnitude
        updated_sum = np.sum(updated_seq)
        if updated_sum > 1e-10:
            updated_seq = updated_seq * sum_seq / updated_sum

        return updated_seq.tolist()

    except Exception as e:
        print(f"Error in gradient update: {e}")
        # Fallback to small random perturbations
        return [(x + random.uniform(-1, 1)) for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Search for the best coefficient sequence using gradient guidance."""
    # Start with a structured sequence (geometric decay)
    n = random.randint(100, 1000)
    base_sequence = [1000 * (0.9 ** (i // 10)) for i in range(n)]
    
    # Add small random noise for diversity
    sequence = [x + random.uniform(-10, 10) for x in base_sequence]

    # Iteratively improve the sequence with gradient updates
    max_iter = 100
    for _ in range(max_iter):
        improved_seq = get_good_direction_to_move_into(sequence)
        if improved_seq is not None:
            sequence = improved_seq
        else:
            # Fallback to slight random noise if gradient fails
            sequence = [x + random.uniform(-1, 1) for x in sequence]

    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
