# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def evaluate_sequence(sequence):
    """
    Evaluate a sequence and return its performance metric (1/C₁).
    
    Args:
        sequence: List of non-negative real numbers
        
    Returns:
        float: Performance metric (1/C₁) - higher is better
    """
    try:
        # Convert to numpy array
        a = np.array(sequence)
        sum_a = np.sum(a)
        
        # Avoid division by zero or negligible sums
        if sum_a < 1e-10:
            return 0.0
            
        # Compute autoconvolution using FFT for efficiency
        b = fftconvolve(a, a, mode='full')
        b = b[len(a)-1:2*len(a)-1]  # Convolution part
        
        max_b = np.max(b)
        
        # Compute C₁ = 2n * max(b) / (sum(a))^2
        n = len(a)
        c1 = 2 * n * max_b / (sum_a ** 2)
        
        # Return inverse for maximization
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0
        
        return inv_c1
    except Exception as e:
        return 0.0

def solve_convolution_lp(f_sequence, rhs):
    """
    Solves the convolution LP for a given sequence and RHS.
    """
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
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

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def gradient_ascent_step(sequence, learning_rate=1e-5, max_iterations=100):
    """
    Perform gradient ascent on the sequence to improve the 1/C₁ value.
    """
    # Clone sequence to avoid modifying original
    seq = np.array(sequence, dtype=float)
    n = len(seq)
    
    # Normalize to avoid numerical issues
    sum_seq = np.sum(seq)
    if sum_seq < 1e-10:
        seq += 1e-5
        sum_seq = np.sum(seq)
        
    # Normalize
    seq /= sum_seq
    
    for _ in range(max_iterations):
        # Compute convolution
        conv = fftconvolve(seq, seq, mode='full')
        conv = conv[n-1:2*n-1]
        
        # Compute derivatives
        # df/dx_i = (d/dx_i) [1 / (2n * max(conv) / (sum(seq)^2))]
        # This requires derivative of max(conv) w.r.t. seq
        
        # Simplified gradient estimation by finite differences
        epsilon = 1e-8
        grad = np.zeros_like(seq)
        
        for i in range(n):
            seq_plus = seq.copy()
            seq_plus[i] += epsilon
            seq_plus /= np.sum(seq_plus)  # Re-normalize
            
            conv_plus = fftconvolve(seq_plus, seq_plus, mode='full')
            conv_plus = conv_plus[n-1:2*n-1]
            max_plus = np.max(conv_plus)
            
            conv_minus = fftconvolve(seq, seq, mode='full')
            conv_minus = conv_minus[n-1:2*n-1]
            max_minus = np.max(conv_minus)
            
            grad[i] = (max_plus - max_minus) / epsilon
            
        # Update sequence
        seq += learning_rate * grad
        seq = np.maximum(seq, 0)  # Ensure non-negative
        
        # Renormalize
        sum_seq = np.sum(seq)
        if sum_seq > 0:
            seq /= sum_seq
    
    # Return final sequence
    return seq.tolist()

def generate_structured_sequence(length):
    """
    Generate a structured sequence using a sinusoidal pattern for better initialization.
    """
    # Use a sine wave pattern for better initial structure
    sequence = [abs(np.sin(i * np.pi / length)) * 1000 for i in range(length)]
    # Add some randomness to avoid perfect symmetry
    for i in range(length):
        if random.random() < 0.1:
            sequence[i] += random.uniform(-100, 100)
    
    # Ensure non-negative
    sequence = [max(0, x) for x in sequence]
    
    # Ensure sum is meaningful
    if sum(sequence) < 0.01:
        sequence[random.randint(0, length-1)] += 0.01
    
    return sequence

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    start_time = time.time()
    timeout = 170  # Leave 10 seconds for cleanup
    
    # Generate initial sequence
    length = random.randint(100, 500)
    sequence = generate_structured_sequence(length)
    best_sequence = sequence[:]
    
    # Evaluate initial sequence
    best_score = evaluate_sequence(sequence)
    
    print(f"Initial score: {best_score:.6f}")
    
    # Try gradient ascent refinement
    refined_sequence = gradient_ascent_step(sequence)
    refined_score = evaluate_sequence(refined_sequence)
    
    if refined_score > best_score:
        best_sequence = refined_sequence
        best_score = refined_score
        print(f"After refinement: {best_score:.6f}")
    
    # Try LP-based optimization as an alternative
    n = len(sequence)
    sum_seq = np.sum(sequence)
    if sum_seq > 0:
        norm_seq = [x * np.sqrt(2 * n) / sum_seq for x in sequence]
        rhs = np.max(np.convolve(norm_seq, norm_seq))
        lp_solution = solve_convolution_lp(norm_seq, rhs)
        
        if lp_solution is not None:
            # Reconstruct from LP solution
            new_sum = np.sum(lp_solution)
            normalized_lp = [x * np.sqrt(2 * n) / new_sum for x in lp_solution]
            t = 0.01
            blended_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_lp)]
            blended_score = evaluate_sequence(blended_sequence)
            
            if blended_score > best_score:
                best_sequence = blended_sequence
                best_score = blended_score
                print(f"After LP blending: {best_score:.6f}")
    
    # Final refinement
    final_sequence = gradient_ascent_step(best_sequence)
    final_score = evaluate_sequence(final_sequence)
    
    if final_score > best_score:
        best_sequence = final_sequence
        best_score = final_score
        print(f"After final refinement: {best_score:.6f}")
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
