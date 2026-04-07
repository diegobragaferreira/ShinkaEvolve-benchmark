# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import convolve
from scipy.optimize import minimize
import random
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

MAX_TIME_SECONDS = 180
MAX_EVALUATIONS = 10000

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution using FFT for efficiency
    conv = convolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution
    max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards

    # Normalize and compute C1
    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    c1 = (2 * len(sequence) * max_conv) / sum_sq
    return c1

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def adjust_sequence_by_convolution(sequence):
    """Modify sequence to reduce the maximum convolution value."""
    seq = np.array(sequence)
    n = len(seq)
    
    # Compute convolution
    conv = convolve(seq, seq, mode='full')
    conv_part = conv[n-1:]  # Relevant part
    
    # Find indices where convolution peaks
    max_conv_val = np.max(conv_part)
    max_indices = np.where(conv_part == max_conv_val)[0]
    
    # Adjust elements near these peaks to reduce convolution max
    new_seq = seq.copy()
    for idx in max_indices[:min(3, len(max_indices))]:  # Limit adjustments
        if idx > 0 and idx < n-1:  # Avoid boundary issues
            # Reduce neighboring elements by a factor
            new_seq[idx-1] *= 0.95
            new_seq[idx+1] *= 0.95
            
    return np.maximum(new_seq, 0)

def perturb_sequence(sequence, delta):
    """Perturb sequence slightly."""
    seq = np.array(sequence)
    # Add small random noise with controlled magnitude
    perturbation = np.random.normal(0, delta, len(seq))
    new_seq = seq + perturbation
    return np.maximum(new_seq, 0)

def random_perturb(sequence, scale):
    """Randomly perturb the sequence."""
    seq = np.array(sequence)
    # Random scaling of elements
    factors = np.random.uniform(0.9, 1.1, len(seq))
    new_seq = seq * factors
    return np.maximum(new_seq, 0)

def optimize_step_sequence(initial_sequence, max_iter=100):
    """Optimize a step sequence using a more robust approach."""
    current_seq = np.array(initial_sequence, dtype=float)

    # Ensure we have at least one non-zero element
    if np.sum(current_seq) < 0.01:
        current_seq[0] = 0.1

    best_seq = current_seq.copy()
    best_inv_c1 = compute_inv_c1(best_seq)

    # Try different optimization strategies
    strategies = [
        lambda s: perturb_sequence(s, 0.05),
        lambda s: random_perturb(s, 0.1),
        lambda s: adjust_sequence_by_convolution(s)
    ]

    for _ in range(max_iter):
        # Try a few different strategies
        for strategy in strategies:
            try:
                candidate_seq = strategy(current_seq)

                # Ensure all elements are non-negative and sum is reasonable
                candidate_seq = np.maximum(candidate_seq, 0)
                if np.sum(candidate_seq) < 0.01:
                    candidate_seq[0] = 0.1

                inv_c1 = compute_inv_c1(candidate_seq)
                if inv_c1 > best_inv_c1:
                    best_seq = candidate_seq.copy()
                    best_inv_c1 = inv_c1

            except Exception as e:
                continue

        # Always keep the best sequence found so far
        current_seq = best_seq.copy()

    return best_seq.tolist()

def generate_initial_sequences(count=10):
    """Generate multiple initial sequences to try."""
    sequences = []

    # Generate sequences of various lengths
    for _ in range(count):
        n = random.randint(50, 500)  # More reasonable range
        # Create sequences with different patterns
        if random.random() < 0.3:
            # Uniform sequence
            seq = np.ones(n) * random.uniform(0.1, 1.0)
        elif random.random() < 0.6:
            # Exponential decay
            decay_factor = random.uniform(0.8, 0.99)
            seq = [1.0 * (decay_factor ** i) for i in range(n)]
            seq = [max(x, 0.01) for x in seq]  # Ensure minimum value
        else:
            # Random sequence with some structure
            seq = [random.uniform(0.01, 10.0) for _ in range(n)]

        sequences.append(seq)

    return sequences

def search_for_best_sequence():
    """Search for the best coefficient sequence."""
    start_time = time.time()
    best_inv_c1 = 0
    best_sequence = []

    # Generate multiple initial sequences
    initial_sequences = generate_initial_sequences(10)

    for seq in initial_sequences:
        # Optimize each sequence
        optimized_seq = optimize_step_sequence(seq, max_iter=50)
        inv_c1 = compute_inv_c1(optimized_seq)

        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized_seq[:]

        # Check if time has been exceeded
        if time.time() - start_time > MAX_TIME_SECONDS:
            break

    # Final refinement
    if best_sequence:
        refined_seq = optimize_step_sequence(best_sequence, max_iter=20)
        final_inv_c1 = compute_inv_c1(refined_seq)
        if final_inv_c1 > best_inv_c1:
            best_sequence = refined_seq

    return best_sequence if best_sequence else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
