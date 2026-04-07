# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
from collections import deque
import random

# Constants for optimization
MAX_TIME_SECONDS = 170
MIN_SEQUENCE_LENGTH = 50
MAX_SEQUENCE_LENGTH = 500
BENCHMARK_RATIO_THRESHOLD = 1.0

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
        return float('inf')
    
    # Use FFT-based convolution for efficiency
    convolved = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved)
    sum_seq = sum(sequence)
    
    # Return C1 value
    return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing 1/C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 > 1e10:
        return float('inf')
    return -1.0 / c1  # We want to maximize 1/C1, so minimize -1/C1

def get_good_direction_to_move_into(sequence):
    """Returns the direction to move into the sequence using a more sophisticated approach."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None
        
    # Normalize sequence
    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence
    
    # Compute convolution to find maximum
    convolved = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
    max_conv_value = np.max(convolved)
    
    # Solve linear program for direction
    g_fun = solve_convolution_lp(normalized_sequence, max_conv_value)
    
    if g_fun is None:
        # Fallback to gradient descent with adaptive step size
        return gradient_descent_step(sequence, normalized_sequence, max_conv_value)
    
    # Scale back to original magnitude
    scaled_g_fun = [x * sum_sequence / np.sqrt(2 * n) for x in g_fun]
    return scaled_g_fun

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    
    # For numerical stability, ensure RHS is positive
    if rhs <= 0:
        rhs = 1.0
    
    # Objective coefficients (minimize sum of variables)
    c = -np.ones(n)
    
    # Build constraints matrix
    a_ub = []
    b_ub = []
    
    # Convolution constraints: sum_{j=0}^{n-1} f_j * x_{k-j} <= rhs for k = 0..2n-2
    # For better numerical conditioning, we work with valid indices only
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: x_i >= 0
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            return result.x
    except Exception:
        pass
    
    return None

def gradient_descent_step(sequence, normalized_sequence, max_conv_value):
    """Perform a gradient descent step to improve sequence."""
    n = len(sequence)
    # Find the index where maximum convolution occurs
    convolved = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
    max_positions = np.where(convolved == max_conv_value)[0]
    if len(max_positions) == 0:
        max_pos = 0
    else:
        max_pos = max_positions[0]
    
    # Create gradient in the direction that reduces max convolution
    grad_dir = np.zeros_like(normalized_sequence)
    
    # Focus on positions surrounding the maximum convolution
    window_size = min(5, n // 4)
    for i in range(max(0, max_pos - window_size), min(n, max_pos + window_size)):
        if i < n:
            grad_dir[i] = -1.0
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad_dir)
    if grad_norm > 1e-10:
        grad_dir /= grad_norm
    else:
        grad_dir = np.zeros_like(normalized_sequence)
        if max_pos < n:
            grad_dir[max_pos] = -1.0
            grad_norm = np.linalg.norm(grad_dir)
            if grad_norm > 1e-10:
                grad_dir /= grad_norm
    
    # Apply small step with adaptive step size
    t = 0.005
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(normalized_sequence, grad_dir)
    ]
    
    # Scale back to original magnitude
    sum_sequence = np.sum(sequence)
    scaled_new_sequence = [x * sum_sequence / np.sqrt(2 * n) for x in new_sequence]
    return scaled_new_sequence

def create_initial_sequences(n_samples=10):
    """Create diverse initial sequences."""
    sequences = []
    for _ in range(n_samples):
        n = np.random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
        seq = np.random.uniform(0, 100, n)
        if np.random.random() < 0.3:
            idxs = np.random.choice(n, size=min(5, n//4), replace=False)
            seq[idxs] *= np.random.uniform(5, 20)
        sequences.append(seq)
    
    # Include some known good structures
    sequences.append(np.array([1.0] * 100))  # Uniform
    sequences.append(np.array([1.0] * 50 + [0.0] * 50))  # Step function
    return sequences

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()
    best_score = float('-inf')
    best_sequence = None
    history = deque(maxlen=10)

    # Create initial diverse sequences
    initial_sequences = create_initial_sequences()

    for attempt in range(20):  # More attempts for better coverage
        if time.time() - start_time > MAX_TIME_SECONDS:
            break
            
        try:
            # Select a random initial sequence
            initial_seq = initial_sequences[np.random.randint(len(initial_sequences))]
            current_seq = initial_seq.copy()
            
            # Ensure minimum length
            if len(current_seq) < MIN_SEQUENCE_LENGTH:
                current_seq = np.pad(current_seq, (0, MIN_SEQUENCE_LENGTH - len(current_seq)),
                                   mode='constant', constant_values=0)
            
            current_score = evaluate_sequence(current_seq)
            history.append(current_score)
            
            # Local search loop
            iter_count = 0
            while iter_count < 100 and time.time() - start_time < MAX_TIME_SECONDS:
                if time.time() - start_time > MAX_TIME_SECONDS:
                    break
                    
                # Get direction to move into
                direction = get_good_direction_to_move_into(current_seq)
                
                if direction is not None:
                    # Update sequence
                    updated_seq = direction
                    updated_score = evaluate_sequence(updated_seq)
                    
                    if updated_score > current_score:
                        current_seq = updated_seq
                        current_score = updated_score
                        history.append(current_score)
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_sequence = current_seq.copy()
                            
                        iter_count = 0  # Reset counter on improvement
                    else:
                        iter_count += 1
                else:
                    # Fallback to random perturbation if direction not available
                    new_seq = current_seq + np.random.normal(0, 0.01, len(current_seq))
                    new_seq = np.maximum(new_seq, 0)  # Ensure non-negativity
                    new_score = evaluate_sequence(new_seq)
                    
                    if new_score > current_score:
                        current_seq = new_seq
                        current_score = new_score
                        history.append(current_score)
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_sequence = current_seq.copy()
                            
                        iter_count = 0  # Reset counter on improvement
                    else:
                        iter_count += 1
                        
        except Exception as e:
            continue
    
    # Final fallback to uniform sequence
    if best_sequence is None:
        best_sequence = np.array([1.0] * 100)
    
    return best_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
