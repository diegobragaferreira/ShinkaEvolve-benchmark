# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
from collections import deque

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
        return float('inf')
    convolved = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved)
    sum_seq = sum(sequence)
    return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing 1/C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 > 1e10:
        return float('-inf')  # Penalize bad sequences heavily
    return 1.0 / c1

def estimate_gradient_direction(sequence, epsilon=1e-5):
    """Estimate gradient direction that reduces max convolution."""
    n = len(sequence)
    if n < 1:
        return None

    # Normalize sequence
    sum_seq = sum(sequence)
    if sum_seq < epsilon:
        return None
        
    normalized = np.array(sequence) / sum_seq
    
    # Compute convolution
    convolved = fftconvolve(normalized, normalized, mode='full')
    max_pos = np.argmax(convolved)
    
    # Estimate gradient by perturbing values around max position
    grad = np.zeros(n)
    window = min(10, n//4)
    for i in range(max(0, max_pos - window), min(n, max_pos + window)):
        if i < n:
            # Perturb in a way that might reduce the convolution at max_pos
            grad[i] = -normalized[i]  # Reverse sign to decrease contribution
            
    # Normalize gradient
    norm = np.linalg.norm(grad)
    if norm > epsilon:
        grad /= norm
    else:
        # Fall back to simple gradient
        grad = np.zeros(n)
        if max_pos < n:
            grad[max_pos] = -1.0
        norm = np.linalg.norm(grad)
        if norm > epsilon:
            grad /= norm
        else:
            return None
            
    # Convert back to original scale
    return grad * sum_seq

def gradient_ascent_step(sequence, step_size=0.01):
    """Perform one gradient ascent step."""
    grad = estimate_gradient_direction(sequence)
    if grad is None:
        return sequence
        
    # Apply gradient ascent
    new_seq = np.array(sequence) - step_size * grad
    new_seq = np.maximum(new_seq, 0)  # Ensure non-negative
    return new_seq.tolist()

def search_for_best_sequence():
    """Main search function using gradient ascent with multi-start."""
    best_score = float('-inf')
    best_sequence = None
    
    start_time = time.time()
    max_attempts = 20  # More attempts for better coverage
    
    for attempt in range(max_attempts):
        if time.time() - start_time > 170:
            break
            
        try:
            # Multi-start from various initial configurations
            n = np.random.randint(50, 500)
            initial_seq = np.random.uniform(0, 100, n)
            
            # Add structured elements occasionally
            if np.random.random() < 0.3:
                idxs = np.random.choice(n, size=min(5, n//4), replace=False)
                initial_seq[idxs] *= np.random.uniform(5, 20)
            
            # Local optimization loop
            current_seq = initial_seq.copy()
            for iteration in range(100):  # Limited iterations to save time
                if time.time() - start_time > 170:
                    break
                    
                # Gradient ascent step
                new_seq = gradient_ascent_step(current_seq, step_size=0.005)
                new_score = evaluate_sequence(new_seq)
                
                # Accept improvement or accept with probability
                current_score = evaluate_sequence(current_seq)
                if new_score > current_score:
                    current_seq = new_seq
                elif np.random.random() < 0.1:  # Sometimes accept worse moves
                    current_seq = new_seq
                    
                # Track best
                if new_score > best_score:
                    best_score = new_score
                    best_sequence = current_seq.copy()
                    
        except Exception as e:
            continue
    
    # Fallback to known good sequence
    if best_sequence is None:
        best_sequence = np.array([1.0] * 100)
        
    return best_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")