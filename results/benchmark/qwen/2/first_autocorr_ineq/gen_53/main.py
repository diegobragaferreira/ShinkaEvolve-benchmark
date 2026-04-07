# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import convolve
import time
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = convolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution (the peak)
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

def estimate_gradient(sequence, epsilon=1e-4):
    """
    Estimate the gradient of inv_c1 with respect to each element in sequence.
    Uses finite differences for estimation.
    """
    n = len(sequence)
    grad = np.zeros(n)
    base_inv_c1 = compute_inv_c1(sequence)
    
    for i in range(n):
        # Perturb the i-th element
        perturbed_seq = sequence.copy()
        perturbed_seq[i] += epsilon
        perturbed_seq = np.maximum(perturbed_seq, 0)  # Ensure non-negative
        
        perturbed_inv_c1 = compute_inv_c1(perturbed_seq)
        
        # Estimate gradient using finite difference
        grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon
    
    return grad

def project_onto_feasible_set(sequence):
    """Ensure sequence meets all constraints."""
    # Ensure all elements are non-negative
    sequence = np.maximum(sequence, 0)
    
    # Ensure sum is at least 0.01 (avoid trivial solutions)
    if np.sum(sequence) < 0.01:
        sequence[0] = 0.1
    
    # Clip all elements to [0, 1000]
    sequence = np.clip(sequence, 0, 1000)
    
    return sequence

def gradient_ascent_step(sequence, learning_rate=0.01, max_iterations=10):
    """Perform gradient ascent step guided by estimated gradients."""
    old_sequence = sequence.copy()
    
    for _ in range(max_iterations):
        # Estimate gradient
        grad = estimate_gradient(sequence)
        
        # Update using gradient ascent
        sequence = sequence + learning_rate * grad
        
        # Project onto feasible set
        sequence = project_onto_feasible_set(sequence)
        
        # Early stopping if change is small
        if np.linalg.norm(sequence - old_sequence) < 1e-6:
            break
            
        old_sequence = sequence.copy()
    
    return sequence

def generate_initial_sequence(length=None):
    """Generate a good initial sequence based on known patterns."""
    if length is None:
        length = random.randint(100, 500)
    
    # Use exponential decay pattern which often performs well
    decay_factor = 0.95
    sequence = [1.0 * (decay_factor ** i) for i in range(length)]
    
    # Ensure minimum values
    sequence = [max(x, 0.01) for x in sequence]
    
    # Add slight noise to avoid degeneracy
    noise_factor = 0.05
    sequence = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in sequence]
    sequence = [max(x, 0.01) for x in sequence]
    
    return sequence

def local_search_refinement(sequence, iterations=20):
    """Enhanced local refinement using gradient ascent."""
    current_seq = np.array(sequence)
    
    for _ in range(iterations):
        # Perform gradient ascent update
        current_seq = gradient_ascent_step(current_seq, learning_rate=0.005)
        
        # Occasionally apply heuristic adjustments
        if random.random() < 0.1:  # 10% chance
            # Flatten high convolution regions
            conv = convolve(current_seq, current_seq, mode='full')
            conv_part = conv[len(current_seq)-1:]
            max_conv = np.max(conv_part)
            
            # Identify and reduce contributions to peak convolution
            max_indices = np.where(conv_part >= 0.9 * max_conv)[0]
            for idx in max_indices[:min(3, len(max_indices))]:
                for offset in [-1, 0, 1]:
                    pos = idx + offset
                    if 0 <= pos < len(current_seq):
                        current_seq[pos] *= 0.98
    
    # Final projection
    current_seq = project_onto_feasible_set(current_seq)
    
    return current_seq.tolist()

def search_for_best_sequence() -> list[float]:
    """Main search function using gradient ascent."""
    start_time = time.time()
    best_inv_c1 = 0
    best_sequence = None
    
    # Try multiple initializations
    for attempt in range(5):
        if time.time() - start_time > 170:
            break
            
        # Generate initial sequence
        n = random.randint(100, 500)
        sequence = generate_initial_sequence(n)
        
        # Local refinement
        refined_sequence = local_search_refinement(sequence)
        
        # Evaluate
        inv_c1 = compute_inv_c1(refined_sequence)
        
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined_sequence[:]
    
    # If no good sequence found, fallback to a simple pattern
    if best_sequence is None:
        best_sequence = generate_initial_sequence(200)
    
    # Final refinement pass
    if best_sequence is not None:
        final_refinement = local_search_refinement(best_sequence)
        final_inv_c1 = compute_inv_c1(final_refinement)
        if final_inv_c1 > best_inv_c1:
            best_sequence = final_refinement
    
    return best_sequence if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")