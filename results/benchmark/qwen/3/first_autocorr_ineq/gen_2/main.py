# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
from joblib import Parallel, delayed
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C1 for a given sequence.
    C1 = 2n * max(convolution) / (sum(sequence))^2
    """
    n = len(sequence)
    if n == 0:
        return float('inf')
    
    # Use FFT for efficient convolution
    convolved = fft_convolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved)
    sum_seq = np.sum(sequence)
    
    if sum_seq < 0.01:
        return float('inf')
        
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    return c1

def compute_inv_c1(sequence):
    """
    Compute 1/C1 for maximization purposes.
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def generate_random_sequence(length, min_height=0.0, max_height=1000.0):
    """
    Generate a random sequence with specified length and height bounds.
    """
    return [random.uniform(min_height, max_height) for _ in range(length)]

def optimize_single_sequence(initial_seq, max_iterations=50):
    """
    Perform local optimization on a single sequence.
    """
    seq = np.array(initial_seq)
    best_seq = seq.copy()
    best_inv_c1 = compute_inv_c1(seq)
    
    for _ in range(max_iterations):
        # Try perturbing each element
        new_seq = seq.copy()
        idx = random.randint(0, len(seq)-1)
        new_seq[idx] = max(0.0, new_seq[idx] + random.gauss(0, 0.1 * np.std(new_seq)))
        
        inv_c1_new = compute_inv_c1(new_seq)
        if inv_c1_new > best_inv_c1:
            best_seq = new_seq
            best_inv_c1 = inv_c1_new
            
        seq = best_seq
        
    return best_seq.tolist()

def evaluate_sequence_parallel(sequences):
    """
    Evaluate multiple sequences in parallel.
    """
    results = Parallel(n_jobs=-1)(delayed(compute_inv_c1)(seq) for seq in sequences)
    return results

def adaptive_search():
    """
    Main adaptive search algorithm.
    """
    # Initialize parameters
    max_time = 180.0
    start_time = time.time()
    
    # Start with several random sequences of varying lengths
    lengths_to_try = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    best_inv_c1 = 0.0
    best_sequence = []
    
    # Try different sequence lengths
    for n in lengths_to_try:
        if time.time() - start_time > max_time * 0.9:
            break
            
        # Generate multiple random sequences of this length
        num_starts = 5
        sequences = [generate_random_sequence(n) for _ in range(num_starts)]
        
        # Optimize each sequence
        optimized_sequences = [optimize_single_sequence(seq) for seq in sequences]
        
        # Evaluate all sequences
        inv_c1_values = evaluate_sequence_parallel(optimized_sequences)
        
        # Select the best among these
        best_idx = np.argmax(inv_c1_values)
        current_inv_c1 = inv_c1_values[best_idx]
        
        if current_inv_c1 > best_inv_c1:
            best_inv_c1 = current_inv_c1
            best_sequence = optimized_sequences[best_idx]
            
    # Final refinement
    if time.time() - start_time < max_time * 0.95:
        refined_sequence = optimize_single_sequence(best_sequence, max_iterations=100)
        final_inv_c1 = compute_inv_c1(refined_sequence)
        if final_inv_c1 > best_inv_c1:
            best_sequence = refined_sequence
            
    return best_sequence

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """
    Returns the direction to move into the sequence.
    """
    try:
        # Attempt to find a better direction using local optimization
        n = len(sequence)
        if n < 2:
            return None
            
        # Create a slightly modified version to test
        modified_sequence = sequence.copy()
        idx = random.randint(0, n-1)
        modified_sequence[idx] = max(0.0, modified_sequence[idx] + random.gauss(0, 0.1))
        
        # Try optimizing this direction
        optimized_sequence = optimize_single_sequence(modified_sequence, max_iterations=20)
        return optimized_sequence
    except Exception as e:
        print(f"Error in direction calculation: {e}")
        return None

def search_for_best_sequence() -> list[float]:
    """
    Function to search for the best coefficient sequence.
    """
    try:
        best_sequence = adaptive_search()
        if not best_sequence:
            # Fallback to simple random generation
            length = random.randint(100, 1000)
            best_sequence = generate_random_sequence(length)
            
        # Ensure minimum sum constraint
        sum_seq = np.sum(best_sequence)
        if sum_seq < 0.01:
            # Adjust if sum too small
            best_sequence[0] += 0.1
            
        return best_sequence
    except Exception as e:
        print(f"Fallback due to error: {e}")
        # Return a simple random sequence
        length = random.randint(100, 1000)
        return generate_random_sequence(length)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
