# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_convolution_fft(sequence):
    """Compute convolution using FFT for efficiency."""
    # Using 'full' mode and then taking the middle part
    conv = fft_convolve(sequence, sequence, mode='full')
    # Return the middle part which corresponds to standard convolution
    mid = len(conv) // 2
    return conv[mid:]

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    try:
        n = len(sequence)
        if n == 0:
            return None
            
        # Compute convolution of current sequence
        conv = compute_convolution_fft(sequence)
        
        # Normalize the sequence for numerical stability
        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            sum_sequence = 1e-10
            
        # Scale for consistent comparison
        scale_factor = np.sqrt(2 * n) / sum_sequence
        normalized_sequence = [x * scale_factor for x in sequence]
        
        # Compute max convolution value for constraint
        max_conv = np.max(conv)
        
        # Try to solve the LP with current constraints
        g_fun = solve_convolution_lp_normalized(normalized_sequence, max_conv)
        
        if g_fun is None:
            # Fallback: try to construct a better direction manually
            return construct_fallback_direction(sequence, normalized_sequence)
        
        # Normalize the result
        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            sum_g = 1e-10
            
        normalized_g_fun = [x * scale_factor / sum_g for x in g_fun]
        
        # Blend with current sequence
        t = 0.01
        new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
        
        return new_sequence
        
    except Exception as e:
        print(f"Error in get_good_direction_to_move_into: {e}")
        return None

def solve_convolution_lp_normalized(f_sequence, rhs):
    """Solves the convolution LP for a normalized sequence and RHS."""
    try:
        n = len(f_sequence)
        if n == 0:
            return None
            
        # Precompute convolution matrix efficiently
        # We'll construct a sparse matrix for the convolution constraints
        c = -np.ones(n)
        
        # Build constraint matrix for convolution
        a_ub_list = []
        b_ub_list = []
        
        # Generate convolution constraints manually for efficiency
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub_list.append(row)
            b_ub_list.append(rhs)

        # Non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)
        
        # Combine all constraints
        a_ub = np.vstack(a_ub_list + [a_ub_nonneg])
        b_ub = np.hstack(b_ub_list + list(b_ub_nonneg))

        # Solve linear program
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            return result.x
        else:
            print('LP optimization failed.')
            return None
            
    except Exception as e:
        print(f"LP solver error: {e}")
        return None

def construct_fallback_direction(sequence, normalized_seq):
    """Construct an alternative direction when LP fails."""
    n = len(sequence)
    if n == 0:
        return None
    
    # Simple heuristic: try to make a sequence with more concentrated mass
    new_seq = np.array(sequence)
    
    # Introduce some variation while preserving the overall shape
    # Add small random noise with some pattern
    noise = np.random.normal(0, 0.01, n)
    
    # Shift towards higher values in a few positions
    indices = np.random.choice(n, min(5, n//4), replace=False)
    for idx in indices:
        noise[idx] = np.abs(noise[idx]) + 0.05
    
    # Apply and clip
    new_seq = np.clip(new_seq + noise, 0, 1000)
    
    # Normalize
    sum_seq = np.sum(new_seq)
    if sum_seq < 1e-10:
        sum_seq = 1e-10
        
    new_seq = new_seq / sum_seq * np.sum(sequence)
    
    return new_seq.tolist()

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    try:
        # Start with a well-chosen initial sequence
        n = random.randint(100, 1000)
        # Initialize with a more structured approach
        initial_sequence = []
        for _ in range(n):
            # Use a mixture of random and structured values
            val = random.random() * 10 + random.random() * 5
            initial_sequence.append(val)
            
        # Normalize slightly
        seq_sum = np.sum(initial_sequence)
        if seq_sum < 1e-10:
            seq_sum = 1e-10
        initial_sequence = [x / seq_sum * 10 for x in initial_sequence]
        
        best_sequence = initial_sequence
        
        # Try multiple iterations to find improvement
        for iteration in range(10):
            # Check if we should add a restart
            if iteration % 3 == 0:
                # Random restart with different structure
                n = random.randint(100, 1000)
                restart_seq = []
                for _ in range(n):
                    val = random.random() * 10 + random.random() * 5
                    restart_seq.append(val)
                seq_sum = np.sum(restart_seq)
                if seq_sum < 1e-10:
                    seq_sum = 1e-10
                restart_seq = [x / seq_sum * 10 for x in restart_seq]
                best_sequence = restart_seq
                
            h_function = get_good_direction_to_move_into(best_sequence)
            
            if h_function is not None:
                best_sequence = h_function
            else:
                # If direction finding fails, try a simple modification
                mod_seq = [x * (1 + 0.01 * (random.random() - 0.5)) for x in best_sequence]
                # Ensure non-negative
                mod_seq = [max(0, x) for x in mod_seq]
                seq_sum = np.sum(mod_seq)
                if seq_sum < 1e-10:
                    seq_sum = 1e-10
                best_sequence = [x / seq_sum * np.sum(best_sequence) for x in mod_seq]

        # Final clean-up to ensure valid constraints
        seq_sum = np.sum(best_sequence)
        if seq_sum < 0.01:
            # If total is too small, create a new valid sequence
            n = random.randint(100, 1000)
            best_sequence = [random.random() for _ in range(n)]
            seq_sum = np.sum(best_sequence)
            best_sequence = [x / seq_sum * 10 for x in best_sequence]
            
        return best_sequence
        
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback to a simple sequence if something goes wrong
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
