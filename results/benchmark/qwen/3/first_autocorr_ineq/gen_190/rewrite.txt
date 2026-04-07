# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import time
import random
from scipy import optimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@lru_cache(maxsize=1024)
def cached_fft_convolve(seq_tuple):
    """Cached FFT-based convolution for performance."""
    seq = np.array(seq_tuple, dtype=np.float64)
    n = len(seq)
    padded_len = 2 * n - 1
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant')
    fft_seq = fft(padded_seq)
    conv_fft = fft_seq * fft_seq.conj()
    conv_result = ifft(conv_fft).real[:n]
    return tuple(conv_result)

def compute_convolution_fft_cached(seq):
    """Efficiently compute convolution using FFT with caching."""
    return cached_fft_convolve(tuple(seq))

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0:
        return float('inf'), 0.0

    a = np.array(sequence, dtype=np.float64)
    n = len(a)

    # Compute convolution using FFT
    conv = compute_convolution_fft_cached(a)
    max_b = np.max(conv)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    C1 = 2 * n * max_b / (sum_a ** 2)
    inv_C1 = 1 / C1

    return C1, inv_C1

def initialize_sequence(n=None, method='gaussian'):
    """
    Initialize a sequence based on known mathematical properties.
    """
    if n is None:
        n = np.random.randint(50, 500)
    
    if method == 'gaussian':
        center = n // 2
        seq = [np.exp(-0.5 * ((i - center)**2) / (n/10)**2) for i in range(n)]
    elif method == 'exponential':
        seq = [0.8 ** i for i in range(n)]
    elif method == 'spike':
        seq = [0.0] * n
        spike_idx = np.random.randint(0, n)
        seq[spike_idx] = 1.0
    else:  # random
        seq = np.random.uniform(0.1, 1.0, n).tolist()
    
    return seq

def get_good_direction_to_move_into(sequence, iteration=0):
    """
    Returns the direction to move into the sequence using a simulated annealing-inspired approach.
    """
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None

    # Normalize
    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence
    
    # Compute convolution and estimate the maximum
    conv = compute_convolution_fft_cached(normalized_sequence)
    rhs = np.max(conv)
    
    # Simulated Annealing approach for direction finding
    try:
        # Generate candidate directions around the current sequence
        T = 1.0 / (1 + iteration)  # Temperature decreases over iterations
        candidates = []
        
        for _ in range(10):  # Generate 10 candidates
            candidate = normalized_sequence.copy()
            # Small random perturbations
            for i in range(n):
                if np.random.rand() < 0.3:
                    delta = np.random.normal(0, 0.05) * T
                    candidate[i] = max(0, candidate[i] + delta)
            
            # Ensure it's a valid sequence
            if np.sum(candidate) > 0.01:
                candidates.append(candidate)
        
        # Evaluate candidates and find the best
        best_candidate = None
        best_c1 = float('inf')
        for candidate in candidates:
            c1, _ = compute_autocorrelation_constant(candidate)
            if c1 < best_c1:
                best_c1 = c1
                best_candidate = candidate
        
        if best_candidate is None:
            return None
            
        # Normalize the best candidate
        sum_best = np.sum(best_candidate)
        normalized_best = best_candidate * np.sqrt(2 * n) / sum_best
        
        # Apply perturbation
        t = 0.05 * (1 - min(iteration / 100, 0.9))
        new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_best)]
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

    except Exception as e:
        return None

def adaptive_frequency_optimize(current_sequence, max_iter=50, iteration=0):
    """
    Optimizes sequence using a hybrid gradient-free method.
    """
    n = len(current_sequence)
    if n == 0:
        return current_sequence

    # Ensure minimal positive values to prevent numerical issues
    a = np.array(current_sequence, dtype=float)
    a = np.maximum(a, 1e-10)
    
    # Try the SA-based approach first
    sa_direction = get_good_direction_to_move_into(current_sequence, iteration)
    if sa_direction is not None:
        current_sequence = sa_direction

    # Local search using Nelder-Mead for fine tuning
    try:
        # Define objective function to minimize C1
        def objective(sequence):
            c1, _ = compute_autocorrelation_constant(sequence)
            return c1
        
        # Use Nelder-Mead algorithm for local optimization
        result = minimize(objective, current_sequence, method='Nelder-Mead', options={'maxiter': max_iter//2})
        if result.success:
            return result.x.tolist()
    except Exception as e:
        pass

    # Fallback to simple perturbation if optimization fails
    new_sequence = current_sequence.copy()
    idx = np.random.randint(0, n)
    noise = np.random.normal(0, 0.1) * (1 - min(iteration / 100, 0.9))
    new_sequence[idx] = max(1e-10, new_sequence[idx] + noise)
    return new_sequence

def evaluate_initial_sequence(seq):
    """Evaluate a single initial sequence."""
    c1, inv_c1 = compute_autocorrelation_constant(seq)
    return seq, c1, inv_c1

def generate_initial_sequences(count=8):
    """Generate diverse initial sequences."""
    initial_sequences = []
    
    # Mix of sequence types
    for _ in range(count // 4):
        n = np.random.randint(50, 200)
        seq = initialize_sequence(n, 'random')
        initial_sequences.append(seq)
    
    for _ in range(count // 4):
        n = np.random.randint(100, 500)
        seq = initialize_sequence(n, 'gaussian')
        initial_sequences.append(seq)
        
    for _ in range(count // 4):
        n = np.random.randint(100, 500)
        seq = initialize_sequence(n, 'exponential')
        initial_sequences.append(seq)
        
    for _ in range(count // 4):
        n = np.random.randint(100, 500)
        seq = initialize_sequence(n, 'spike')
        initial_sequences.append(seq)
        
    return initial_sequences

def multi_start_optimization_parallel(initial_sequences, max_time):
    """
    Performs multi-start optimization with enhanced strategies.
    """
    best_inv_C1 = 0.0
    best_sequence = None
    best_C1 = float('inf')
    start_time = time.time()

    # Evaluate initial sequences
    for seq in initial_sequences:
        if time.time() - start_time > max_time - 5:
            break
        _, c1, inv_c1 = compute_autocorrelation_constant(seq)
        if inv_c1 > best_inv_C1:
            best_inv_C1 = inv_c1
            best_sequence = seq.copy()
            best_C1 = c1

    # Continue optimization with best starting point
    if best_sequence is not None:
        current_seq = best_sequence.copy()
        current_C1, current_inv_C1 = best_C1, best_inv_C1

        for round_num in range(3):  # Multiple rounds
            if time.time() - start_time > max_time - 5:
                break

            # Use different optimization strategies
            if round_num == 0:
                # Direct optimization
                improved_seq = adaptive_frequency_optimize(current_seq, max_iter=30, iteration=round_num)
            elif round_num == 1:
                # SA-based direction
                sa_direction = get_good_direction_to_move_into(current_seq, iteration=round_num)
                if sa_direction is not None:
                    improved_seq = sa_direction
                else:
                    improved_seq = adaptive_frequency_optimize(current_seq, max_iter=20, iteration=round_num)
            else:
                # Random perturbation followed by optimization
                perturbed_seq = current_seq.copy()
                idx = np.random.randint(0, len(current_seq))
                perturbed_seq[idx] = max(1e-10, perturbed_seq[idx] + np.random.normal(0, 0.05))
                improved_seq = adaptive_frequency_optimize(perturbed_seq, max_iter=20, iteration=round_num)

            improved_C1, improved_inv_C1 = compute_autocorrelation_constant(improved_seq)

            if improved_inv_C1 > current_inv_C1:
                current_seq = improved_seq
                current_C1 = improved_C1
                current_inv_C1 = improved_inv_C1

                if current_inv_C1 > best_inv_C1:
                    best_inv_C1 = current_inv_C1
                    best_sequence = current_seq.copy()
                    best_C1 = current_C1

    return best_sequence, best_C1, best_inv_C1

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    start_time = time.time()
    max_time = 180  # seconds

    # Generate diverse initial sequences
    initial_sequences = generate_initial_sequences()
    
    # Multi-start optimization
    best_sequence, best_C1, best_inv_C1 = multi_start_optimization_parallel(initial_sequences, max_time)

    # Final optimization with enhanced parameters
    if best_sequence is not None:
        final_seq = adaptive_frequency_optimize(best_sequence, max_iter=100, iteration=100)
        final_C1, final_inv_C1 = compute_autocorrelation_constant(final_seq)
        if final_inv_C1 > best_inv_C1:
            best_sequence = final_seq
            best_C1 = final_C1
            best_inv_C1 = final_inv_C1

    return best_sequence if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")