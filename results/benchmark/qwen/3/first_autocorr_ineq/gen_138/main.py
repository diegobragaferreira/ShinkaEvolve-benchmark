# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import time
import random
from scipy import optimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_convolution_fft(seq):
    """Efficiently compute convolution using FFT."""
    n = len(seq)
    padded_len = 2 * n - 1
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant')
    fft_seq = fft(padded_seq)
    conv_fft = fft_seq * fft_seq.conj()
    conv_result = ifft(conv_fft).real[:n]
    return conv_result

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0:
        return float('inf'), 0.0

    a = np.array(sequence, dtype=np.float64)
    n = len(a)

    # Compute convolution using FFT for efficiency
    conv = compute_convolution_fft(a)
    max_b = np.max(conv)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    C1 = 2 * n * max_b / (sum_a ** 2)
    inv_C1 = 1 / C1

    return C1, inv_C1

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP with improved numerical stability and fallbacks."""
    n = len(f_sequence)
    if n == 0:
        return None

    try:
        # Create constraint matrix efficiently using vectorized operations
        a_ub = np.zeros((2 * n - 1, n))
        b_ub = np.full(2 * n - 1, rhs)
        
        # Generate convolution constraints efficiently
        for k in range(2 * n - 1):
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    a_ub[k, j] = f_sequence[i]

        # Non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        # Combine all constraints
        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Objective: minimize sum of variables (which corresponds to maximizing 1/C1)
        c = np.ones(n)

        # Solve with robust solver settings
        result = optimize.linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            method='highs',
            options={'presolve': True, 'time_limit': 5}
        )

        if result.success:
            return result.x
        else:
            # Fallback to simplex if highs fails
            try:
                result = optimize.linprog(
                    c,
                    A_ub=a_ub,
                    b_ub=b_ub,
                    method='simplex'
                )
                if result.success:
                    return result.x
            except:
                pass
            return None

    except Exception as e:
        return None

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence using LP optimization.
    """
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None

    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

    try:
        # Compute maximum convolution value using FFT
        conv = compute_convolution_fft(normalized_sequence)
        rhs = np.max(conv)

        # Solve LP optimization
        g_fun = solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None or np.any(np.isnan(g_fun)):
            return None

        # Normalize the resulting sequence
        sum_g = np.sum(g_fun)
        if sum_g < 1e-8:
            return None

        normalized_g_fun = g_fun * np.sqrt(2 * n) / sum_g

        # Apply perturbation to maintain diversity
        t = 0.05
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

    except Exception as e:
        return None

def adaptive_frequency_optimize(current_sequence, max_iter=50):
    """
    Optimizes sequence using a hybrid approach combining frequency and time-domain optimization.
    """
    n = len(current_sequence)
    if n == 0:
        return current_sequence

    # Ensure minimal positive values to prevent numerical issues
    a = np.array(current_sequence, dtype=float)
    a = np.maximum(a, 1e-10)

    # First, try the LP-based optimization approach
    lp_direction = get_good_direction_to_move_into(current_sequence)
    if lp_direction is not None:
        # Use LP direction as a good starting point
        current_sequence = lp_direction

    # Convert to frequency domain
    a_fft = fft(a, 2*n-1)

    # Objective function in frequency domain
    def objective(freq_vals):
        # Convert back to time domain
        reconstructed = ifft(freq_vals, 2*n-1).real[:n]
        reconstructed = np.maximum(reconstructed, 1e-10)

        # Compute convolution using FFT for better performance
        conv = compute_convolution_fft(reconstructed)
        max_conv = np.max(conv)
        sum_reconstructed = np.sum(reconstructed)

        if sum_reconstructed < 0.01:
            return float('inf')

        # Minimize max(conv) / (sum^2) to maximize 1/C1
        C1 = 2 * n * max_conv / (sum_reconstructed ** 2)
        return C1

    # Use L-BFGS-B for better convergence
    try:
        initial_freq = a_fft.copy()
        result = minimize(objective, initial_freq, method='L-BFGS-B', options={'maxiter': max_iter})
        if result.success:
            optimized_freq = result.x
            optimized_time = ifft(optimized_freq, 2*n-1).real[:n]
            optimized_time = np.maximum(optimized_time, 1e-10)
            return optimized_time.tolist()
    except Exception as e:
        pass

    # Fallback to simple perturbation if optimization fails
    new_sequence = current_sequence.copy()
    idx = np.random.randint(0, n)
    new_sequence[idx] = max(1e-10, new_sequence[idx] + np.random.normal(0, 0.1))
    return new_sequence

def evaluate_initial_sequence(seq):
    """Evaluate a single initial sequence in parallel."""
    c1, inv_c1 = compute_autocorrelation_constant(seq)
    return seq, c1, inv_c1

def generate_initial_sequences():
    """Generate diverse initial sequences."""
    initial_sequences = []
    
    # Random sequences
    for _ in range(2):
        n = np.random.randint(50, 200)
        seq = np.random.uniform(0.1, 1.0, n).tolist()
        initial_sequences.append(seq)

    # Exponential decay sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        seq = [0.8 ** i for i in range(n)]
        initial_sequences.append(seq)

    # Spike sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        seq = [0.0] * n
        spike_idx = np.random.randint(0, n)
        seq[spike_idx] = 1.0
        initial_sequences.append(seq)

    # Gaussian-like sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        center = n // 2
        seq = [np.exp(-0.5 * ((i - center)**2) / (n/10)**2) for i in range(n)]
        initial_sequences.append(seq)
        
    return initial_sequences

def multi_start_optimization_parallel(initial_sequences, max_time):
    """
    Performs multi-start optimization using parallel evaluation of initial sequences.
    """
    best_inv_C1 = 0.0
    best_sequence = None
    best_C1 = float('inf')
    start_time = time.time()

    # Parallel evaluation of initial sequences
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        future_to_seq = {executor.submit(evaluate_initial_sequence, seq): seq for seq in initial_sequences}
        results = []
        for future in as_completed(future_to_seq):
            seq, c1, inv_c1 = future.result()
            results.append((seq, c1, inv_c1))
            if inv_c1 > best_inv_C1:
                best_inv_C1 = inv_c1
                best_sequence = seq.copy()
                best_C1 = c1

    # Continue optimization with best starting point
    if best_sequence is not None:
        # Multiple rounds of optimization with different strategies
        current_seq = best_sequence.copy()
        current_C1, current_inv_C1 = best_C1, best_inv_C1

        for round_num in range(3):  # Multiple rounds
            if time.time() - start_time > max_time - 5:
                break

            # Use different optimization strategies
            if round_num == 0:
                # Direct optimization
                improved_seq = adaptive_frequency_optimize(current_seq, max_iter=30)
            elif round_num == 1:
                # LP-based direction
                lp_direction = get_good_direction_to_move_into(current_seq)
                if lp_direction is not None:
                    improved_seq = lp_direction
                else:
                    improved_seq = adaptive_frequency_optimize(current_seq, max_iter=20)
            else:
                # Random perturbation followed by optimization
                perturbed_seq = current_seq.copy()
                idx = np.random.randint(0, len(current_seq))
                perturbed_seq[idx] = max(1e-10, perturbed_seq[idx] + np.random.normal(0, 0.05))
                improved_seq = adaptive_frequency_optimize(perturbed_seq, max_iter=20)

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
    
    # Multi-start optimization with parallel processing
    best_sequence, best_C1, best_inv_C1 = multi_start_optimization_parallel(initial_sequences, max_time)

    # Final optimization with enhanced parameters
    if best_sequence is not None:
        final_seq = adaptive_frequency_optimize(best_sequence, max_iter=100)
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