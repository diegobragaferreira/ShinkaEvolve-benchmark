# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from scipy.signal import convolve
import time
import random
import copy
from typing import List, Tuple, Optional

# Constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
FFT_THRESHOLD = 100  # Threshold for switching to FFT-based convolution
CONVERGENCE_TOLERANCE = 1e-6
MAX_STAGNANT_ITERATIONS = 50
ELITE_SIZE = 5  # Number of top sequences to preserve

def validate_and_normalize_sequence(sequence: list[float]) -> tuple[list[float], float]:
    """Validate input sequence and normalize it for further computations."""
    if not sequence:
        raise ValueError("Empty sequence provided.")

    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        sequence = [0.01] + [0.0] * (len(sequence) - 1) if len(sequence) > 1 else [0.01]
        sum_seq = 0.01

    return sequence, sum_seq

def fast_convolution(sequence: list[float]) -> list[float]:
    """Efficiently compute convolution using FFT if applicable."""
    n = len(sequence)
    if n > FFT_THRESHOLD:
        padded_len = 2 * n - 1
        seq_fft = fft(sequence, padded_len)
        conv_fft = seq_fft * seq_fft.conj()
        autoconv = ifft(conv_fft).real
        return autoconv.tolist()
    else:
        # Use standard convolution for small sequences
        return np.convolve(sequence, sequence, mode='full').tolist()

def compute_max_convolution(sequence: list[float]) -> float:
    """Compute maximum value of autoconvolution."""
    conv_result = fast_convolution(sequence)
    n = len(sequence)
    return max(conv_result[n - 1:])

def compute_c1(sequence: list[float]) -> float:
    """Calculate C₁ constant."""
    n = len(sequence)
    sequence, sum_seq = validate_and_normalize_sequence(sequence)

    if sum_seq < 0.01:
        return float('inf')

    max_conv = compute_max_convolution(sequence)
    c1 = (2 * n * max_conv) / (sum_seq ** 2)
    return c1

def compute_inv_c1(sequence: list[float]) -> float:
    """Calculate inverse of C₁ (what we aim to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0.0
    return 1.0 / c1

def solve_convolution_lp(f_sequence: list[float], rhs: float) -> list[float] | None:
    """Solve the convolution LP constraint problem efficiently."""
    n = len(f_sequence)
    if n == 0:
        return None

    try:
        # Build the A_ub matrix efficiently
        a_ub_rows = []
        for k in range(2 * n - 1):
            row = [0.0] * n
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub_rows.append(row)
        
        # Append non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        a_ub_rows.extend(a_ub_nonneg.tolist())
        
        a_ub = np.array(a_ub_rows)
        b_ub = np.array([rhs] * (2 * n - 1) + [0.0] * n)
        c = -np.ones(n)  # Coefficients for objective function (minimize negative sum)

        # Try multiple solvers for robustness
        solvers = ['highs', 'interior-point', 'revised simplex']
        for solver in solvers:
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method=solver)
                if result.success:
                    return result.x.tolist()
            except:
                continue
        
        return None
    except Exception as e:
        return None

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence using enhanced strategies."""
    start_time = time.time()
    
    if not sequence:
        return None
    
    # Validate and normalize sequence
    sequence, sum_sequence = validate_and_normalize_sequence(sequence)
    
    # Normalize the input sequence for the LP problem
    n = len(sequence)
    if sum_sequence < 0.01:
        sum_sequence = 0.01  # Prevent division by zero
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    
    # Compute the right-hand side for convolution constraints
    max_conv_value = compute_max_convolution(normalized_sequence)
    rhs = max_conv_value
    
    # Solve the LP to get the direction vector g
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        # Multi-tiered fallback strategies
        # Tier 1: Try symmetric initialization
        try:
            g_fun = [normalized_sequence[i] for i in range(n)]
            g_fun = [max(0, x) for x in g_fun]
        except:
            # Tier 2: Random perturbation
            g_fun = [random.uniform(0, 1) for _ in range(n)]
    
    if g_fun is None:
        return None
    
    # Normalize g_fun similarly
    sum_g_fun = sum(g_fun)
    if sum_g_fun < 0.01:
        sum_g_fun = 0.01
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    
    # Update sequence using an adaptive step size
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    
    return new_sequence

def initialize_good_sequence(length: int = None) -> list[float]:
    """Initialize a good starting sequence based on theoretical insights."""
    if length is None:
        length = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    
    # Use exponential decay pattern with some randomness
    sequence = [1.0 * (0.95 ** i) for i in range(length)]
    sequence = [max(x, 0.01) for x in sequence]
    
    # Add randomness to avoid local optima
    noise_factor = 0.1
    sequence = [x * (1 + random.uniform(-noise_factor, noise_factor)) for x in sequence]
    sequence = [max(x, 0.01) for x in sequence]
    
    return sequence

def search_for_best_sequence() -> list[float]:
    """Main search function to find the best coefficient sequence."""
    start_time = time.time()
    
    # Initialize elite sequences
    elites = []
    elite_scores = []
    
    # Try multiple random starting points with good initialization
    for attempt in range(20):  # Increased attempts
        n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        sequence = initialize_good_sequence(n)
        
        current_inv_c1 = compute_inv_c1(sequence)
        
        # Keep elite sequences
        if current_inv_c1 > 0.01:
            elites.append(copy.deepcopy(sequence))
            elite_scores.append(current_inv_c1)
            
            # Keep top 5 elites
            sorted_indices = np.argsort(elite_scores)[::-1][:ELITE_SIZE]
            elites = [elites[i] for i in sorted_indices]
            elite_scores = [elite_scores[i] for i in sorted_indices]
        
        # Evolve the sequence
        current_seq = sequence[:]
        for gen in range(10):
            if time.time() - start_time > MAX_TIME_SECONDS - 2:
                break
            
            improved_seq = get_good_direction_to_move_into(current_seq)
            if improved_seq is not None:
                current_seq = improved_seq
                current_inv_c1 = compute_inv_c1(current_seq)
                
                # Preserve elite
                if current_inv_c1 > 0.01:
                    elites.append(copy.deepcopy(current_seq))
                    elite_scores.append(current_inv_c1)
                    
                    # Keep top 5 elites
                    sorted_indices = np.argsort(elite_scores)[::-1][:ELITE_SIZE]
                    elites = [elites[i] for i in sorted_indices]
                    elite_scores = [elite_scores[i] for i in sorted_indices]
            else:
                # Fallback: slight random perturbation
                idx = random.randint(0, len(current_seq) - 1)
                current_seq[idx] = max(0, current_seq[idx] + random.uniform(-0.05, 0.05))
    
    # Select the best among elites
    best_sequence = None
    best_inv_c1 = 0
    
    if elites:
        best_index = np.argmax(elite_scores)
        best_sequence = elites[best_index]
        best_inv_c1 = elite_scores[best_index]
    
    # Final optimization if needed
    if best_sequence is not None:
        # Local refinement
        try:
            def objective_func(seq_array):
                return -compute_inv_c1(seq_array.tolist())
            
            x0 = np.array(best_sequence, dtype=float)
            result = optimize.minimize(
                objective_func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 20, 'adaptive': True}
            )
            
            if result.success:
                refined_seq = np.maximum(result.x, 0)
                if np.sum(refined_seq) < 0.01:
                    refined_seq[0] = 0.1
                refined_inv_c1 = compute_inv_c1(refined_seq.tolist())
                if refined_inv_c1 > best_inv_c1:
                    best_sequence = refined_seq.tolist()
        except:
            pass
    
    # Return best found sequence or a default
    if best_sequence is None:
        best_sequence = initialize_good_sequence(200)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")