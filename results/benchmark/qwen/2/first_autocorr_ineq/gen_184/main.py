# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize, signal
from scipy.fft import fft, ifft
import time
from typing import List, Optional

# Constants
MAX_TIME_SECONDS = 180
FFT_THRESHOLD = 100  # Use FFT for sequences longer than this

def autocorrelation_constant(sequence: List[float]) -> float:
    """
    Calculates C₁ = 2n * max(b) / (sum(a))^2 where b = a * a (autoconvolution).
    Returns the inverse 1/C₁ which we want to maximize.
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    sum_a = sum(sequence)
    if sum_a < 0.01:
        return 0.0

    # Compute autoconvolution using FFT for efficiency
    if n > FFT_THRESHOLD:
        # Use FFT for fast convolution
        padded_len = 2 * n - 1
        seq_fft = fft(sequence, padded_len)
        conv_fft = seq_fft * seq_fft.conj()  # Element-wise multiplication
        autoconv = ifft(conv_fft).real
        max_conv = max(autoconv)
    else:
        # Direct convolution for small sequences
        autoconv = signal.convolve(sequence, sequence, mode='full')
        max_conv = max(autoconv)

    # Calculate C₁
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    if c1 == 0:
        return 0.0
    return 1.0 / c1

def compute_autocorr_gradient(sequence: List[float], epsilon: float = 1e-4) -> List[float]:
    """
    Approximate gradient using finite differences, adapted for autocorrelation functions.
    """
    n = len(sequence)
    grad = []
    for i in range(n):
        # Perturb dimension i
        perturbed_plus = sequence[:]
        perturbed_minus = sequence[:]
        perturbed_plus[i] += epsilon
        perturbed_minus[i] -= epsilon
        
        # Ensure non-negativity
        perturbed_plus[i] = max(0, perturbed_plus[i])
        perturbed_minus[i] = max(0, perturbed_minus[i])
        
        # Evaluate function
        f_plus = autocorrelation_constant(perturbed_plus)
        f_minus = autocorrelation_constant(perturbed_minus)
        
        grad_i = (f_plus - f_minus) / (2 * epsilon)
        grad.append(grad_i)
    
    return grad

def solve_convolution_lp(f_sequence: list[float], rhs: float) -> list[float] | None:
    """Solves the convolution LP for a given sequence and RHS with better numerical stability."""
    try:
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

        # Use a more robust method with increased tolerance
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': False})
        
        if result.success:
            g_sequence = result.x
            return g_sequence.tolist()
        else:
            return None
    except Exception:
        return None

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using a quasi-Newton method."""
    start_time = time.time()
    
    n = len(sequence)
    if n == 0:
        return None

    # Normalize the input sequence
    sum_sequence = sum(sequence)
    if sum_sequence < 0.01:
        sum_sequence = 0.01
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute the right-hand side for convolution constraints
    max_conv_value = max(signal.convolve(normalized_sequence, normalized_sequence, mode='full')[n-1:])
    rhs = max_conv_value

    # Solve the LP to get the direction vector g
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    
    if g_fun is None:
        # Fallback to BFGS-based optimization if LP fails
        try:
            # Define objective function for BFGS
            def objective(x):
                return -autocorrelation_constant(x.tolist())
            
            # Initial guess
            x0 = np.array(normalized_sequence)
            
            # Run BFGS optimization
            result = optimize.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * len(x0),
                options={'maxiter': 100}
            )
            
            if result.success:
                return result.x.tolist()
            else:
                return None
        except:
            return None
        
        return None

    # Normalize g_fun similarly
    sum_g_fun = sum(g_fun)
    if sum_g_fun < 0.01:
        sum_g_fun = 0.01
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Update sequence using a fixed step size (t=0.01)
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    return new_sequence

def search_for_best_sequence() -> list[float]:
    """Main search function to find the best coefficient sequence."""
    start_time = time.time()
    
    # Initialize a random sequence with varying length
    n = np.random.randint(100, 1000)
    best_sequence = [np.random.random() for _ in range(n)]
    
    # Iteratively improve the sequence
    for _ in range(100):  # Limited iterations to respect time budget
        if time.time() - start_time > MAX_TIME_SECONDS - 2:
            break
            
        improved_seq = get_good_direction_to_move_into(best_sequence)
        if improved_seq is not None:
            best_sequence = improved_seq
        else:
            # Fallback: modify the sequence slightly
            idx = np.random.randint(0, len(best_sequence))
            best_sequence[idx] = (best_sequence[idx] + np.random.rand()) % 1
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
