# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

# Constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
FFT_THRESHOLD = 100  # Threshold for switching to FFT-based convolution
CONVERGENCE_TOLERANCE = 1e-6
MAX_STAGNANT_ITERATIONS = 50

def validate_and_normalize_sequence(sequence: list[float]) -> tuple[list[float], float]:
    """Validate input sequence and normalize it for further computations."""
    if not sequence:
        raise ValueError("Empty sequence provided.")

    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        # Return a minimal valid sequence if sum is too small
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
    # Return the maximum from the relevant part (after shifting)
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
    """Solve the convolution LP constraint problem efficiently using FFT."""
    n = len(f_sequence)
    if n == 0:
        return None

    try:
        # Generate convolution constraints using FFT for efficiency
        # Use FFT to compute convolution constraints
        padded_len = 2 * n - 1
        f_fft = fft(f_sequence, padded_len)
        # Precompute all constraint rows using FFT-based convolution
        a_ub_rows = []
        for k in range(2 * n - 1):
            # Construct the kernel for convolution at index k
            kernel = [0.0] * padded_len
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    kernel[j] = f_sequence[i]
            # Convert kernel to frequency domain
            kernel_fft = fft(kernel, padded_len)
            # Multiply with f_fft
            conv_fft = f_fft * kernel_fft.conj()
            # Inverse transform to get the convolution result
            conv = ifft(conv_fft).real
            # Extract relevant part (first n elements for constraint row)
            row = conv[:n].tolist()
            a_ub_rows.append(row)

        # Append non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        a_ub_rows.extend(a_ub_nonneg.tolist())

        a_ub = np.array(a_ub_rows)
        b_ub = np.array([rhs] * (2 * n - 1) + [0.0] * n)
        c = -np.ones(n)  # Coefficients for objective function (minimize negative sum)

        # Warm start: use previous solution if available (optional for future extensions)
        # For now, solve directly with defaults

        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            return result.x.tolist()
        else:
            return None
    except Exception as e:
        # Log error and return None
        return None

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using optimized strategies."""
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
        # Fallback to simple modification
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
    n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
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