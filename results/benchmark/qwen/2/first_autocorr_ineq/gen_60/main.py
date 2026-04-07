# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from collections import deque
import time

# Optimization configuration
MAX_ITERATIONS = 500
STAGNATION_THRESHOLD = 1e-6
HISTORY_SIZE = 10
INITIAL_STEP_SIZE = 0.01
ADAPTIVE_DECAY = 0.95

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
        return float('inf')
    # Use FFT-based convolution for efficiency
    convolved = np.convolve(sequence, sequence, mode='full')
    max_conv = np.max(convolved)
    sum_seq = sum(sequence)
    return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

def evaluate_sequence(sequence):
    """Evaluate a sequence by computing 1/C1."""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return float('inf')
    return -1.0 / c1  # We want to maximize 1/C1, so minimize -1/C1

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    if sum_sequence < 1e-10:
        return None

    # Normalize the sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution to find maximum
    convolved = np.convolve(normalized_sequence, normalized_sequence, mode='full')
    rhs = np.max(convolved)

    # Solve the LP problem
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        return None

    # Convert back to original scale
    sum_g_fun = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Apply adaptive step size
    t = INITIAL_STEP_SIZE * (ADAPTIVE_DECAY ** len(sequence))
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build convolution constraints
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            # Fallback to random perturbation
            return np.random.uniform(0, 1, n)
    except Exception:
        # If LP fails completely, return a random sequence
        return np.random.uniform(0, 1, n)

def generate_structured_sequence(length):
    """Generate a structured sequence with good properties."""
    # Start with a base random sequence
    base_sequence = np.random.uniform(0, 100, length)

    # Add some structure
    if np.random.random() < 0.5:
        # Add some large values for diversity
        idxs = np.random.choice(length, size=min(10, length//4), replace=False)
        base_sequence[idxs] *= np.random.uniform(5, 20)

    # Sometimes make it more step-like
    if np.random.random() < 0.3:
        threshold = np.random.choice(length)
        base_sequence[threshold:] = 0

    return base_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Track history for stagnation detection
    history = deque(maxlen=HISTORY_SIZE)

    # Initialize with a diverse set of sequences
    initial_sequences = []

    # Generate random structured sequences
    for _ in range(5):
        n = np.random.randint(100, 500)
        seq = generate_structured_sequence(n)
        initial_sequences.append(seq)

    # Add known good structures
    initial_sequences.append(np.array([1.0] * 100))  # Uniform
    initial_sequences.append(np.array([1.0] * 50 + [0.0] * 50))  # Step function

    # Select initial sequence
    best_sequence = initial_sequences[np.random.randint(len(initial_sequences))]
    best_score = evaluate_sequence(best_sequence)

    # Optimization loop
    start_time = time.time()
    iteration = 0

    while iteration < MAX_ITERATIONS and (time.time() - start_time) < 170:
        # Get direction to move into
        h_function = get_good_direction_to_move_into(best_sequence)

        if h_function is not None:
            # Evaluate the new sequence
            new_score = evaluate_sequence(h_function)

            # Accept improvement or accept with probability
            if new_score > best_score:
                best_sequence = h_function
                best_score = new_score
            else:
                # Occasionally accept worse solutions to escape local minima
                if np.random.random() < 0.05:
                    best_sequence = h_function
                    best_score = new_score
        else:
            # Fallback to simple random perturbation
            n = len(best_sequence)
            perturbed = best_sequence + np.random.normal(0, 0.1, n)
            perturbed = np.maximum(perturbed, 0)
            new_score = evaluate_sequence(perturbed)

            if new_score > best_score:
                best_sequence = perturbed
                best_score = new_score

        # Track history for stagnation detection
        history.append(best_score)

        # Check for stagnation
        if len(history) == HISTORY_SIZE:
            recent_change = abs(history[-1] - history[0])
            if recent_change < STAGNATION_THRESHOLD:
                # Reset with a new random sequence
                n = np.random.randint(100, 500)
                best_sequence = generate_structured_sequence(n)
                best_score = evaluate_sequence(best_sequence)

        iteration += 1

    # Final check to ensure we have a valid result
    if best_score == float('inf'):
        best_sequence = np.array([1.0] * 100)

    return best_sequence.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")