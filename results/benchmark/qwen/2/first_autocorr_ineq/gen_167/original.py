# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import random
import time
from collections import deque

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

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
        return float('-inf')
    return 1.0 / c1

def generate_structured_sequence(length):
    """Generate a structured sequence with good properties."""
    base_sequence = np.random.uniform(0, 100, length)

    if np.random.random() < 0.5:
        idxs = np.random.choice(length, size=min(10, length//4), replace=False)
        base_sequence[idxs] *= np.random.uniform(5, 20)

    if np.random.random() < 0.3:
        threshold = np.random.choice(length)
        base_sequence[threshold:] = 0

    return base_sequence.tolist()

def generate_uniform_sequence(length):
    """Generate a uniform sequence."""
    return [1.0] * length

def generate_step_sequence(length):
    """Generate a step sequence."""
    mid = length // 2
    return [1.0] * mid + [0.0] * (length - mid)

def initialize_sequences(count=10, min_length=100, max_length=1000):
    """Initialize a diverse set of sequences."""
    sequences = []

    # Add known good structures
    sequences.append(generate_uniform_sequence(100))
    sequences.append(generate_step_sequence(100))

    # Add random structured sequences
    for _ in range(count - 2):
        n = random.randint(min_length, max_length)
        seq = generate_structured_sequence(n)
        sequences.append(seq)

    return sequences

def gradient_ascent_step(sequence, step_size=0.01):
    """Perform a gradient ascent step to improve the sequence."""
    try:
        # Convert to numpy array
        seq_array = np.array(sequence, dtype=float)
        n = len(seq_array)

        # Compute convolution to estimate gradient
        convolved = fftconvolve(seq_array, seq_array, mode='full')
        max_conv_index = np.argmax(convolved)
        max_conv_value = np.max(convolved)

        # Estimate gradient direction
        grad_dir = np.zeros(n)
        # Perturb around the maximum convolution index
        window = min(5, n // 2)
        for i in range(max(0, max_conv_index - window), min(n, max_conv_index + window)):
            grad_dir[i] = -0.1  # Slight decrease to reduce peak

        # Apply gradient update
        updated_seq = seq_array + step_size * grad_dir
        updated_seq = np.maximum(updated_seq, 0.0)  # Ensure non-negativity

        # Normalize
        sum_updated = np.sum(updated_seq)
        if sum_updated > 0.01:
            updated_seq = updated_seq / sum_updated
        else:
            updated_seq = updated_seq + 0.01
            updated_seq = updated_seq / np.sum(updated_seq)

        return updated_seq.tolist()
    except Exception:
        # Fallback to random perturbation if gradient fails
        new_sequence = sequence.copy()
        for i in range(len(new_sequence)):
            if random.random() < 0.1:
                new_sequence[i] = max(0, new_sequence[i] + random.uniform(-10, 10))
        return new_sequence

def optimize_with_de(sequence):
    """Optimize using differential evolution for global refinement."""
    bounds = [(0.0, 1000.0)] * len(sequence)
    try:
        result = differential_evolution(
            lambda s: -evaluate_sequence(s),
            bounds,
            maxiter=30,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=random.randint(0, 1000),
            polish=True
        )
        if result.success:
            return result.x.tolist()
    except Exception:
        pass
    return sequence

def search_for_best_sequence():
    """Main search function implementing multi-stage optimization."""
    start_time = time.time()
    max_time_seconds = 170

    # Initialize diverse sequences
    sequences = initialize_sequences()
    best_sequence = None
    best_inv_c1 = float('-inf')

    # History tracking for stagnation detection
    history = deque(maxlen=10)

    for attempt in range(20):  # Multiple attempts
        if time.time() - start_time > max_time_seconds:
            break

        # Select a random initial sequence
        current_sequence = random.choice(sequences)

        # Local optimization loop
        for iteration in range(100):
            if time.time() - start_time > max_time_seconds:
                break

            # Gradient ascent step
            new_sequence = gradient_ascent_step(current_sequence)

            # Evaluate new sequence
            new_inv_c1 = evaluate_sequence(new_sequence)

            # Accept improvement
            if new_inv_c1 > best_inv_c1:
                best_inv_c1 = new_inv_c1
                best_sequence = new_sequence.copy()

            current_sequence = new_sequence

            # Track history
            history.append(new_inv_c1)

            # Stagnation detection and reset
            if len(history) == history.maxlen:
                recent_change = abs(history[-1] - history[0])
                if recent_change < 1e-6:
                    current_sequence = generate_structured_sequence(len(current_sequence))

            # Occasionally perform DE optimization
            if iteration % 10 == 0:
                current_sequence = optimize_with_de(current_sequence)

    # Final DE refinement
    if best_sequence is not None:
        best_sequence = optimize_with_de(best_sequence)

    # Fallback
    if best_sequence is None:
        best_sequence = [1.0] * 100

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")