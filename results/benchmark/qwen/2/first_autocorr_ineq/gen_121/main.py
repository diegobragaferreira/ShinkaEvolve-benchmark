# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
from typing import List, Optional, Tuple
from collections import deque
import random

class ConvolutionComputer:
    """Encapsulates convolution-related computations for efficiency and clarity."""
    
    @staticmethod
    def compute_c1(sequence: List[float]) -> float:
        """Compute C1 for a given sequence."""
        if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
            return float('inf')
        # Use FFT-based convolution for efficiency
        convolved = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(convolved)
        sum_seq = sum(sequence)
        return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

    @staticmethod
    def compute_convolution(sequence: List[float]) -> np.ndarray:
        """Compute the convolution of a sequence with itself."""
        return fftconvolve(sequence, sequence, mode='full')

class SequenceOptimizer:
    """Handles sequence optimization logic with adaptive strategies."""
    
    def __init__(self, max_iterations: int = 1000, adaptive_decay_rate: float = 0.95,
                 min_length: int = 100, max_length: int = 1000, time_limit: int = 170):
        self.max_iterations = max_iterations
        self.adaptive_decay_rate = adaptive_decay_rate
        self.min_length = min_length
        self.max_length = max_length
        self.time_limit = time_limit

    def evaluate_sequence(self, sequence: List[float]) -> float:
        """Evaluate a sequence by computing 1/C1."""
        c1 = ConvolutionComputer.compute_c1(sequence)
        if c1 == float('inf') or c1 > 1e10:
            return float('-inf')  # Penalize invalid sequences
        return -1.0 / c1  # We want to maximize 1/C1, so minimize -1/C1

    def get_good_direction_to_move_into(self, sequence: List[float]) -> Optional[List[float]]:
        """Determine the best movement direction for the sequence."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)

        if sum_sequence < 1e-10:
            return None

        # Normalize the sequence
        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Compute convolution to find maximum
        convolved = ConvolutionComputer.compute_convolution(normalized_sequence)
        rhs = np.max(convolved)

        # Solve the LP problem
        g_fun = self.solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None:
            # Fallback to simple gradient estimation if LP fails
            return self.get_simple_gradient_direction(sequence, normalized_sequence, rhs)

        # Convert back to original scale
        sum_g_fun = np.sum(g_fun)
        if sum_g_fun < 1e-10:
            return None
        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

        # Apply adaptive step size
        t = 0.01 * (self.adaptive_decay_rate ** len(sequence))
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        return new_sequence

    def solve_convolution_lp(self, f_sequence: List[float], rhs: float) -> Optional[np.ndarray]:
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build convolution constraints efficiently
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
                return result.x
            else:
                return None
        except Exception:
            return None

    def get_simple_gradient_direction(self, sequence: List[float], 
                                     normalized_sequence: List[float], 
                                     rhs: float) -> List[float]:
        """Provides a simple gradient direction when LP fails."""
        n = len(sequence)
        # Estimate gradient by perturbing elements that contribute most to max convolution
        convolved = ConvolutionComputer.compute_convolution(normalized_sequence)
        max_pos = np.argmax(convolved)

        # Simple gradient update: decrease values around max position
        grad_dir = np.zeros(n)
        half_window = min(10, n // 2)
        for i in range(max(0, max_pos - half_window), min(n, max_pos + half_window)):
            grad_dir[i] = -0.1  # Small negative gradient

        # Apply to original sequence
        t = 0.005
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, grad_dir)
        ]

        # Ensure non-negativity
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

def generate_structured_sequence(length: int) -> List[float]:
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

    return base_sequence.tolist()

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    best_sequence = None
    best_score = float('-inf')
    
    optimizer = SequenceOptimizer()

    # Initialize with multiple diverse sequences
    initial_sequences = []

    # Add known good structures
    initial_sequences.append([1.0] * 100)  # Uniform
    initial_sequences.append([1.0] * 50 + [0.0] * 50)  # Step function

    # Add random structured sequences
    for _ in range(5):
        n = np.random.randint(optimizer.min_length, optimizer.max_length)
        seq = generate_structured_sequence(n)
        initial_sequences.append(seq)

    # Try multiple starting points
    for attempt in range(10):
        if time.time() - start_time > optimizer.time_limit:
            break

        try:
            # Select a random initial sequence
            initial_seq = initial_sequences[np.random.randint(len(initial_sequences))]

            # Ensure minimum length
            if len(initial_seq) < optimizer.min_length:
                initial_seq = initial_seq + [0.0] * (optimizer.min_length - len(initial_seq))

            current_seq = initial_seq[:]
            current_score = optimizer.evaluate_sequence(current_seq)

            # Optimization loop
            for iteration in range(optimizer.max_iterations):
                if time.time() - start_time > optimizer.time_limit:
                    break

                # Get direction to move into
                h_function = optimizer.get_good_direction_to_move_into(current_seq)

                if h_function is not None:
                    # Evaluate the new sequence
                    new_score = optimizer.evaluate_sequence(h_function)

                    # Accept improvement or accept with probability
                    if new_score > current_score:
                        current_seq = h_function
                        current_score = new_score
                    else:
                        # Occasionally accept worse solutions to escape local minima
                        if np.random.random() < 0.05:
                            current_seq = h_function
                            current_score = new_score
                else:
                    # Fallback to random perturbation
                    n = len(current_seq)
                    perturbed = [max(0, x + np.random.normal(0, 0.1)) for x in current_seq]
                    new_score = optimizer.evaluate_sequence(perturbed)
                    if new_score > current_score:
                        current_seq = perturbed
                        current_score = new_score

                # Track best sequence found
                if current_score > best_score and current_score != float('-inf'):
                    best_score = current_score
                    best_sequence = current_seq[:]

        except Exception:
            continue

    # Return best sequence found
    if best_sequence is None:
        best_sequence = [1.0] * 100  # Fallback to uniform sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")