# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from typing import List, Optional, Tuple

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000
CACHE_SIZE = 1000

class AutocorrelationOptimizer:
    def __init__(self):
        self.cache = {}
        self.cache_keys = []

    def _get_cache_key(self, sequence: List[float]) -> str:
        """Generate a hashable key for caching."""
        return str(tuple(sequence))

    def _update_cache(self, key: str, value):
        """Update cache with new entry, maintaining size limit."""
        if len(self.cache) >= CACHE_SIZE:
            # Remove oldest entry
            old_key = self.cache_keys.pop(0)
            del self.cache[old_key]
        self.cache[key] = value
        self.cache_keys.append(key)

    def convolve_fft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency with optimized padding."""
        n = len(a)
        # Use next power of 2 for better FFT performance
        fft_size = 1 << (n - 1).bit_length()  # Next power of 2
        if fft_size < 2 * n - 1:
            fft_size *= 2

        # Pad to fft_size
        a_padded = np.pad(a, (0, fft_size - n), 'constant')
        b_padded = np.pad(b, (0, fft_size - n), 'constant')

        # Perform convolution in frequency domain
        a_fft = fft(a_padded)
        b_fft = fft(b_padded)
        conv_result = ifft(a_fft * np.conj(b_fft))
        return np.real(conv_result[:2*n-1])

    def compute_c1_constant(self, sequence: List[float]) -> float:
        """Computes the C1 constant for a given sequence."""
        key = self._get_cache_key(sequence)
        if key in self.cache:
            return self.cache[key]

        n = len(sequence)
        if n == 0:
            result = float('inf')
        else:
            # Compute convolution using FFT
            conv = self.convolve_fft(np.array(sequence), np.array(sequence))
            max_conv = np.max(conv)
            sum_sq = np.sum(sequence)**2

            if sum_sq < 1e-10:
                result = float('inf')
            else:
                c1 = 2 * n * max_conv / sum_sq
                result = c1

        self._update_cache(key, result)
        return result

    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float, n: int) -> Optional[np.ndarray]:
        """Solves the convolution LP for a given sequence and RHS."""
        try:
            # Create the constraint matrix using the convolution structure
            a_ub = []
            b_ub = []

            # Generate convolution constraints
            f_seq = np.array(f_sequence)
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_seq[i]
                a_ub.append(row)
                b_ub.append(rhs)

            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            # Define objective function (we want to minimize negative sum)
            c = -np.ones(n)

            # Solve the linear program
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True})

            if result.success:
                g_sequence = result.x
                # Ensure non-negativity due to numerical errors
                g_sequence = np.maximum(g_sequence, 0)
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence: List[float]) -> Optional[List[float]]:
        """Returns the direction to move into the sequence."""
        try:
            n = len(sequence)
            if n < MIN_SEQ_LENGTH:
                return None

            # Normalize sequence for processing
            sum_sequence = np.sum(sequence)
            if sum_sequence < 1e-10:
                return None

            # Normalize to avoid numerical issues
            normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

            # Compute the target RHS for LP solver
            conv = self.convolve_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(conv)

            # Solve the LP optimization
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs, n)
            if g_fun is None:
                return None

            # Normalize the result and create new sequence
            sum_g = np.sum(g_fun)
            if sum_g < 1e-10:
                return None

            normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g
            t = 0.01
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence
        except Exception:
            return None

    def initialize_sequence(self) -> List[float]:
        """Initialize a promising sequence for optimization."""
        # Start with a structured sequence
        n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        # Create a sequence with decreasing values to encourage sparsity
        sequence = [1.0 / (i + 1) for i in range(n)]
        # Normalize to have reasonable magnitude
        total = sum(sequence)
        sequence = [x * 2.0 / total for x in sequence]
        return sequence

    def search_for_best_sequence(self) -> List[float]:
        """Search for the best coefficient sequence."""
        start_time = time.time()

        # Initialize with a promising sequence
        best_sequence = self.initialize_sequence()
        prev_c1 = self.compute_c1_constant(best_sequence)
        iterations = 0

        while time.time() - start_time < MAX_TIME_SECONDS - 5 and iterations < MAX_ITERATIONS:
            # Try gradient-based improvement
            improved_sequence = self.get_good_direction_to_move_into(best_sequence)

            if improved_sequence is not None:
                new_c1 = self.compute_c1_constant(improved_sequence)
                if new_c1 < prev_c1:
                    best_sequence = improved_sequence
                    prev_c1 = new_c1
                    iterations += 1
                    continue

            # If gradient fails or doesn't improve, try random perturbation
            n = max(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, int(len(best_sequence) * 0.95)))
            if random.random() < 0.5:
                # Random perturbation
                best_sequence = [random.uniform(0.1, 2.0) for _ in range(n)]
            else:
                # Randomly modify some elements
                mutated_sequence = best_sequence.copy()
                for i in range(len(mutated_sequence)):
                    if random.random() < 0.3:
                        mutated_sequence[i] *= random.uniform(0.8, 1.2)
                best_sequence = mutated_sequence

            iterations += 1

        # Final check
        final_c1 = self.compute_c1_constant(best_sequence)
        if final_c1 >= 1.5031:
            # Attempt one final optimization
            refined = self.get_good_direction_to_move_into(best_sequence)
            if refined is not None:
                test_c1 = self.compute_c1_constant(refined)
                if test_c1 < final_c1:
                    best_sequence = refined

        return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main entry point for searching the best sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")