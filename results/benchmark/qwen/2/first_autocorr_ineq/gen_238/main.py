# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from collections import deque

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class AdaptiveConvolutionOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180, elite_size=50):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.elite_size = elite_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self.history = deque(maxlen=10)

    def convolve_fft(self, a, b):
        """Compute convolution using FFT for better performance."""
        n = len(a)
        # Zero-pad to avoid circular convolution effects
        padded_length = 2 * n - 1
        fa = fft(a, padded_length)
        fb = fft(b, padded_length)
        result = ifft(fa * fb).real
        return result[:n]

    def compute_c1(self, sequence):
        """Compute the C1 constant from the sequence."""
        if len(sequence) == 0:
            return float('inf')
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')

        convolved = self.convolve_fft(sequence, sequence)
        max_conv = np.max(convolved)
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

    def compute_inv_c1(self, sequence):
        """Compute inverse of C1 (the value we want to maximize)."""
        c1 = self.compute_c1(sequence)
        return 1.0 / c1 if c1 > 0 else 0.0

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        try:
            n = len(f_sequence)
            c = -np.ones(n)
            a_ub = []
            b_ub = []

            # Build convolution constraints using FFT-based approach
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)

            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)

            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            # Solve linear program
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                return result.x
            else:
                return None

        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence, iteration):
        """Returns the direction to move into the sequence."""
        try:
            n = len(sequence)
            sum_sequence = np.sum(sequence)
            if sum_sequence < 1e-10:
                return None

            # Normalize sequence
            normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

            # Compute maximum convolution value
            convolved = self.convolve_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(convolved)

            # Solve LP
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
            if g_fun is None:
                return None

            # Normalize g_fun
            sum_g_fun = np.sum(g_fun)
            if sum_g_fun < 1e-10:
                return None

            normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

            # Adaptive learning rate based on iteration count
            t = 0.05 * (0.98 ** iteration)
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence

        except Exception:
            return None

    def generate_structured_sequence(self, length):
        """Generate a structured sequence with good properties."""
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

    def generate_initial_sequence(self):
        """Generate an initial random sequence."""
        n = random.randint(100, 1000)
        return self.generate_structured_sequence(n)

    def optimize_sequence(self):
        """Main optimization loop."""
        start_time = time.time()

        # Generate initial sequences
        initial_sequences = []
        for _ in range(10):
            initial_sequences.append(self.generate_initial_sequence())

        # Initialize with best among initial sequences
        current_sequence = max(initial_sequences, key=self.compute_inv_c1)
        self.best_sequence = current_sequence.copy()
        self.best_inv_c1 = self.compute_inv_c1(current_sequence)

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break

            # Compute current performance
            inv_c1 = self.compute_inv_c1(current_sequence)
            self.history.append(inv_c1)

            if inv_c1 > self.best_inv_c1:
                self.best_inv_c1 = inv_c1
                self.best_sequence = current_sequence.copy()

            # Preserve elites by adding current sequence to elite list if it's good enough
            # or replacing the worst elite if current is better
            if len(self.elite_sequences) < self.elite_size:
                self.elite_sequences.append((current_sequence.copy(), inv_c1))
            else:
                # Replace worst elite if current is better
                worst_idx = min(range(len(self.elite_sequences)), key=lambda i: self.elite_sequences[i][1])
                if inv_c1 > self.elite_sequences[worst_idx][1]:
                    self.elite_sequences[worst_idx] = (current_sequence.copy(), inv_c1)

            # Attempt to find better direction
            new_sequence = self.get_good_direction_to_move_into(current_sequence, iteration)

            if new_sequence is not None:
                current_sequence = new_sequence
            else:
                # Fallback to elite selection
                if self.elite_sequences:
                    # Pick a random elite sequence
                    selected_elite = random.choice(self.elite_sequences)[0]
                    current_sequence = selected_elite
                else:
                    # Resort to generating a new random sequence
                    current_sequence = self.generate_initial_sequence()

            # Detect stagnation and trigger restart
            if len(self.history) == self.history.maxlen:
                recent_change = abs(self.history[-1] - self.history[0])
                if recent_change < 1e-8:
                    # Restart with elite sequence if available
                    if self.elite_sequences:
                        current_sequence = random.choice(self.elite_sequences)[0]
                    else:
                        current_sequence = self.generate_initial_sequence()

        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AdaptiveConvolutionOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")