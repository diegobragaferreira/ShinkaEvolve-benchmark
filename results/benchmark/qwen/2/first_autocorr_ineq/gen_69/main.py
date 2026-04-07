# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class EnsembleAutocorrelationOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180, ensemble_size=5):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.ensemble_size = ensemble_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.diversity_threshold = 0.1  # Minimum diversity needed to continue exploring
        self.min_sequence_length = 50
        self.max_sequence_length = 2000

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

            # Build convolution constraints
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

    def get_good_direction_to_move_into(self, sequence, iteration=0):
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

            # Adaptive learning rate that decreases exponentially
            t = 0.01 * np.exp(-iteration / 100)
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence

        except Exception as e:
            # Log error or handle gracefully
            return None

    def generate_initial_sequence(self, strategy="random"):
        """Generate an initial sequence with specified strategy."""
        n = random.randint(self.min_sequence_length, self.max_sequence_length)

        if strategy == "symmetric":
            # Create symmetric sequence
            half = n // 2
            sequence = [random.random() for _ in range(half)]
            sequence.extend(sequence[::-1])
            return sequence[:n]
        elif strategy == "sparse":
            # Create sparse sequence with few non-zero elements
            sequence = [0.0] * n
            nonzero_indices = random.sample(range(n), random.randint(10, min(100, n//2)))
            for idx in nonzero_indices:
                sequence[idx] = random.random()
            return sequence
        else:  # random
            return [random.random() for _ in range(n)]

    def calculate_diversity(self, ensemble):
        """Calculate diversity among ensemble members."""
        if len(ensemble) < 2:
            return 0

        # Calculate pairwise differences
        diffs = []
        for i in range(len(ensemble)):
            for j in range(i+1, len(ensemble)):
                diff = np.linalg.norm(np.array(ensemble[i]) - np.array(ensemble[j]))
                diffs.append(diff)

        return np.mean(diffs) if diffs else 0

    def optimize_sequence(self):
        """Main optimization loop with ensemble approach."""
        start_time = time.time()

        # Initialize ensemble with diverse strategies
        ensemble = []
        strategies = ["random", "symmetric", "sparse"]

        for i in range(self.ensemble_size):
            strategy = strategies[i % len(strategies)]
            seq = self.generate_initial_sequence(strategy)
            ensemble.append(seq)

        # Track best across ensemble
        best_in_ensemble = None
        best_inv_c1_in_ensemble = 0

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break

            # Evaluate all ensemble members
            current_best_in_ensemble = None
            current_best_inv_c1 = 0

            for i, sequence in enumerate(ensemble):
                inv_c1 = self.compute_inv_c1(sequence)
                if inv_c1 > current_best_inv_c1:
                    current_best_inv_c1 = inv_c1
                    current_best_in_ensemble = sequence.copy()

                # Update global best
                if inv_c1 > self.best_inv_c1:
                    self.best_inv_c1 = inv_c1
                    self.best_sequence = sequence.copy()

            # Update ensemble best
            if current_best_inv_c1 > best_inv_c1_in_ensemble:
                best_inv_c1_in_ensemble = current_best_inv_c1
                best_in_ensemble = current_best_in_ensemble.copy()

            # Check diversity and potentially regenerate low diversity members
            diversity = self.calculate_diversity(ensemble)
            if diversity < self.diversity_threshold and iteration > 100:
                # Regenerate some members to increase diversity
                for i in range(min(2, len(ensemble))):
                    strategy = random.choice(["random", "symmetric", "sparse"])
                    ensemble[i] = self.generate_initial_sequence(strategy)

            # Move ensemble members
            for i in range(len(ensemble)):
                sequence = ensemble[i]
                new_sequence = self.get_good_direction_to_move_into(sequence, iteration)

                if new_sequence is not None:
                    ensemble[i] = new_sequence
                else:
                    # Fallback to random perturbation if optimization fails
                    ensemble[i] = [
                        max(0, x + random.uniform(-0.05, 0.05))
                        for x in sequence
                    ]

                # Occasionally introduce fresh randomness to prevent stagnation
                if random.random() < 0.05:
                    ensemble[i] = self.generate_initial_sequence()

        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = EnsembleAutocorrelationOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")