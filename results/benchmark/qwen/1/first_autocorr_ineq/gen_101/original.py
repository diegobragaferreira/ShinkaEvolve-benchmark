# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
from scipy import optimize
from joblib import Parallel, delayed
import time
import random
from numba import jit

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class Configuration:
    """Configuration class to manage all parameters centrally."""
    def __init__(self):
        self.max_time_seconds = 180
        self.min_sequence_length = 100
        self.max_sequence_length = 2000
        self.base_sequence_length = 500
        self.stagnation_threshold = 200
        self.learning_rate = 0.05
        self.epsilon = 1e-4
        self.benchmark_threshold = 1.5031
        self.parallel_jobs = -1  # Use all available cores
        self.max_iterations = 5000
        self.lp_max_iterations = 1000
        self.constraint_subset = 100

class SequenceGenerator:
    """Handles generation of various types of sequences."""

    @staticmethod
    def sine_wave_sequence(n):
        """Generate a sine wave based sequence."""
        return [abs(np.sin(np.pi * i / n)) * 100 for i in range(n)]

    @staticmethod
    def structured_sequence(n):
        """Generate a structured sequence with noise."""
        base = SequenceGenerator.sine_wave_sequence(n)
        noise = [np.random.random() * 10 for _ in range(n)]
        return [max(0, b + n) for b, n in zip(base, noise)]

    @staticmethod
    def random_sequence(n):
        """Generate a random sequence."""
        seq = np.random.rand(n)
        # Normalize to ensure sum > 0.01
        seq = seq * (0.01 / (np.sum(seq) + 1e-10))
        return seq.tolist()

    @staticmethod
    def adaptive_sequence_length(config):
        """Adaptively choose sequence length."""
        base = config.base_sequence_length
        return np.random.randint(base // 2, base * 2)

class ConvolutionProcessor:
    """Handles convolution calculations efficiently."""

    @staticmethod
    @jit(nopython=True)
    def fast_convolve(a, b):
        """Fast convolution using Numba JIT."""
        n = len(a)
        result = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if 0 <= i+j < n:
                    result[i+j] += a[i] * b[j]
        return result

    @staticmethod
    def compute_fft_convolution(seq):
        """Compute convolution using FFT for efficiency."""
        n = len(seq)
        padded_seq = np.pad(seq, (0, n-1), mode='constant')
        fft_seq = fft(padded_seq)
        conv_result = ifft(fft_seq * np.conj(fft_seq)).real[:n]
        return conv_result

class SequenceEvaluator:
    """Handles evaluation of sequence quality."""

    @staticmethod
    def compute_autocorrelation_constant(sequence):
        """Compute the C1 constant for a given sequence."""
        if len(sequence) == 0:
            return float('inf')

        sum_a = np.sum(sequence)
        if sum_a < 0.01:
            return float('inf')

        # Use FFT for efficient convolution
        conv_result = ConvolutionProcessor.compute_fft_convolution(sequence)
        max_conv = np.max(conv_result)

        # Compute C1 = 2n * max(conv) / (sum(a))^2
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

    @staticmethod
    def evaluate_sequence(sequence):
        """Evaluate a sequence by computing its inverse C1."""
        c1 = SequenceEvaluator.compute_autocorrelation_constant(sequence)
        if c1 == float('inf'):
            return 0.0  # Invalid sequence
        return 1.0 / c1  # Return 1/C1 as the objective

    @staticmethod
    def evaluate_batch(sequences):
        """Batch evaluation of sequences to leverage parallelism."""
        return Parallel(n_jobs=-1)(delayed(SequenceEvaluator.evaluate_sequence)(seq) for seq in sequences)

class OptimizationEngine:
    """Main optimization engine with adaptive strategies."""

    def __init__(self, config):
        self.config = config

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
        if n == 0:
            return None

        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build constraint matrix efficiently
        if n > 1000:
            # For large sequences, use a subset of key constraints
            constraint_indices = np.linspace(0, 2*n-2, min(self.config.constraint_subset, 2*n-1), dtype=int)
            for k in constraint_indices:
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)
        else:
            # For smaller sequences, include all convolution constraints
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

        # Add bounds to avoid numerical issues
        bounds = [(0, 1000) for _ in range(n)]  # Clips heights to [0, 1000]

        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                                    method='highs', options={'maxiter': self.config.lp_max_iterations})

            if result.success:
                g_sequence = result.x
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence):
        """Returns the direction to move into the sequence."""
        n = len(sequence)
        if n == 0:
            return None

        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        # Normalize the sequence
        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Compute the target value for the LP constraint
        conv_result = ConvolutionProcessor.compute_fft_convolution(normalized_sequence)
        rhs = np.max(conv_result)

        # Solve the LP problem
        g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is None:
            return None

        # Normalize the resulting sequence
        sum_g_fun = np.sum(g_fun)
        if sum_g_fun < 0.01:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

        # Apply gradient descent-like update with momentum
        new_sequence = [
            (1 - self.config.learning_rate) * x + self.config.learning_rate * y
            for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

    def get_gradient_estimate(self, sequence):
        """Estimate gradient using finite differences."""
        n = len(sequence)
        if n == 0:
            return None

        grad = []
        for i in range(n):
            # Create perturbed sequences
            seq_plus = sequence.copy()
            seq_minus = sequence.copy()

            seq_plus[i] += self.config.epsilon
            seq_minus[i] -= self.config.epsilon

            # Evaluate both and estimate derivative
            val_plus = SequenceEvaluator.evaluate_sequence(seq_plus)
            val_minus = SequenceEvaluator.evaluate_sequence(seq_minus)

            grad_i = (val_plus - val_minus) / (2 * self.config.epsilon)
            grad.append(grad_i)

        return np.array(grad)

    def search_for_best_sequence(self):
        """Main optimization function."""
        start_time = time.time()

        # Initialize with a structured sequence
        n = SequenceGenerator.adaptive_sequence_length(self.config)
        best_sequence = SequenceGenerator.structured_sequence(n)

        best_score = SequenceEvaluator.evaluate_sequence(best_sequence)
        best_c1 = SequenceEvaluator.compute_autocorrelation_constant(best_sequence)

        iterations = 0
        stagnation_count = 0
        best_overall_score = best_score
        best_overall_sequence = best_sequence.copy()

        while iterations < self.config.max_iterations and time.time() - start_time < self.config.max_time_seconds:
            # Try to improve the current sequence
            improved_sequence = self.get_good_direction_to_move_into(best_sequence)

            if improved_sequence is not None:
                # Check if this improvement is beneficial
                new_score = SequenceEvaluator.evaluate_sequence(improved_sequence)
                new_c1 = SequenceEvaluator.compute_autocorrelation_constant(improved_sequence)

                if new_score > best_score:
                    best_sequence = improved_sequence
                    best_score = new_score
                    best_c1 = new_c1
                    stagnation_count = 0  # Reset stagnation counter

                    # Print progress every 100 iterations
                    if iterations % 100 == 0:
                        print(f"Iteration {iterations}: Score = {best_score:.6f}, C1 = {best_c1:.6f}")

                    # Check if we beat the benchmark
                    benchmark_ratio = self.config.benchmark_threshold / best_c1
                    if benchmark_ratio > 1.0:
                        print(f"BEAT BENCHMARK at iteration {iterations}! Ratio = {benchmark_ratio:.6f}")
                        break

                    # Update overall best
                    if new_score > best_overall_score:
                        best_overall_score = new_score
                        best_overall_sequence = improved_sequence.copy()
                else:
                    stagnation_count += 1
                    if stagnation_count > self.config.stagnation_threshold:
                        # Restart with a new structured sequence
                        n = SequenceGenerator.adaptive_sequence_length(self.config)
                        best_sequence = SequenceGenerator.structured_sequence(n)
                        best_score = SequenceEvaluator.evaluate_sequence(best_sequence)
                        best_c1 = SequenceEvaluator.compute_autocorrelation_constant(best_sequence)
                        stagnation_count = 0  # Reset stagnation count
            else:
                # If we couldn't improve, try random restart
                n = SequenceGenerator.adaptive_sequence_length(self.config)
                best_sequence = SequenceGenerator.structured_sequence(n)

            iterations += 1

        # Final validation
        final_score = SequenceEvaluator.evaluate_sequence(best_sequence)
        final_c1 = SequenceEvaluator.compute_autocorrelation_constant(best_sequence)
        benchmark_ratio = self.config.benchmark_threshold / final_c1

        print(f"Final result:")
        print(f"  Score: {final_score:.6f}")
        print(f"  C1: {final_c1:.6f}")
        print(f"  Benchmark ratio: {benchmark_ratio:.6f}")
        print(f"  Iterations: {iterations}")

        # Return the best sequence found
        return best_overall_sequence

def search_for_best_sequence():
    """Main entry point function."""
    config = Configuration()
    optimizer = OptimizationEngine(config)
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")