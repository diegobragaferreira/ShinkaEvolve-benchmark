# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional, Tuple
import numba
from numba import jit
import time
import random

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    """
    An optimized optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    This version combines evolutionary search, gradient-based refinement, and FFT-based convolution.
    """

    def __init__(self, max_sequence_length: int = 1000, min_sequence_length: int = 100):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.refinement_steps = 10  # Number of refinement steps after evolution
        self.seed = 42  # For deterministic results
        np.random.seed(self.seed)

    @staticmethod
    @jit(nopython=True)
    def fast_convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Fast convolution using Numba JIT compilation."""
        n1, n2 = len(arr1), len(arr2)
        result = np.zeros(n1 + n2 - 1)
        for i in range(n1):
            for j in range(n2):
                result[i + j] += arr1[i] * arr2[j]
        return result

    def compute_convolution(self, sequence: np.ndarray) -> np.ndarray:
        """Compute the autoconvolution of the sequence efficiently."""
        n = len(sequence)
        if n > 500:  # For large sequences, use FFT
            try:
                # Ensure sequence is properly shaped for FFT
                padded_seq = np.pad(sequence, (0, n - 1), 'constant')
                # Use proper conjugate multiplication for convolution
                fft_seq = fft(padded_seq, 2*n-1)
                conv_result = np.real(ifft(fft_seq * np.conj(fft_seq)))
                return conv_result[:2*n - 1]
            except Exception as e:
                print(f"FFT convolution error: {e}")
                # Fallback to direct computation if FFT fails
                return self.fast_convolve(sequence, sequence)
        else:
            # For smaller sequences, use direct computation for precision
            return self.fast_convolve(sequence, sequence)

    def compute_c1(self, sequence: np.ndarray) -> float:
        """Compute C₁ constant for the given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')

        # Compute convolution
        conv = self.compute_convolution(sequence)

        # Calculate C₁
        max_conv = np.max(conv)
        sum_sq = np.sum(sequence) ** 2
        if sum_sq == 0:
            return float('inf')

        c1 = 2 * n * max_conv / sum_sq
        return c1

    def compute_inv_c1(self, sequence: np.ndarray) -> float:
        """Compute 1/C₁ for the given sequence."""
        c1 = self.compute_c1(sequence)
        if c1 == 0:
            return float('inf')
        return 1.0 / c1

    def compute_gradient(self, sequence: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute approximate gradient of 1/C₁ using finite differences."""
        n = len(sequence)
        grad = np.zeros(n)
        base_inv_c1 = self.compute_inv_c1(sequence)

        # Add small noise to avoid numerical issues
        epsilon = max(1e-8, epsilon)

        for i in range(n):
            perturbed = sequence.copy()
            perturbed[i] += epsilon
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negative
            perturbed_inv_c1 = self.compute_inv_c1(perturbed)
            grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon

        return grad

    def gradient_refinement(self, sequence: np.ndarray, steps: int = 10, lr: float = 0.01) -> np.ndarray:
        """Perform gradient-based refinement on the sequence with momentum."""
        current = np.array(sequence, dtype=float)
        velocity = np.zeros_like(current)  # Momentum term
        beta = 0.9  # Momentum parameter

        for step in range(steps):
            grad = self.compute_gradient(current)

            # Update velocity with momentum
            velocity = beta * velocity + (1 - beta) * grad

            # Adjust step size based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                step_size = lr / (grad_norm + 1e-8)  # Add small value to prevent division by zero
                current += step_size * velocity
                current = np.maximum(current, 0)  # Keep non-negative

        return current.tolist()

    def generate_initial_sequence(self, n: int = None) -> List[float]:
        """Generate a good initial sequence with structured patterns."""
        if n is None:
            n = np.random.randint(self.min_sequence_length, self.max_sequence_length)

        # Use a combination of exponential decay and sinusoidal components
        sequence = []
        for i in range(n):
            # Combine exponential decay with sinusoidal modulation
            exp_component = np.exp(-i / (n / 3))
            sin_component = abs(np.sin(np.pi * i / n))
            # Mix them together with some randomness
            component = 0.7 * exp_component + 0.3 * sin_component + 0.1 * np.random.random()
            sequence.append(component * 100)

        # Normalize to have reasonable total weight
        total = sum(sequence)
        if total > 0:
            sequence = [x / total * 100 for x in sequence]

        # Ensure no element is negative
        sequence = [max(0, x) for x in sequence]

        return sequence

    def get_good_direction_to_move_into(
        self, sequence: list[float]
    ) -> list[float] | None:
        """Returns the direction to move into the sequence with enhanced strategy."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)

        # Avoid division by zero
        if sum_sequence < 1e-10:
            return None

        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Use FFT for faster convolution
        try:
            conv_result = np.real(ifft(fft(normalized_sequence, 2*n-1) *
                                       np.conj(fft(normalized_sequence, 2*n-1))))
            rhs = np.max(conv_result[:2*n-1])  # Only consider the actual convolution results
        except Exception as e:
            print(f"Error during FFT convolution: {e}")
            return None

        g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is None:
            return None

        sum_g_fun = np.sum(g_fun)
        if sum_g_fun < 1e-10:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
        t = 0.05  # Increased blending factor for more substantial updates
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Apply additional smoothing to the sequence to reduce noise
        smoothed_sequence = []
        for i in range(len(new_sequence)):
            # Average with neighbors for smoothing
            neighbors = []
            if i > 0:
                neighbors.append(new_sequence[i-1])
            neighbors.append(new_sequence[i])
            if i < len(new_sequence) - 1:
                neighbors.append(new_sequence[i+1])

            smoothed_sequence.append(np.mean(neighbors))

        return smoothed_sequence

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Precompute the convolution constraint matrix using explicit loop
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

        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            if result.success:
                g_sequence = result.x
                return g_sequence
            else:
                print('LP optimization failed:', result.message)
                return None
        except Exception as e:
            print(f'LP optimization error: {e}')
            return None

    def search_for_best_sequence(self) -> list[float]:
        """Function to search for the best coefficient sequence."""
        # Initialize with a random sequence of appropriate size
        n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
        best_sequence = self.generate_initial_sequence(n)

        # Attempt to improve the sequence
        h_function = self.get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # If improvement fails, just perturb the initial sequence slightly
            if len(best_sequence) > 1:
                best_sequence[1] = (best_sequence[1] + np.random.rand()) % 1

        # Perform gradient refinement
        refined_sequence = self.gradient_refinement(best_sequence, self.refinement_steps, 0.01)
        refined_inv_c1 = self.compute_inv_c1(refined_sequence)
        initial_inv_c1 = self.compute_inv_c1(best_sequence)

        if refined_inv_c1 > initial_inv_c1:
            best_sequence = refined_sequence

        return best_sequence

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")