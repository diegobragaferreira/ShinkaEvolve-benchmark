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

class AdaptiveFFTEvolutionaryOptimizer:
    """
    An advanced optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    This version focuses on leveraging FFT performance, adaptive search strategies,
    and intelligent initialization to surpass the benchmark.
    """

    def __init__(self, max_sequence_length: int = 2000, min_sequence_length: int = 100):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.refinement_steps = 20  # More refinement steps for better convergence
        self.seed = 42  # For deterministic results
        np.random.seed(self.seed)
        self.best_score = 0.0
        self.stagnation_count = 0
        self.max_stagnation = 50

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
        if n > 1000:  # For large sequences, use FFT
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

    def gradient_refinement(self, sequence: np.ndarray, steps: int = 20, lr_range=(0.005, 0.02)) -> np.ndarray:
        """Perform gradient-based refinement on the sequence with adaptive learning rates."""
        current = np.array(sequence, dtype=float)
        velocity = np.zeros_like(current)  # Momentum term
        beta = 0.9  # Momentum parameter

        # Adaptive learning rate scheduling
        lr_schedule = np.linspace(lr_range[0], lr_range[1], steps)

        for step in range(steps):
            grad = self.compute_gradient(current)
            
            # Adaptive learning rate
            lr = lr_schedule[step]
            velocity = beta * velocity + (1 - beta) * grad
            
            # Adjust step size based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                step_size = lr / (grad_norm + 1e-8)  # Add small value to prevent division by zero
                current += step_size * velocity
                current = np.maximum(current, 0)  # Keep non-negative

        return current.tolist()

    def generate_initial_sequence(self, n: int = None) -> List[float]:
        """Generate a good initial sequence with multi-modal patterns."""
        if n is None:
            n = np.random.randint(self.min_sequence_length, self.max_sequence_length)

        # Use a combination of exponential decay, sinusoidal components, and step functions
        sequence = []
        for i in range(n):
            # Combine exponential decay with sinusoidal modulation and step variations
            exp_component = np.exp(-i / (n / 3))
            sin_component = abs(np.sin(np.pi * i / n))
            # Add step-like characteristic
            step_component = 1 if i % (n//10) < n//20 else 0.5
            # Mix them together with some randomness
            component = 0.5 * exp_component + 0.3 * sin_component + 0.2 * step_component + 0.05 * np.random.random()
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
        
        # Use a higher blending factor for more decisive moves
        t = 0.1  
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Apply additional smoothing to the sequence to reduce noise
        # Use a moving average with a window of 3
        if len(new_sequence) > 2:
            smoothed_sequence = []
            for i in range(len(new_sequence)):
                if i == 0:
                    window_sum = new_sequence[i] + new_sequence[i+1]
                    window_size = 2
                elif i == len(new_sequence) - 1:
                    window_sum = new_sequence[i-1] + new_sequence[i]
                    window_size = 2
                else:
                    window_sum = new_sequence[i-1] + new_sequence[i] + new_sequence[i+1]
                    window_size = 3
                smoothed_sequence.append(window_sum / window_size)
            return smoothed_sequence
        else:
            return new_sequence

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Precompute the convolution constraint matrix using explicit loop
        # Limit constraints for large sequences to improve performance
        if n > 1000:
            constraint_indices = np.linspace(0, 2*n-2, min(200, 2*n-1), dtype=int)
            for k in constraint_indices:
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)
        else:
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
            # Use high-performance linear programming solver
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 1000})
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
        """Function to search for the best coefficient sequence with adaptive strategies."""
        start_time = time.time()
        max_time = 175  # Leave some time for cleanup
        
        # Dynamic population size based on time left
        population_size = 30
        generation = 0
        
        # Initialize with a structured sequence
        n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
        best_sequence = self.generate_initial_sequence(n)
        
        # Evaluate initial best sequence
        self.best_score = self.compute_inv_c1(best_sequence)
        
        while time.time() - start_time < max_time and generation < 1000:
            # Perform evolutionary refinement
            improved_sequence = self.get_good_direction_to_move_into(best_sequence)
            if improved_sequence is not None:
                improved_score = self.compute_inv_c1(improved_sequence)
                if improved_score > self.best_score:
                    best_sequence = improved_sequence
                    self.best_score = improved_score
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1
            else:
                self.stagnation_count += 1
            
            # Restart if stagnating
            if self.stagnation_count > self.max_stagnation:
                n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
                best_sequence = self.generate_initial_sequence(n)
                self.stagnation_count = 0
                
            # Refine using gradient descent
            refined_sequence = self.gradient_refinement(best_sequence, self.refinement_steps)
            refined_score = self.compute_inv_c1(refined_sequence)
            
            if refined_score > self.best_score:
                best_sequence = refined_sequence
                self.best_score = refined_score
                
            generation += 1

        return best_sequence

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AdaptiveFFTEvolutionaryOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")