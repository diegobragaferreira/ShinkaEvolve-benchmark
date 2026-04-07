# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial
import copy
from typing import List, Tuple, Optional
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class HybridAutocorrelationOptimizer:
    """
    A hybrid optimizer combining evolutionary and gradient-based methods to maximize 1/C₁.
    """

    def __init__(self, pop_size: int = 50, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.best_score = 0.0
        self.best_sequence = None
        self.fitness_cache = {}  # Memoization for fitness evaluations

    @staticmethod
    def evaluate_sequence(sequence: List[float]) -> Tuple[float, float, float, float]:
        """
        Evaluate a sequence and return its performance metrics.

        Args:
            sequence: List of non-negative real numbers representing step heights

        Returns:
            tuple: (C₁, 1/C₁, max_convolution_value, sum_of_sequence)
        """
        # Check cache first
        seq_tuple = tuple(sequence)
        if seq_tuple in HybridAutocorrelationOptimizer.fitness_cache:
            return HybridAutocorrelationOptimizer.fitness_cache[seq_tuple]

        try:
            # Convert to numpy array
            a = np.array(sequence)
            sum_a = np.sum(a)

            # Avoid division by zero or negligible sums
            if sum_a < 1e-10:
                result = (float('inf'), 0.0, 0.0, sum_a)
                HybridAutocorrelationOptimizer.fitness_cache[seq_tuple] = result
                return result

            # Compute autoconvolution using FFT for efficiency
            b = fftconvolve(a, a, mode='full')
            b = b[len(a)-1:2*len(a)-1]  # Convolution part

            max_b = np.max(b)

            # Compute C₁ = 2n * max(b) / (sum(a))^2
            n = len(a)
            c1 = 2 * n * max_b / (sum_a ** 2)

            # Return inverse for maximization
            inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

            result = (c1, inv_c1, max_b, sum_a)
            HybridAutocorrelationOptimizer.fitness_cache[seq_tuple] = result
            return result
        except Exception as e:
            result = (float('inf'), 0.0, 0.0, 0.0)
            HybridAutocorrelationOptimizer.fitness_cache[seq_tuple] = result
            return result

    @staticmethod
    def generate_structured_sequence(length: int) -> List[float]:
        """Generate a structured sequence using a sinusoidal pattern for better initialization."""
        # Use a sine wave pattern for better initial structure
        sequence = [abs(np.sin(i * np.pi / length)) * 1000 for i in range(length)]
        # Add some randomness to avoid perfect symmetry
        for i in range(length):
            if random.random() < 0.1:
                sequence[i] += random.uniform(-100, 100)

        # Ensure non-negative
        sequence = [max(0, x) for x in sequence]

        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01

        return sequence

    @staticmethod
    def gradient_refine(sequence: List[float], steps: int = 50, lr: float = 1e-4) -> List[float]:
        """Apply gradient-based refinement to improve the sequence."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)
        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq

        prev_max_conv = float('inf')

        for step in range(steps):
            # Compute convolution
            conv = fftconvolve(seq, seq, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = np.max(conv)

            # Early stopping if improvement is minimal
            if abs(prev_max_conv - max_conv) < 1e-10:
                break
            prev_max_conv = max_conv

            # Compute gradient estimate using finite differences
            grad = np.zeros_like(seq)
            epsilon = 1e-6
            for i in range(n):
                # Perturb forward
                seq_forward = seq.copy()
                seq_forward[i] += epsilon
                seq_forward /= np.sum(seq_forward)

                # Perturb backward
                seq_backward = seq.copy()
                seq_backward[i] -= epsilon
                seq_backward /= np.sum(seq_backward)

                # Compute convolution for both perturbed sequences
                conv_forward = fftconvolve(seq_forward, seq_forward, mode='full')
                conv_forward = conv_forward[n-1:2*n-1]
                max_forward = np.max(conv_forward)

                conv_backward = fftconvolve(seq_backward, seq_backward, mode='full')
                conv_backward = conv_backward[n-1:2*n-1]
                max_backward = np.max(conv_backward)

                # Central difference gradient estimate
                grad[i] = (max_forward - max_backward) / (2 * epsilon)

            # Adaptive learning rate that decreases over iterations
            adaptive_lr = lr * (1.0 - 0.9 * (step / steps))

            # Update using gradient ascent
            seq += adaptive_lr * grad
            seq = np.maximum(seq, 0)  # Ensure non-negative

            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq

        return seq.tolist()

    def optimize(self) -> List[float]:
        """Run the hybrid optimization process."""
        start_time = time.time()
        timeout = 170  # Leave 10 seconds for cleanup

        # Generate initial structured sequence
        length = random.randint(100, 500)
        sequence = self.generate_structured_sequence(length)

        # Refine the initial sequence
        refined_sequence = self.gradient_refine(sequence)
        _, best_score, _, _ = self.evaluate_sequence(refined_sequence)

        self.best_score = best_score
        self.best_sequence = copy.deepcopy(refined_sequence)

        print(f"Initial refined score: {self.best_score:.6f}")

        # Further improvement with iterative refinement
        for iteration in range(5):  # Limit iterations to avoid long runtimes
            if time.time() - start_time > timeout:
                break

            # Try generating new sequences to find better ones
            candidate_sequences = [self.generate_structured_sequence(random.randint(100, 500))
                                 for _ in range(10)]

            # Evaluate candidates
            candidate_scores = [self.evaluate_sequence(seq)[1] for seq in candidate_sequences]

            # Select best candidate
            best_candidate_idx = np.argmax(candidate_scores)
            if candidate_scores[best_candidate_idx] > self.best_score:
                self.best_score = candidate_scores[best_candidate_idx]
                self.best_sequence = copy.deepcopy(candidate_sequences[best_candidate_idx])
                print(f"New best score: {self.best_score:.6f}")

            # Refine current best
            refined_current = self.gradient_refine(self.best_sequence)
            _, refined_score, _, _ = self.evaluate_sequence(refined_current)

            if refined_score > self.best_score:
                self.best_score = refined_score
                self.best_sequence = copy.deepcopy(refined_current)
                print(f"Refined best score: {self.best_score:.6f}")

        return self.best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = HybridAutocorrelationOptimizer(pop_size=50, generations=50)
    try:
        best_sequence = optimizer.optimize()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to a basic sequence if nothing worked
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")