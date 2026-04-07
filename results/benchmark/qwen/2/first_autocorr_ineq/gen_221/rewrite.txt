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

class AutocorrelationGradientOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180, elite_size=50):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.elite_size = elite_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self.history = deque(maxlen=20)

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

    def compute_gradient_via_finite_diff(self, sequence, epsilon=1e-5):
        """Estimate gradient using finite differences."""
        n = len(sequence)
        grad = np.zeros(n)
        base_c1 = self.compute_c1(sequence)
        
        # Compute gradient for each element
        for i in range(n):
            perturbed_seq = sequence.copy()
            perturbed_seq[i] += epsilon
            perturbed_c1 = self.compute_c1(perturbed_seq)
            grad[i] = (base_c1 - perturbed_c1) / epsilon  # Negative since we want to minimize
            
        return grad

    def get_good_direction_to_move_into(self, sequence, iteration):
        """Returns the direction to move into the sequence using gradient descent."""
        try:
            n = len(sequence)
            sum_sequence = np.sum(sequence)
            if sum_sequence < 1e-10:
                return None

            # Estimate gradient via finite differences
            grad = self.compute_gradient_via_finite_diff(sequence)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return None
                
            normalized_grad = grad / grad_norm

            # Adaptive learning rate
            t = 0.05 * np.exp(-iteration / 50)
            
            # Update sequence
            new_sequence = [
                max(0, (1 - t) * x + t * y) for x, y in zip(sequence, normalized_grad)
            ]

            return new_sequence

        except Exception:
            return None

    def generate_initial_sequence(self):
        """Generate an initial random sequence."""
        n = random.randint(100, 1000)
        return [random.random() for _ in range(n)]

    def optimize_sequence(self):
        """Main optimization loop."""
        start_time = time.time()

        # Generate initial sequence
        current_sequence = self.generate_initial_sequence()

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break

            # Compute current performance
            inv_c1 = self.compute_inv_c1(current_sequence)
            if inv_c1 > self.best_inv_c1:
                self.best_inv_c1 = inv_c1
                self.best_sequence = current_sequence.copy()

            # Preserve elites
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
                # Three-tiered fallback strategy
                # Tier 1: Try mirrored sequence
                mirrored_seq = current_sequence[::-1]
                mirrored_inv_c1 = self.compute_inv_c1(mirrored_seq)
                if mirrored_inv_c1 > inv_c1:
                    current_sequence = mirrored_seq
                else:
                    # Tier 2: Apply bounded random perturbations
                    perturbed_seq = [
                        max(0, x + random.uniform(-0.05, 0.05))
                        for x in current_sequence
                    ]
                    perturbed_inv_c1 = self.compute_inv_c1(perturbed_seq)
                    if perturbed_inv_c1 > inv_c1:
                        current_sequence = perturbed_seq
                    else:
                        # Tier 3: Resort to randomization
                        current_sequence = [
                            random.random()
                            for _ in range(len(current_sequence))
                        ]

        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationGradientOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")