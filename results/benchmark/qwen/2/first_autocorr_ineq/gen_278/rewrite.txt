# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random

class AutocorrelationOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180, elite_size=50):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.elite_size = elite_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []

    def convolve_fft(self, a, b):
        """Compute convolution using FFT for better performance."""
        n = len(a)
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
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 1000})

            if result.success:
                return result.x
            else:
                return None

        except Exception:
            return None

    def estimate_gradient(self, sequence, epsilon=1e-4):
        """Estimate gradient using finite differences with adaptive epsilon."""
        n = len(sequence)
        grad = []
        for i in range(n):
            elem_mag = abs(sequence[i])
            if elem_mag < 1e-6:
                eps = epsilon
            else:
                eps = epsilon * elem_mag

            # Perturb dimension i
            perturbed_plus = sequence[:]
            perturbed_minus = sequence[:]
            perturbed_plus[i] += eps
            perturbed_minus[i] -= eps

            # Ensure non-negativity
            perturbed_plus[i] = max(0, perturbed_plus[i])
            perturbed_minus[i] = max(0, perturbed_minus[i])

            # Evaluate function
            f_plus = self.compute_inv_c1(perturbed_plus)
            f_minus = self.compute_inv_c1(perturbed_minus)

            grad_i = (f_plus - f_minus) / (2 * eps)
            grad.append(grad_i)
        return grad

    def adaptive_step_size(self, grad_norm, iteration):
        """Adaptive step size that decreases with iteration and adjusts by gradient magnitude."""
        base_step = 0.01
        decay_rate = 0.95
        base_step *= (decay_rate ** iteration)
        if grad_norm > 1e-3:
            base_step *= min(1.0, 1.0 / grad_norm)
        return max(base_step, 1e-6)

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

            # Adaptive learning rate
            t = self.adaptive_step_size(np.linalg.norm(normalized_g_fun), iteration)
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence

        except Exception:
            return None

    def ensemble_gradient_ascent(self, sequence, num_directions=5):
        """Perform ensemble gradient ascent to find better directions."""
        best_sequence = sequence[:]
        best_score = self.compute_inv_c1(sequence)

        for i in range(num_directions):
            current = sequence[:]
            grad = self.estimate_gradient(current)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < 1e-10:
                continue

            normalized_grad = [g / grad_norm for g in grad]
            step_size = self.adaptive_step_size(grad_norm, 0)
            new_sequence = [max(0, x + step_size * g) for x, g in zip(current, normalized_grad)]

            new_score = self.compute_inv_c1(new_sequence)
            if new_score > best_score:
                best_score = new_score
                best_sequence = new_sequence[:]

        return best_sequence

    def generate_initial_sequence(self):
        """Generate an initial random sequence."""
        n = random.randint(100, 1000)
        return [random.random() for _ in range(n)]

    def optimize_sequence(self):
        """Main optimization loop."""
        start_time = time.time()
        current_sequence = self.generate_initial_sequence()

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break

            inv_c1 = self.compute_inv_c1(current_sequence)
            if inv_c1 > self.best_inv_c1:
                self.best_inv_c1 = inv_c1
                self.best_sequence = current_sequence.copy()

            # Elite preservation
            if len(self.elite_sequences) < self.elite_size:
                self.elite_sequences.append((current_sequence.copy(), inv_c1))
            else:
                worst_idx = min(range(len(self.elite_sequences)), key=lambda i: self.elite_sequences[i][1])
                if inv_c1 > self.elite_sequences[worst_idx][1]:
                    self.elite_sequences[worst_idx] = (current_sequence.copy(), inv_c1)

            # Ensemble approach
            ensemble_sequence = self.ensemble_gradient_ascent(current_sequence)
            ensemble_score = self.compute_inv_c1(ensemble_sequence)

            if ensemble_score > inv_c1:
                current_sequence = ensemble_sequence
            else:
                new_sequence = self.get_good_direction_to_move_into(current_sequence, iteration)
                if new_sequence is not None:
                    current_sequence = new_sequence
                else:
                    # Fallback strategies
                    mirrored_seq = current_sequence[::-1]
                    mirrored_inv_c1 = self.compute_inv_c1(mirrored_seq)
                    if mirrored_inv_c1 > inv_c1:
                        current_sequence = mirrored_seq
                    else:
                        perturbed_seq = [max(0, x + random.uniform(-0.05, 0.05)) for x in current_sequence]
                        perturbed_inv_c1 = self.compute_inv_c1(perturbed_seq)
                        if perturbed_inv_c1 > inv_c1:
                            current_sequence = perturbed_seq
                        else:
                            current_sequence = [random.random() for _ in range(len(current_sequence))]

        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")