# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    def __init__(self, max_iterations=1000, timeout_seconds=180, ensemble_size=5, elite_size=50):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self._best_sequence_history = []

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

    def solve_convolution_lp(self, f_sequence, rhs, warm_start=None):
        """Solves the convolution LP for a given sequence and RHS."""
        try:
            n = len(f_sequence)
            c = -np.ones(n)
            a_ub = []
            b_ub = []

            # Build convolution constraints using FFT for efficiency
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
            options = {'maxiter': 1000}
            if warm_start is not None:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', 
                                        x0=warm_start, options=options)
            else:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options=options)

            if result.success:
                return result.x
            else:
                return None

        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence, iteration, warm_start=None):
        """Returns the direction to move into the sequence with adaptive learning rate."""
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

            # Solve LP with warm start
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs, warm_start)
            if g_fun is None:
                return None

            # Normalize g_fun
            sum_g_fun = np.sum(g_fun)
            if sum_g_fun < 1e-10:
                return None

            normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

            # Adaptive learning rate that decreases exponentially
            t = 0.05 * np.exp(-iteration / 50)
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence

        except Exception:
            return None

    def generate_initial_sequence(self, length=None):
        """Generate an initial random sequence."""
        if length is None:
            n = random.randint(100, 1000)
        else:
            n = length
        return [random.random() for _ in range(n)]

    def get_ensemble_direction(self, sequence, iteration):
        """Combine multiple strategies to get a better optimization direction."""
        strategies = []

        # Strategy 1: Gradient-based update with warm start
        warm_start = None
        if len(self.elite_sequences) > 0:
            # Use best elite sequence as warm start
            best_elite = max(self.elite_sequences, key=lambda x: x[1])
            warm_start = best_elite[0]
        
        grad_direction = self.get_good_direction_to_move_into(sequence, iteration, warm_start)
        if grad_direction is not None:
            strategies.append(('gradient', grad_direction))

        # Strategy 2: Random perturbation (as fallback)
        random_perturbation = [max(0, x + random.uniform(-0.1, 0.1)) for x in sequence]
        strategies.append(('random', random_perturbation))

        # Strategy 3: Evolutionary mutation
        if len(sequence) > 10:
            mutated = sequence.copy()
            for i in range(len(mutated)):
                if random.random() < 0.1:
                    mutated[i] = max(0, mutated[i] * random.uniform(0.9, 1.1))
            strategies.append(('mutation', mutated))

        # Strategy 4: Mirror
        mirror_seq = sequence[::-1]
        strategies.append(('mirror', mirror_seq))

        # Strategy 5: Sequence length adjustment (explore different n)
        adjusted_seq = self.generate_initial_sequence(max(50, min(2000, len(sequence) * random.uniform(0.8, 1.2))))
        strategies.append(('adjust_length', adjusted_seq))

        # Weighted selection of best strategy
        best_strategy = None
        best_fitness = float('-inf')

        for strategy_name, strat_seq in strategies:
            fitness = self.compute_inv_c1(strat_seq)
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = strat_seq

        return best_strategy

    def optimize_sequence(self):
        """Main optimization loop."""
        start_time = time.time()

        # Ensemble of optimizations to increase chances of finding better solution
        best_in_ensemble = None
        best_inv_c1_in_ensemble = 0.0

        for ensemble_run in range(self.ensemble_size):
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

                if inv_c1 > best_inv_c1_in_ensemble:
                    best_inv_c1_in_ensemble = inv_c1
                    best_in_ensemble = current_sequence.copy()

                # Preserve elites by adding current sequence to elite list if it's good enough
                # or replacing the worst elite if current is better
                if len(self.elite_sequences) < self.elite_size:
                    self.elite_sequences.append((current_sequence.copy(), inv_c1))
                else:
                    # Replace worst elite if current is better
                    worst_idx = min(range(len(self.elite_sequences)), key=lambda i: self.elite_sequences[i][1])
                    if inv_c1 > self.elite_sequences[worst_idx][1]:
                        self.elite_sequences[worst_idx] = (current_sequence.copy(), inv_c1)

                # Attempt to find better direction using ensemble method
                new_sequence = self.get_ensemble_direction(current_sequence, iteration)

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

            # Update best overall if this ensemble run was better
            if best_inv_c1_in_ensemble > self.best_inv_c1:
                self.best_inv_c1 = best_inv_c1_in_ensemble
                self.best_sequence = best_in_ensemble.copy()

        return self.best_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")