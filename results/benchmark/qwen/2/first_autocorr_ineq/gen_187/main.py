# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time
import math

class AutocorrelationOptimizer:
    def __init__(self, max_time_seconds=170, max_iterations=1000):
        self.max_time_seconds = max_time_seconds
        self.max_iterations = max_iterations
        self.elite_sequences = []
        self.seed = 42  # For deterministic behavior
        random.seed(self.seed)
        np.random.seed(self.seed)

    def compute_convolution_fft(self, seq):
        """Compute the autoconvolution using FFT for efficiency."""
        n = len(seq)
        padded_seq = np.pad(seq, (0, n-1), mode='constant')
        conv_result = ifft(fft(padded_seq) * np.conj(fft(padded_seq)))
        return np.real(conv_result[:n])

    def calculate_c1(self, sequence):
        """Calculate the C1 constant for a given sequence."""
        if len(sequence) == 0:
            return float('inf')

        sequence = np.array(sequence)
        sum_a = np.sum(sequence)
        if sum_a < 0.01:
            return float('inf')

        conv = self.compute_convolution_fft(sequence)
        max_b = np.max(conv)
        n = len(sequence)

        if max_b <= 1e-12:
            return float('inf')

        c1 = (2 * n * max_b) / (sum_a ** 2)
        return c1

    def evaluate_fitness(self, sequence):
        """Evaluate the inverse of C1 as fitness (we want to maximize 1/C1)."""
        c1 = self.calculate_c1(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Non-negativity constraints: b_i >= 0
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            if result.success:
                g_sequence = result.x
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence, iteration=0):
        """Returns the direction to move into the sequence using gradient approach."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
        conv = self.compute_convolution_fft(normalized_sequence)
        rhs = np.max(conv)

        g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is None:
            # Fallback to simple gradient descent on a smaller scale
            t = 0.001
            new_sequence = [(1 - t) * x + t * (x + np.random.normal(0, 0.01)) for x in sequence]
            return new_sequence

        sum_g = np.sum(g_fun)
        if sum_g < 0.01:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

        # Adaptive step size based on iteration
        t = max(0.001, 0.1 * (1 - iteration / self.max_iterations))

        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity
        new_sequence = [max(0.0, x) for x in new_sequence]
        return new_sequence

    def generate_initial_sequence(self):
        """Generate a better initial sequence to start optimization."""
        n = random.randint(100, 1000)
        sequence = []
        for i in range(n):
            if i % 5 == 0:
                sequence.append(random.uniform(100, 1000))
            else:
                sequence.append(random.uniform(0, 100))
        return sequence

    def adaptive_evolution_step(self, current_sequence, population_size=20, elite_fraction=0.2):
        """Perform adaptive evolution step with elite preservation."""
        elite_size = max(1, int(elite_fraction * population_size))
        offspring = []
        for i in range(population_size):
            if i < elite_size:
                offspring.append(current_sequence.copy())
            else:
                mutated = current_sequence.copy()
                for j in range(len(mutated)):
                    if np.random.random() < 0.1:
                        delta = np.random.normal(0, 0.1 * np.mean(mutated) if np.mean(mutated) > 0 else 0.1)
                        mutated[j] = max(0.0, mutated[j] + delta)
                offspring.append(mutated)
        return offspring

    def run_optimization(self):
        """Main optimization loop."""
        start_time = time.time()
        best_sequence = self.generate_initial_sequence()
        best_fitness = self.evaluate_fitness(best_sequence)
        iteration = 0

        while time.time() - start_time < self.max_time_seconds and iteration < self.max_iterations:
            iteration += 1
            
            # Alternate between methods
            if iteration % 3 == 0:
                candidates = self.adaptive_evolution_step(best_sequence)
                fitness_scores = [self.evaluate_fitness(seq) for seq in candidates]
                best_candidate_idx = np.argmax(fitness_scores)
                candidate_fitness = fitness_scores[best_candidate_idx]
                if candidate_fitness > best_fitness:
                    best_sequence = candidates[best_candidate_idx].copy()
                    best_fitness = candidate_fitness
                    self.elite_sequences.append(best_sequence)
                    if len(self.elite_sequences) > 10:
                        elite_fitnesses = [self.evaluate_fitness(s) for s in self.elite_sequences]
                        top_indices = np.argsort(elite_fitnesses)[-5:]
                        self.elite_sequences = [self.elite_sequences[i] for i in top_indices]
            else:
                h_function = self.get_good_direction_to_move_into(best_sequence, iteration)
                if h_function is not None:
                    candidate_sequence = h_function
                    candidate_fitness = self.evaluate_fitness(candidate_sequence)
                    if candidate_fitness > best_fitness:
                        best_sequence = candidate_sequence
                        best_fitness = candidate_fitness
                        self.elite_sequences.append(candidate_sequence)
                        if len(self.elite_sequences) > 10:
                            elite_fitnesses = [self.evaluate_fitness(s) for s in self.elite_sequences]
                            top_indices = np.argsort(elite_fitnesses)[-5:]
                            self.elite_sequences = [self.elite_sequences[i] for i in top_indices]
                else:
                    new_sequence = []
                    for x in best_sequence:
                        new_sequence.append(max(0.0, x + random.uniform(-1, 1)))
                    if self.evaluate_fitness(new_sequence) > best_fitness:
                        best_sequence = new_sequence

        return best_sequence

def search_for_best_sequence(max_time_seconds=170) -> list[float]:
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer(max_time_seconds=max_time_seconds)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")