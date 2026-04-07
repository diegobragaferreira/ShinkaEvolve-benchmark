# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class EnhancedAutocorrelationOptimizer:
    def __init__(self, max_time_seconds=170, max_iterations=1000, elite_size=10):
        self.max_time_seconds = max_time_seconds
        self.max_iterations = max_iterations
        self.elite_size = elite_size
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self.performance_history = []

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
        try:
            n = len(f_sequence)
            c = -np.ones(n)
            a_ub = []
            b_ub = []

            # Efficiently build convolution constraints using precomputed Toeplitz structure
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

            # Use 'highs' method for better numerical stability
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            if result.success:
                g_sequence = result.x
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence, iteration):
        """Returns the direction to move into the sequence using LP-based approach."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        # Normalize sequence
        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Compute maximum convolution value
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

        # Adaptive step size with exponential decay
        t = 0.05 * np.exp(-iteration / 100)
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity
        new_sequence = [max(0.0, x) for x in new_sequence]
        return new_sequence

    def generate_initial_sequence(self):
        """Generate a better initial sequence to start optimization."""
        n = random.randint(100, 1000)
        # Try to generate a sequence that's likely to perform well
        sequence = []
        for i in range(n):
            # Mix of different values to avoid trivial solutions
            if i % 5 == 0:
                sequence.append(random.uniform(100, 1000))
            else:
                sequence.append(random.uniform(0, 100))
        return sequence

    def adaptive_evolution_step(self, current_sequence, population_size=20, elite_fraction=0.2):
        """Perform adaptive evolution step with elite preservation."""
        # Preserve elite individuals
        elite_size = max(1, int(elite_fraction * population_size))
        offspring = []
        
        # Keep elite members
        for i in range(elite_size):
            offspring.append(current_sequence.copy())
            
        # Generate diverse offspring
        for i in range(elite_size, population_size):
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
        
        # Initialize with a good starting sequence
        current_sequence = self.generate_initial_sequence()
        prev_solution = None

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_time_seconds:
                break

            # Compute current performance
            inv_c1 = self.evaluate_fitness(current_sequence)
            self.performance_history.append(inv_c1)

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

            # Alternate between gradient and evolutionary approaches
            if iteration % 3 == 0:
                # Use evolutionary strategy
                candidates = self.adaptive_evolution_step(current_sequence)
                fitness_scores = [self.evaluate_fitness(seq) for seq in candidates]
                best_candidate_idx = np.argmax(fitness_scores)
                candidate_fitness = fitness_scores[best_candidate_idx]
                if candidate_fitness > self.best_inv_c1:
                    current_sequence = candidates[best_candidate_idx].copy()
                    prev_solution = current_sequence.copy()
            else:
                # Use LP-based update
                h_function = self.get_good_direction_to_move_into(current_sequence, iteration)

                if h_function is not None:
                    candidate_sequence = h_function
                    candidate_fitness = self.evaluate_fitness(candidate_sequence)

                    if candidate_fitness > self.best_inv_c1:
                        current_sequence = candidate_sequence
                        prev_solution = candidate_sequence.copy()
                        self.elite_sequences.append((candidate_sequence, candidate_fitness))
                        if len(self.elite_sequences) > self.elite_size:
                            elite_fitnesses = [fitness for _, fitness in self.elite_sequences]
                            top_indices = np.argsort(elite_fitnesses)[-self.elite_size:]
                            self.elite_sequences = [self.elite_sequences[i] for i in top_indices]
                else:
                    # Multi-tiered fallback strategy
                    new_sequence = None

                    # Tier 1: Try mirrored sequence
                    if len(current_sequence) > 1:
                        mirrored = current_sequence[::-1]
                        mirrored = [max(0.0, x) for x in mirrored]
                        if self.evaluate_fitness(mirrored) > self.best_inv_c1:
                            new_sequence = mirrored

                    # Tier 2: Try symmetric pattern
                    if new_sequence is None and len(current_sequence) > 2:
                        sym_sequence = []
                        mid = len(current_sequence) // 2
                        for i in range(len(current_sequence)):
                            if i < mid:
                                sym_sequence.append(current_sequence[i])
                            else:
                                sym_sequence.append(current_sequence[len(current_sequence)-1-i])
                        sym_sequence = [max(0.0, x) for x in sym_sequence]
                        if self.evaluate_fitness(sym_sequence) > self.best_inv_c1:
                            new_sequence = sym_sequence

                    # Tier 3: Random perturbation with better strategy
                    if new_sequence is None:
                        new_sequence = []
                        for x in current_sequence:
                            # More controlled random perturbation
                            perturbation = np.random.normal(0, 0.1 * max(1.0, x))
                            new_sequence.append(max(0.0, x + perturbation))

                    # Apply the best fallback
                    if new_sequence is not None and self.evaluate_fitness(new_sequence) > self.best_inv_c1:
                        current_sequence = new_sequence
                        prev_solution = None

        return self.best_sequence

def search_for_best_sequence(max_time_seconds=170) -> list[float]:
    """Function to search for the best coefficient sequence."""
    optimizer = EnhancedAutocorrelationOptimizer(max_time_seconds=max_time_seconds)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")