# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
from scipy import optimize
import time
from collections import deque
import random

# Configuration parameters
ADAPTIVE_THRESHOLD = 0.01
LOCAL_SEARCH_ITERATIONS = 20
MAX_LOCAL_SEARCHES = 3
MIN_SEQUENCE_LENGTH = 50
MAX_SEQUENCE_LENGTH = 500

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class AutocorrelationOptimizer:
    def __init__(self):
        self.best_score = float('-inf')
        self.best_sequence = None
        self.history = deque(maxlen=10)
        self.start_time = time.time()
        
    def compute_c1(self, sequence):
        """
        Compute C1 for a given sequence.
        C1 = 2*n*max(convolution) / (sum(sequence))^2
        We want to maximize 1/C1, which means minimizing C1.
        """
        if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
            return float('inf')

        # Use FFT-based convolution for efficiency
        convolved = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(convolved[len(sequence)-1:])  # Only consider relevant part
        sum_seq = sum(sequence)

        # Return C1 value
        return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

    def evaluate_sequence(self, sequence):
        """
        Evaluate a sequence by computing 1/C1.
        Returns negative because differential_evolution minimizes.
        """
        c1 = self.compute_c1(sequence)
        if c1 == float('inf') or c1 > 1e10:  # Prevent overflow
            return float('inf')
        return -1.0 / c1  # We want to maximize 1/C1, so minimize -1/C1

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
        a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
        b_ub_nonneg = np.zeros(n)  # Zero vector

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
            if result.success:
                g_sequence = result.x
                return g_sequence
        except:
            pass
        return None

    def get_good_direction_to_move_into(self, sequence: list[float]) -> list[float] | None:
        """Returns the direction to move into the sequence."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)

        # Avoid division by zero
        if sum_sequence < 1e-10:
            return None

        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
        conv_result = np.convolve(normalized_sequence, normalized_sequence)
        rhs = np.max(conv_result)

        g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is None:
            return None
        
        sum_g_fun = np.sum(g_fun)
        if sum_g_fun < 1e-10:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
        t = 0.01
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        return new_sequence

    def adaptive_search(self, current_seq, current_score):
        """
        Perform adaptive search combining global and local optimization.
        """
        n = len(current_seq)
        
        # Check if we should perform local search
        if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < ADAPTIVE_THRESHOLD:
            # Try local gradient-based update
            direction = self.get_good_direction_to_move_into(current_seq)
            if direction is not None:
                # Apply gradient update
                updated_seq = direction
                updated_score = self.evaluate_sequence(updated_seq)

                if updated_score > current_score:
                    return updated_seq, updated_score, True  # Success with local update

        # Default to global optimization approach
        return current_seq, current_score, False

    def create_initial_sequences(self, n_samples=10):
        """
        Create diverse initial sequences for optimization.
        """
        sequences = []

        # Random sequences with different configurations
        for _ in range(n_samples):
            n = np.random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
            seq = np.random.uniform(0, 100, n)
            # Add some structure
            if np.random.random() < 0.3:
                # Add a few large values
                idxs = np.random.choice(n, size=min(5, n//4), replace=False)
                seq[idxs] *= np.random.uniform(5, 20)
            sequences.append(seq)

        # Include some known good structures
        sequences.append(np.array([1.0] * 100))  # Uniform
        sequences.append(np.array([1.0] * 50 + [0.0] * 50))  # Step function

        return sequences

    def refine_sequence(self, sequence):
        """Apply local refinement to improve sequence quality."""
        current_seq = np.array(sequence)
        
        # Apply gradient-like adjustments
        for _ in range(10):  # Limited iterations for efficiency
            # Perform simple convolution peak reduction
            conv = fftconvolve(current_seq, current_seq, mode='full')
            conv_part = conv[len(current_seq)-1:]
            max_conv = np.max(conv_part)
            
            # Identify and reduce contributions to peak convolution
            max_indices = np.where(conv_part >= 0.9 * max_conv)[0]
            for idx in max_indices[:min(3, len(max_indices))]:
                for offset in [-1, 0, 1]:
                    pos = idx + offset
                    if 0 <= pos < len(current_seq):
                        current_seq[pos] *= 0.995
        
        # Ensure non-negativity and clipping
        current_seq = np.clip(current_seq, 0, 1000)
        if np.sum(current_seq) < 0.01:
            current_seq[0] = 0.1
            
        return current_seq.tolist()

    def run_optimization(self):
        """Main optimization loop."""
        # Initialize with diverse starting sequences
        initial_sequences = self.create_initial_sequences()

        # Try multiple search strategies
        for attempt in range(10):  # Increased attempts
            if time.time() - self.start_time > 170:  # Leave 10 seconds for final processing
                break

            try:
                # Select a random initial sequence
                initial_seq = initial_sequences[np.random.randint(len(initial_sequences))]

                # Ensure minimum length
                if len(initial_seq) < MIN_SEQUENCE_LENGTH:
                    initial_seq = np.pad(initial_seq, (0, MIN_SEQUENCE_LENGTH - len(initial_seq)),
                                       mode='constant', constant_values=0)

                # Initialize with local search if possible
                current_seq = initial_seq.copy()
                current_score = self.evaluate_sequence(current_seq)
                self.history.append(current_score)

                # Adaptive optimization loop
                local_searches = 0
                for iteration in range(100):  # Reduced iterations due to time constraint
                    if time.time() - self.start_time > 170:
                        break

                    # Try adaptive search
                    new_seq, new_score, was_local = self.adaptive_search(current_seq, current_score)

                    if was_local:
                        local_searches += 1
                        current_seq = new_seq
                        current_score = new_score
                        self.history.append(current_score)

                        if local_searches >= MAX_LOCAL_SEARCHES:
                            # Switch to global search
                            local_searches = 0

                    # Occasionally do global optimization
                    elif np.random.random() < 0.1 and local_searches < MAX_LOCAL_SEARCHES:
                        # Use differential evolution for diversity
                        bounds = [(0.0, 1000.0)] * len(current_seq)
                        result = differential_evolution(
                            self.evaluate_sequence,
                            bounds,
                            maxiter=20,
                            popsize=10,
                            mutation=(0.5, 1),
                            recombination=0.7,
                            seed=42+attempt+iteration,
                            polish=False
                        )
                        if result.success:
                            current_seq = result.x
                            current_score = self.evaluate_sequence(current_seq)
                            self.history.append(current_score)

                    # Early termination if improving
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.best_sequence = current_seq.copy()

            except Exception as e:
                continue

        # If we didn't find anything, fallback to uniform sequence
        if self.best_sequence is None:
            self.best_sequence = np.array([1.0] * 100)

        # Final refinement pass
        final_refinement = self.refine_sequence(self.best_sequence)
        final_inv_c1 = 1.0 / self.compute_c1(final_refinement)
        if final_inv_c1 > 1.0 / self.compute_c1(self.best_sequence):
            self.best_sequence = final_refinement

        return self.best_sequence.tolist()

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
