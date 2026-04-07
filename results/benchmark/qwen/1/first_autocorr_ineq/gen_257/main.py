# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from typing import List, Optional, Tuple
import time
import random
from collections import OrderedDict

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CacheManager:
    """Manages caching for autocorrelation evaluations."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

# Global cache manager
_cache_manager = CacheManager()

class AutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants with caching."""

    def __init__(self):
        self.cache = _cache_manager

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching to avoid redundant computations.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        cached = self.cache.get(seq_tuple)
        if cached is not None:
            return cached

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self.cache.put(seq_tuple, result)
            return result

        n = len(sequence)

        # Compute convolution using FFT for efficiency
        try:
            padded_seq = np.pad(sequence, (0, len(sequence) - 1), 'constant')
            conv = np.convolve(padded_seq, sequence, mode='valid')
        except Exception:
            # Fallback to direct computation if FFT fails
            conv = np.convolve(sequence, sequence, mode='full')[:2*n-1]

        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self.cache.put(seq_tuple, result)
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self.cache.put(seq_tuple, result)
        return result

# Global evaluator instance
_evaluator = AutocorrelationEvaluator()

class AutocorrelationOptimizer:
    """
    An optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    """

    def __init__(self, max_sequence_length: int = 1000, min_sequence_length: int = 100, max_time_seconds: int = 170):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.max_time_seconds = max_time_seconds

    def compute_c1(self, sequence: np.ndarray) -> float:
        """Compute C₁ constant for the given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')

        # Compute convolution
        try:
            padded_seq = np.pad(sequence, (0, len(sequence) - 1), 'constant')
            conv = np.convolve(padded_seq, sequence, mode='valid')
        except Exception:
            conv = np.convolve(sequence, sequence, mode='full')[:2*n-1]

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

    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float) -> Optional[np.ndarray]:
        """Solve the convolution LP for a given sequence and RHS."""
        try:
            n = len(f_sequence)
            if n == 0:
                return None

            # Objective: minimize -sum(x) (i.e., maximize sum(x))
            c = -np.ones(n)

            # Build constraint matrix - optimized for performance
            a_ub = np.zeros((2 * n - 1 + n, n))
            b_ub = np.zeros(2 * n - 1 + n)

            # Convolution constraints
            for k in range(2 * n - 1):
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        a_ub[k, j] = f_sequence[i]
                b_ub[k] = rhs

            # Non-negativity constraints
            a_ub[2*n-1:, :] = -np.eye(n)
            b_ub[2*n-1:] = np.zeros(n)

            # Solve LP
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                return result.x
            else:
                return None
        except Exception as e:
            return None

    def adaptive_mutate_sequence(self, sequence: List[float], generation: int,
                                max_generations: int, diversity: float = 0.0) -> List[float]:
        """Apply adaptive mutation to a sequence with rate based on generation and diversity."""
        mutated = sequence.copy()
        # Base mutation rate decreases over generations
        base_rate = 0.3 * (1.0 - generation / max_generations)
        # Increase rate if diversity is low
        diversity_factor = max(0.5, 1.0 - diversity / 100.0) if diversity > 0 else 1.0
        mutation_rate = max(0.05, base_rate * diversity_factor)

        # Apply mutation with Gaussian noise
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                noise = random.gauss(0, 0.5 * mutated[i])
                mutated[i] = max(0.01, mutated[i] + noise)
        return mutated

    def local_search(self, sequence: List[float], iterations: int = 5) -> List[float]:
        """Perform local search using gradient approximation."""
        # Convert to numpy array for easier manipulation
        refined = np.array(sequence, dtype=float)
        epsilon = 1e-6

        # Simple gradient ascent using finite differences
        for _ in range(iterations):
            grad = np.zeros_like(refined)
            base_inv_c1 = self.compute_inv_c1(refined.tolist())

            for i in range(len(refined)):
                perturbed = refined.copy()
                perturbed[i] += epsilon
                perturbed = np.maximum(perturbed, 0.01)  # Keep non-negative
                perturbed_inv_c1 = self.compute_inv_c1(perturbed.tolist())
                grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon

            # Update with gradient ascent (since we're maximizing)
            refined += 0.01 * grad
            refined = np.maximum(refined, 0.01)  # Keep non-negative

        return refined.tolist()

    def get_good_direction_to_move_into(self, sequence: List[float]) -> Optional[List[float]]:
        """Get the direction to move into the sequence."""
        try:
            n = len(sequence)
            if n == 0:
                return None

            sequence = np.array(sequence)

            # Normalize for consistent scale
            sum_sequence = np.sum(sequence)
            if sum_sequence < 0.01:
                return None

            # Normalize sequence
            normalized_sequence = sequence * np.sqrt(2 * n) / sum_sequence

            # Compute the RHS for LP
            try:
                padded_seq = np.pad(normalized_sequence, (0, len(normalized_sequence) - 1), 'constant')
                conv = np.convolve(padded_seq, normalized_sequence, mode='valid')
            except Exception:
                conv = np.convolve(normalized_sequence, normalized_sequence, mode='full')[:2*n-1]
            rhs = np.max(conv)

            # Solve LP
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)

            if g_fun is None:
                return None

            # Normalize result back
            sum_g_fun = np.sum(g_fun)
            if sum_g_fun < 0.01:
                return None
            normalized_g_fun = g_fun * np.sqrt(2 * n) / sum_g_fun

            # Apply mixing
            t = 0.01
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence
        except Exception as e:
            return None

    def generate_initial_sequence(self) -> List[float]:
        """Generate a good initial sequence."""
        n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
        # Generate a sequence with some structure to avoid poor starts
        sequence = np.random.exponential(scale=1.0, size=n).tolist()
        # Normalize to have reasonable total weight
        total = sum(sequence)
        if total > 0:
            sequence = [x / total * 100 for x in sequence]
        return sequence

    def run_evolution(self) -> Tuple[List[float], float]:
        """Run the evolutionary optimization process."""
        start_time = time.time()
        _cache_manager.clear()

        # Generate initial sequence
        best_sequence = self.generate_initial_sequence()
        best_inv_c1 = self.compute_inv_c1(best_sequence)

        print(f'Initial inv_C1: {best_inv_c1:.6f}')

        # Evolution loop
        iteration = 0
        max_generations = 500  # Define max generations to control adaptive mutation
        while time.time() - start_time < self.max_time_seconds:
            try:
                # Calculate diversity (simplified proxy)
                diversity = 0.0  # Placeholder for actual diversity calculation

                # Apply adaptive mutation to best sequence
                mutated_sequence = self.adaptive_mutate_sequence(
                    best_sequence, iteration, max_generations, diversity
                )

                # Local search on mutated sequence
                local_refined = self.local_search(mutated_sequence)

                # Evaluate the local refined version
                refined_inv_c1 = self.compute_inv_c1(local_refined)

                if refined_inv_c1 > best_inv_c1:
                    best_sequence = local_refined
                    best_inv_c1 = refined_inv_c1
                    print(f'Iteration {iteration}: New best inv_C1: {best_inv_c1:.6f}')

                    # Early exit if we beat the benchmark
                    if best_inv_c1 > 1.0 / 1.5031:
                        print(f'Beaten benchmark at iteration {iteration}')
                        break

                # Optionally use the direction-based approach for additional exploration
                direction = self.get_good_direction_to_move_into(best_sequence)
                if direction is not None:
                    direction_inv_c1 = self.compute_inv_c1(direction)
                    if direction_inv_c1 > best_inv_c1:
                        best_sequence = direction
                        best_inv_c1 = direction_inv_c1

            except Exception as e:
                print(f'Error in evolution iteration {iteration}: {e}')
                # Perturb slightly if there is an issue
                if best_sequence:
                    index = np.random.randint(len(best_sequence))
                    best_sequence[index] = max(0, best_sequence[index] + np.random.normal(0, 0.5))

            iteration += 1

        return best_sequence, best_inv_c1

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationOptimizer()
    best_sequence, _ = optimizer.run_evolution()
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")