# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from collections import OrderedDict
from typing import List, Optional, Tuple

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_GENERATIONS = 500
POPULATION_SIZE = 100
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
INITIAL_POPULATION_COUNT = 10
ADAPTIVE_T_START = 0.05
ADAPTIVE_T_DECAY = 0.98

class ConvolutionHelper:
    @staticmethod
    def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency."""
        n = len(a)
        fft_size = 1 << (n - 1).bit_length()  # Next power of 2
        if fft_size < 2 * n - 1:
            fft_size *= 2

        a_padded = np.pad(a, (0, fft_size - n), 'constant')
        b_padded = np.pad(b, (0, fft_size - n), 'constant')

        a_fft = fft(a_padded)
        b_fft = fft(b_padded)
        conv_result = ifft(a_fft * np.conj(b_fft))
        return np.real(conv_result[:2*n-1])

    @staticmethod
    def compute_c1_constant(sequence: List[float]) -> float:
        """Computes the C1 constant for a given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')

        conv = ConvolutionHelper.convolve_fft(np.array(sequence), np.array(sequence))
        max_conv = np.max(conv)
        sum_sq = np.sum(sequence)**2

        if sum_sq < 1e-10:
            return float('inf')

        c1 = 2 * n * max_conv / sum_sq
        return c1

class CacheManager:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: tuple) -> Optional[float]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: tuple, value: float) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

class OptimizationEngine:
    def __init__(self):
        self.cache = CacheManager(1000)
        self.conv_helper = ConvolutionHelper()

    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float, n: int) -> Optional[np.ndarray]:
        """Solves the convolution LP for a given sequence and RHS with constraint relaxation."""
        try:
            a_ub = []
            b_ub = []

            f_seq = np.array(f_sequence)
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_seq[i]
                a_ub.append(row)
                b_ub.append(rhs)

            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            c = -np.ones(n)
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                g_sequence = result.x
                g_sequence = np.maximum(g_sequence, 0)
                return g_sequence
            else:
                try:
                    b_ub_relaxed = np.array(b_ub) * 0.95
                    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub_relaxed, method='highs')
                    if result.success:
                        g_sequence = result.x
                        g_sequence = np.maximum(g_sequence, 0)
                        return g_sequence
                except Exception:
                    pass
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence: List[float], generation: int = 0) -> Optional[List[float]]:
        """Returns the direction to move into the sequence with adaptive step size."""
        try:
            n = len(sequence)
            if n < MIN_SEQ_LENGTH:
                return None

            sum_sequence = np.sum(sequence)
            if sum_sequence < 1e-10:
                return None

            normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence
            conv = self.conv_helper.convolve_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(conv)

            g_fun = self.solve_convolution_lp(normalized_sequence, rhs, n)
            if g_fun is None:
                return None

            sum_g = np.sum(g_fun)
            if sum_g < 1e-10:
                return None

            normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g

            t = ADAPTIVE_T_START * (ADAPTIVE_T_DECAY ** generation)
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

            return new_sequence
        except Exception:
            return None

    def initialize_sequence(self) -> List[float]:
        """Initialize a promising sequence for optimization with strategic diversity."""
        strategy = random.choice(['harmonic', 'random', 'spike'])

        if strategy == 'harmonic':
            n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            sequence = [1.0 / (i + 1) for i in range(n)]
            total = sum(sequence)
            sequence = [x * 2.0 / total for x in sequence]
        elif strategy == 'spike':
            n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            sequence = [0.0] * n
            spike_idx = random.randint(0, n-1)
            sequence[spike_idx] = random.uniform(1.0, 5.0)
        else:  # random
            n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            sequence = [random.uniform(0.1, 2.0) for _ in range(n)]

        return sequence

    def search_for_best_sequence(self) -> List[float]:
        """Search for the best coefficient sequence with enhanced initialization."""
        start_time = time.time()

        # Multi-start initialization
        best_sequence = None
        best_c1 = float('inf')

        for _ in range(INITIAL_POPULATION_COUNT):
            candidate_sequence = self.initialize_sequence()
            key = tuple(round(x, 6) for x in candidate_sequence)
            cached = self.cache.get(key)
            if cached is not None:
                candidate_c1 = cached
            else:
                candidate_c1 = self.conv_helper.compute_c1_constant(candidate_sequence)
                self.cache.put(key, candidate_c1)
                
            if candidate_c1 < best_c1:
                best_c1 = candidate_c1
                best_sequence = candidate_sequence

        if best_sequence is None:
            best_sequence = self.initialize_sequence()

        # Initialize optimization parameters
        prev_c1 = best_c1
        stagnation_counter = 0
        last_improvement_gen = 0

        # Evolutionary loop
        for gen in range(MAX_GENERATIONS):
            if time.time() - start_time > MAX_TIME_SECONDS - 5:
                break

            improved_sequence = self.get_good_direction_to_move_into(best_sequence, gen)

            if improved_sequence is not None:
                key = tuple(round(x, 6) for x in improved_sequence)
                cached = self.cache.get(key)
                if cached is not None:
                    new_c1 = cached
                else:
                    new_c1 = self.conv_helper.compute_c1_constant(improved_sequence)
                    self.cache.put(key, new_c1)
                    
                if new_c1 < prev_c1:
                    best_sequence = improved_sequence
                    prev_c1 = new_c1
                    stagnation_counter = 0
                    last_improvement_gen = gen
                    continue

            # Stagnation check and restart if needed
            if gen - last_improvement_gen > 20:
                stagnation_counter += 1
                if stagnation_counter > 3:
                    best_sequence = self.initialize_sequence()
                    key = tuple(round(x, 6) for x in best_sequence)
                    cached = self.cache.get(key)
                    if cached is not None:
                        prev_c1 = cached
                    else:
                        prev_c1 = self.conv_helper.compute_c1_constant(best_sequence)
                        self.cache.put(key, prev_c1)
                    last_improvement_gen = gen
                    stagnation_counter = 0
            else:
                n = max(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, int(len(best_sequence) * 0.95)))
                if random.random() < 0.5:
                    mutated_sequence = best_sequence.copy()
                    for i in range(len(mutated_sequence)):
                        if random.random() < MUTATION_RATE:
                            if random.random() < 0.5:
                                mutated_sequence[i] *= random.uniform(0.8, 1.2)
                            else:
                                mutated_sequence[i] = random.uniform(0.1, 2.0)
                    best_sequence = mutated_sequence
                else:
                    best_sequence = [random.uniform(0.1, 2.0) for _ in range(n)]

        # Final check and refinement
        final_c1 = self.conv_helper.compute_c1_constant(best_sequence)
        if final_c1 >= 1.5031:
            refined = self.get_good_direction_to_move_into(best_sequence, MAX_GENERATIONS)
            if refined is not None:
                test_c1 = self.conv_helper.compute_c1_constant(refined)
                if test_c1 < final_c1:
                    best_sequence = refined

        return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main entry point for searching the best sequence."""
    engine = OptimizationEngine()
    return engine.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")