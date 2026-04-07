# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random
from typing import List, Optional, Tuple
from collections import OrderedDict

# Global constants
MAX_TIME_SECONDS = 180
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 1000
BENCHMARK_RATIO = 1.5031
IMPROVEMENT_THRESHOLD = 1e-6
MAX_ITERATIONS = 1000
CACHE_SIZE = 1000
INITIAL_POPULATION_COUNT = 20
ADAPTIVE_T_START = 0.1
ADAPTIVE_T_DECAY = 0.95
STAGNATION_THRESHOLD = 15
RESTART_THRESHOLD = 5
DIVERSITY_INJECTION_INTERVAL = 10

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class ConvolutionCache:
    """Manages cached convolution computations to improve performance."""
    def __init__(self, max_size: int = CACHE_SIZE):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: float):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)
        self.cache[key] = value

class AutocorrelationOptimizer:
    def __init__(self):
        self.cache = ConvolutionCache()
        self.best_known_sequence = None
        self.best_known_c1 = float('inf')
        
    def _get_cache_key(self, sequence: List[float]) -> str:
        """Generate a hashable key for caching with deterministic rounding."""
        rounded_sequence = tuple(round(x, 8) for x in sequence)
        return str(rounded_sequence)
        
    def convolve_fft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency with optimized padding."""
        n = len(a)
        # For very small sequences, use direct convolution for numerical stability
        if n < 10:
            return np.convolve(a, b, mode='full')
        
        # Use next power of 2 for better FFT performance
        fft_size = 1 << (n - 1).bit_length()  # Next power of 2
        if fft_size < 2 * n - 1:
            fft_size *= 2

        # Pad to fft_size
        a_padded = np.pad(a, (0, fft_size - n), 'constant')
        b_padded = np.pad(b, (0, fft_size - n), 'constant')

        # Perform convolution in frequency domain
        a_fft = fft(a_padded)
        b_fft = fft(b_padded)
        conv_result = ifft(a_fft * np.conj(b_fft))
        return np.real(conv_result[:2*n-1])

    def compute_c1_constant(self, sequence: List[float]) -> float:
        """Computes the C1 constant for a given sequence."""
        key = self._get_cache_key(sequence)
        cached_result = self.cache.get(key)
        if cached_result is not None:
            return cached_result
            
        n = len(sequence)
        if n == 0:
            result = float('inf')
        else:
            # Compute convolution using FFT
            conv = self.convolve_fft(np.array(sequence), np.array(sequence))
            max_conv = np.max(conv)
            sum_sq = np.sum(sequence)**2
            
            if sum_sq < 1e-10:
                result = float('inf')
            else:
                c1 = 2 * n * max_conv / sum_sq
                result = c1
                
        self.cache.put(key, result)
        return result

    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float, n: int) -> Optional[np.ndarray]:
        """Solves the convolution LP for a given sequence and RHS with fallbacks."""
        try:
            # Create the constraint matrix using the convolution structure
            a_ub = []
            b_ub = []

            # Generate convolution constraints
            f_seq = np.array(f_sequence)
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_seq[i]
                a_ub.append(row)
                b_ub.append(rhs)

            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            # Define objective function (we want to minimize negative sum)
            c = -np.ones(n)
            
            # Solve the linear program with fallbacks
            result = None
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True, 'maxiter': 1000})
            except:
                try:
                    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point', options={'maxiter': 1000})
                except:
                    # Try with relaxed constraints
                    try:
                        b_ub_relaxed = np.array(b_ub) * 0.95
                        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub_relaxed, method='highs', options={'presolve': True, 'maxiter': 1000})
                    except:
                        # Try simplex method as last resort
                        try:
                            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex', options={'maxiter': 1000})
                        except:
                            return None  # All methods failed

            if result is not None and result.success:
                g_sequence = result.x
                # Ensure non-negativity due to numerical errors
                g_sequence = np.maximum(g_sequence, 0)
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def get_good_direction_to_move_into(self, sequence: List[float], generation: int = 0) -> Optional[List[float]]:
        """Returns the direction to move into the sequence with adaptive step size."""
        try:
            n = len(sequence)
            if n < MIN_SEQ_LENGTH:
                return None
                
            # Normalize sequence for processing
            sum_sequence = np.sum(sequence)
            if sum_sequence < 1e-10:
                return None
                
            # Normalize to avoid numerical issues
            normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence
            
            # Compute the target RHS for LP solver
            conv = self.convolve_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(conv)
            
            # Solve the LP optimization
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs, n)
            if g_fun is None:
                return None
                
            # Normalize the result and create new sequence
            sum_g = np.sum(g_fun)
            if sum_g < 1e-10:
                return None
                
            normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g
            
            # Adaptive step size with improved decay formula
            t = ADAPTIVE_T_START * (ADAPTIVE_T_DECAY ** (generation // 3))
            new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
            
            return new_sequence
        except Exception:
            return None

    def initialize_sequence(self) -> List[float]:
        """Initialize a promising sequence for optimization."""
        # Try different initialization strategies to find a good starting point
        strategy = random.choice(['harmonic', 'random', 'spike', 'wavelet', 'gaussian'])
        
        if strategy == 'harmonic':
            # Start with a structured sequence
            n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            # Create a sequence with decreasing values to encourage sparsity
            sequence = [1.0 / (i + 1) for i in range(n)]
            # Normalize to have reasonable magnitude
            total = sum(sequence)
            sequence = [x * 2.0 / total for x in sequence]
        elif strategy == 'spike':
            n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            # Create a sparse sequence with one large spike
            sequence = [0.0] * n
            spike_idx = random.randint(0, n-1)
            sequence[spike_idx] = random.uniform(1.0, 5.0)
        elif strategy == 'wavelet':
            # Create a wavelet-inspired sequence
            n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            sequence = [random.uniform(0.1, 2.0) for _ in range(n)]
            # Apply a simple low-pass filter to make it smoother
            for i in range(1, n-1):
                sequence[i] = 0.3 * sequence[i-1] + 0.4 * sequence[i] + 0.3 * sequence[i+1]
        elif strategy == 'gaussian':
            # Create a Gaussian-like sequence
            n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            mean = n // 2
            std = n // 4
            sequence = [np.exp(-0.5 * ((i - mean) / std) ** 2) for i in range(n)]
            # Normalize
            total = sum(sequence)
            sequence = [x / total for x in sequence]
        else:  # random
            n = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            sequence = [random.uniform(0.1, 2.0) for _ in range(n)]
            
        return sequence

    def multi_start_initialization(self) -> Tuple[List[float], float]:
        """Perform multi-start initialization to find a promising starting point."""
        best_sequence = None
        best_c1 = float('inf')

        for _ in range(INITIAL_POPULATION_COUNT):
            candidate_sequence = self.initialize_sequence()
            candidate_c1 = self.compute_c1_constant(candidate_sequence)
            if candidate_c1 < best_c1:
                best_c1 = candidate_c1
                best_sequence = candidate_sequence

        if best_sequence is None:
            best_sequence = self.initialize_sequence()
            
        return best_sequence, best_c1

    def hybrid_mutation(self, sequence: List[float], iterations: int) -> List[float]:
        """Apply hybrid mutation strategies for exploration."""
        n = max(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, int(len(sequence) * 0.95)))
        
        if random.random() < 0.5:
            # Structured perturbation for exploration
            mutated_sequence = sequence.copy()
            # Modify some elements with structured changes
            for i in range(len(mutated_sequence)):
                if random.random() < 0.3:
                    # Apply either random change or systematic perturbation
                    if random.random() < 0.5:
                        mutated_sequence[i] *= random.uniform(0.8, 1.2)
                    else:
                        mutated_sequence[i] = random.uniform(0.1, 2.0)
            return mutated_sequence
        else:
            # Inject diversity periodically
            if iterations % DIVERSITY_INJECTION_INTERVAL == 0:
                # Introduce a few random spikes to maintain diversity
                mutated_sequence = sequence.copy()
                for i in range(min(3, len(mutated_sequence))):  # Perturb up to 3 elements
                    idx = random.randint(0, len(mutated_sequence)-1)
                    mutated_sequence[idx] = random.uniform(0.1, 2.0)
                return mutated_sequence
            else:
                # Random perturbation
                return [random.uniform(0.1, 2.0) for _ in range(n)]

    def adaptive_restart_logic(self, start_time: float, stagnation_counter: int, 
                              last_improvement_gen: int, iterations: int) -> bool:
        """Implement adaptive restart logic."""
        if iterations - last_improvement_gen > STAGNATION_THRESHOLD:
            stagnation_counter += 1
            if stagnation_counter > RESTART_THRESHOLD:
                # Exponential backoff for restarts
                restart_delay = 2 ** stagnation_counter
                if time.time() - start_time > restart_delay:
                    return True
        return False

    def search_for_best_sequence(self) -> List[float]:
        """Search for the best coefficient sequence."""
        start_time = time.time()
        
        # Multi-start initialization
        best_sequence, best_c1 = self.multi_start_initialization()
        prev_c1 = best_c1
        stagnation_counter = 0
        last_improvement_gen = 0
        iterations = 0
        
        while time.time() - start_time < MAX_TIME_SECONDS - 5 and iterations < MAX_ITERATIONS:
            # Try gradient-based improvement
            improved_sequence = self.get_good_direction_to_move_into(best_sequence, iterations)
            
            if improved_sequence is not None:
                new_c1 = self.compute_c1_constant(improved_sequence)
                if new_c1 < prev_c1:
                    best_sequence = improved_sequence
                    prev_c1 = new_c1
                    iterations += 1
                    stagnation_counter = 0
                    last_improvement_gen = iterations
                    continue
                    
            # Check for stagnation and apply adaptive restart logic
            if self.adaptive_restart_logic(start_time, stagnation_counter, last_improvement_gen, iterations):
                # Restart with new initialization
                best_sequence, prev_c1 = self.multi_start_initialization()
                last_improvement_gen = iterations
                stagnation_counter = 0
            else:
                # Apply hybrid mutation
                best_sequence = self.hybrid_mutation(best_sequence, iterations)
                
            iterations += 1

        # Final check and refinement
        final_c1 = self.compute_c1_constant(best_sequence)
        if final_c1 >= 1.5031:
            # Attempt one final optimization
            refined = self.get_good_direction_to_move_into(best_sequence, MAX_ITERATIONS)
            if refined is not None:
                test_c1 = self.compute_c1_constant(refined)
                if test_c1 < final_c1:
                    best_sequence = refined
                    
        return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main entry point for searching the best sequence."""
    optimizer = AutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")