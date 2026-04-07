# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    """
    An improved optimizer for finding step functions that maximize 1/C₁
    """
    
    def __init__(self):
        self.max_time = 170  # seconds
        self.start_time = None

    def compute_convolution_fft(self, seq):
        """
        Compute convolution using FFT for better performance.
        """
        n = len(seq)
        if n < 1:
            return np.array([])
        padded_seq = np.pad(seq, (0, n - 1), mode='constant')
        fft_seq = fft(padded_seq)
        conv_result = ifft(fft_seq * np.conj(fft_seq)).real
        return conv_result[:2 * n - 1]

    def calculate_fitness(self, sequence):
        """
        Calculate fitness as inverse of C1, i.e., (sum(a))^2 / (2*n*max(conv)).
        """
        n = len(sequence)
        if n < 1:
            return 0.0
        
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return 0.0
        
        try:
            conv = self.compute_convolution_fft(sequence)
            max_conv = np.max(conv)
        except Exception:
            # Fallback to direct convolution if FFT fails
            conv = np.convolve(sequence, sequence)
            max_conv = np.max(conv)
        
        if max_conv < 1e-10:
            return 0.0
        
        # Calculate fitness: (sum(a))^2 / (2*n*max(conv))
        fitness = (sum_a ** 2) / (2 * n * max_conv)
        return fitness

    def solve_convolution_lp(self, f_sequence, rhs):
        """
        Solves the convolution LP for a given sequence and RHS.
        """
        n = len(f_sequence)
        if n < 1:
            return None

        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Generate constraint matrix efficiently
        try:
            for k in range(2 * n - 1):
                row = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        row[j] = f_sequence[i]
                a_ub.append(row)
                b_ub.append(rhs)
        except Exception:
            # Fallback to manual construction if something goes wrong
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

        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True})
            if result.success:
                g_sequence = result.x
                return g_sequence
            else:
                return None
        except Exception:
            return None

    def solve_convolution_lp_with_fallback(self, f_sequence, rhs):
        """
        Solves the convolution LP with fallback strategies.
        """
        n = len(f_sequence)
        if n < 1:
            return None

        # Try normal LP approach first
        g_fun = self.solve_convolution_lp(f_sequence, rhs)
        if g_fun is not None:
            return g_fun

        # Fallback 1: Try with slightly relaxed constraints
        try:
            g_fun = self.solve_convolution_lp(f_sequence, rhs * 1.01)
            if g_fun is not None:
                return g_fun
        except Exception:
            pass

        # Fallback 2: Return symmetric pattern (mirrored)
        try:
            g_fun = [f_sequence[n // 2]] * n
            sum_g = np.sum(g_fun)
            if sum_g > 0:
                g_fun = [x / sum_g for x in g_fun]
                return g_fun
        except Exception:
            pass

        # Fallback 3: Return simple uniform pattern
        try:
            return np.ones(n) / n
        except Exception:
            pass

        # Fallback 4: Return original sequence (no change)
        return f_sequence

    def get_good_direction_to_move_into(self, sequence):
        """
        Returns the direction to move into the sequence with improved strategies.
        """
        n = len(sequence)
        if n < 1:
            return None

        # Compute current convolution
        try:
            conv_result = self.compute_convolution_fft(sequence)
            max_conv = np.max(conv_result)
        except Exception:
            # Fallback to direct convolution if FFT fails
            conv_result = np.convolve(sequence, sequence)
            max_conv = np.max(conv_result)

        # Normalize sequence for better numerics
        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            return None

        normalized_sequence = [x / sum_sequence for x in sequence]

        # Adaptive step size
        base_t = 0.01
        t = base_t * (1.0 / (1.0 + n / 1000.0))

        # Solve LP with better initialization and fallback strategies
        g_fun = self.solve_convolution_lp_with_fallback(normalized_sequence, max_conv)

        if g_fun is None:
            # Try simple gradient ascent as fallback
            try:
                # Simple gradient ascent - move towards increasing values
                g_fun = [max(0, x + 0.01 * (random.random() - 0.5)) for x in sequence]
                # Normalize again
                sum_g = np.sum(g_fun)
                if sum_g > 0:
                    g_fun = [x / sum_g for x in g_fun]
            except Exception:
                return None

        # Apply the update
        if g_fun is not None:
            sum_g = np.sum(g_fun)
            if sum_g > 0:
                normalized_g_fun = [x / sum_g for x in g_fun]
                new_sequence = [
                    (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
                ]
                return new_sequence

        return sequence

    def search_for_best_sequence(self):
        """
        Function to search for the best coefficient sequence with improved strategy.
        """
        self.start_time = time.time()
        best_sequence = None
        best_fitness = 0.0
        num_starts = 20  # Increase number of starts for better coverage

        # Multi-start strategy to avoid local minima
        for start_idx in range(num_starts):
            if time.time() - self.start_time > self.max_time:
                break
                
            # Randomly initialize sequence length and values
            n = np.random.randint(100, 1000)
            current_sequence = [random.random() * 10 for _ in range(n)]
            
            # Local search with multiple iterations
            local_max_iter = 100
            for iter_idx in range(local_max_iter):
                if time.time() - self.start_time > self.max_time:
                    break
                    
                h_function = self.get_good_direction_to_move_into(current_sequence)
                if h_function is not None:
                    current_sequence = h_function
                else:
                    # If we can't improve, try a new random sequence
                    n = np.random.randint(100, 1000)
                    current_sequence = [random.random() * 10 for _ in range(n)]
                
                # Evaluate fitness
                current_fitness = self.calculate_fitness(current_sequence)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_sequence = current_sequence[:]
        
        # Final check to ensure we have a valid sequence
        if best_sequence is None:
            n = np.random.randint(100, 1000)
            best_sequence = [random.random() * 10 for _ in range(n)]

        return best_sequence

def evaluate_performance(sequence):
    """
    Evaluates the performance of the found sequence.
    """
    optimizer = AutocorrelationOptimizer()
    fitness = optimizer.calculate_fitness(sequence)
    inv_c1 = fitness
    benchmark_ratio = 1.5031 / fitness if fitness > 0 else 0.0
    eval_time = time.perf_counter() - optimizer.start_time if optimizer.start_time else 0.0
    
    return {
        'inv_c1': inv_c1,
        'benchmark_ratio': benchmark_ratio,
        'eval_time': eval_time
    }

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    optimizer = AutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
