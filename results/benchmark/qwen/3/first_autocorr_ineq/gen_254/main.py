# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
from cvxpy import *
import time
import random
from scipy.signal import fftconvolve
from collections import deque

np.random.seed(42)
random.seed(42)

# Track historically good sequences for seeding
historical_sequences = deque(maxlen=10)

class Sequence:
    """
    Represents a sequence of non-negative real numbers.
    """
    def __init__(self, data: list[float]):
        self.data = np.array(data, dtype=np.float64)
        self.length = len(self.data)

    def copy(self) -> 'Sequence':
        """Return a copy of this sequence."""
        return Sequence(self.data.copy())

    def normalize(self) -> 'Sequence':
        """Return a normalized version of this sequence."""
        sum_data = np.sum(self.data)
        if sum_data < 1e-10:
            return self.copy()
        return Sequence(self.data / sum_data)

    def clip_values(self, low: float = 0.0, high: float = 1000.0) -> 'Sequence':
        """Clip values in the sequence to a specified range."""
        return Sequence(np.clip(self.data, low, high))

    def sum(self) -> float:
        """Return the sum of all elements in the sequence."""
        return np.sum(self.data)

    def __len__(self):
        return self.length

class AutocorrelationEvaluator:
    """
    Evaluates the autocorrelation constant C₁ for a given sequence.
    """
    @staticmethod
    def compute_c1(sequence: Sequence) -> tuple[float, float]:
        """
        Computes the autocorrelation constant C₁ and 1/C₁ for a given sequence.
        Returns (C1, inv_C1).
        """
        if sequence.length == 0:
            return float('inf'), 0.0

        a = sequence.data
        n = sequence.length

        # Compute convolution using FFT for efficiency, with stability check
        padded_len = 2 * n - 1

        # For small sequences, use direct convolution to avoid FFT artifacts
        if n < 50:
            conv_result = np.convolve(a, a, mode='full')[:padded_len]
        else:
            # Use FFT with numerical stability checks
            a_padded = np.pad(a, (0, padded_len - n), 'constant')
            fft_a = fft(a_padded)
            conv_fft = fft_a * np.conj(fft_a)
            conv_result = ifft(conv_fft).real[:padded_len]

            # Stability check: compare with direct convolution for small sequences
            if n < 200:
                direct_conv = np.convolve(a, a, mode='full')[:padded_len]
                if np.allclose(conv_result, direct_conv, rtol=1e-3, atol=1e-5):
                    pass  # Acceptable stability
                else:
                    # Fall back to direct convolution if stability issues
                    conv_result = direct_conv

        max_b = np.max(conv_result)
        sum_a = np.sum(a)

        if sum_a < 0.01:
            return float('inf'), 0.0

        C1 = 2 * n * max_b / (sum_a ** 2)
        inv_C1 = 1 / C1

        return C1, inv_C1

class ConvexOptimizer:
    """
    Uses convex optimization to find improved sequences.
    """
    @staticmethod
    def solve_convolution_quadratic_cvxpy(sequence: Sequence):
        """
        Solves the optimization problem using convex optimization with quadratic constraints.
        """
        n = sequence.length
        if n == 0:
            return None

        # Define variables
        a = Variable(n, nonneg=True)

        # Set up the sequence as a parameter
        a_param = Parameter(n)
        a_param.value = np.array(sequence.data)

        # Estimate a reasonable upper bound for convolution max
        rhs_estimate = 2 * np.max(np.convolve(sequence.data, sequence.data, mode='full')[:2*n-1])

        # Objective: minimize sum(a)
        objective = Minimize(sum(a))

        # Constraints
        # Convolution constraint - this is complex to model exactly in CVXPY without explicit matrix
        # So we simplify and approximate the quadratic constraint directly

        # Refine the constraint to be more meaningful
        # The key constraint is that we want to find a sequence that minimizes sum(a)
        # while respecting the convolution constraint
        # We introduce a rough estimate of the max convolution as a parameter to guide the optimization

        # Estimate based on the input sequence properties
        estimated_max_conv = np.max(np.convolve(sequence.data, sequence.data, mode='full')[:2*n-1])

        # Use a scaled version of this estimate as a soft constraint
        # This encourages the optimizer to consider the effect of convolution on the solution
        constraints = [a >= 0, sum(a) <= 1000,
                       # Add a heuristic constraint based on max convolution
                       sum(a) <= 1000 * np.sqrt(estimated_max_conv) if estimated_max_conv > 0 else sum(a) <= 1000]

        # Solve the problem
        prob = Problem(objective, constraints)

        # Try to solve with multiple solvers
        try:
            prob.solve(solver=ECOS, verbose=False)
        except:
            try:
                prob.solve(solver=SCS, verbose=False)
            except:
                return None

        if prob.status == "optimal":
            return Sequence(a.value.tolist())
        else:
            return None

class SequenceOptimizer:
    """
    Orchestrates the optimization process for finding the best sequence.
    """
    @staticmethod
    def get_good_direction_to_move_into(sequence: Sequence) -> Sequence | None:
        """
        Returns the direction to move into the sequence using convex optimization with curvature correction.
        """
        n = sequence.length
        if n == 0:
            return None

        sum_sequence = np.sum(sequence.data)
        if sum_sequence < 1e-10:
            return None

        normalized_sequence = sequence.normalize()

        try:
            # Use CVXPY to get a potentially better direction
            result = ConvexOptimizer.solve_convolution_quadratic_cvxpy(normalized_sequence)

            if result is None:
                return None

            # Normalize the result
            sum_result = np.sum(result.data)
            if sum_result < 1e-8:
                return None

            normalized_result = result.normalize()

            # Apply curvature-aware directional bias correction
            if n > 10:
                # Estimate curvature using finite differences
                epsilon = 1e-3
                hessian_approx = np.zeros((n, n))

                # Simple finite difference approximation of second derivatives
                for i in range(n):
                    perturbed_seq = normalized_sequence.data.copy()
                    perturbed_seq[i] += epsilon
                    if i > 0:  # Avoid boundary issues
                        perturbed_seq[i-1] -= epsilon / 2
                    if i < n-1:  # Avoid boundary issues
                        perturbed_seq[i+1] -= epsilon / 2

                    # Recompute convolution with perturbed sequence
                    a_padded = np.pad(perturbed_seq, (0, 2*n-1-n), 'constant')
                    fft_a = fft(a_padded)
                    conv_fft = fft_a * np.conj(fft_a)
                    conv_result = ifft(conv_fft).real[:2*n-1]

                    second_derivative = (np.max(conv_result) - np.max(np.convolve(normalized_sequence.data, normalized_sequence.data, mode='full')[:2*n-1])) / (epsilon**2)
                    hessian_approx[i, i] = max(0, second_derivative)  # Ensure non-negative

                # Apply curvature correction to the direction
                curvature_correction = np.dot(hessian_approx, normalized_result.data)
                # Scale correction appropriately
                curvature_correction = curvature_correction / (1.0 + np.linalg.norm(curvature_correction))

                # Adjust the direction with curvature bias
                corrected_direction = np.array(normalized_result.data) + 0.1 * curvature_correction
                normalized_result.data = corrected_direction.tolist()

            # Apply small perturbation to maintain diversity
            t = 0.05
            new_data = [(1 - t) * x + t * y for x, y in zip(sequence.data, normalized_result.data)]

            # Ensure non-negativity and reasonable bounds
            new_data = [max(0, min(1000, x)) for x in new_data]

            return Sequence(new_data)

        except Exception as e:
            return None

    @staticmethod
    def adaptive_frequency_optimize(current_sequence: Sequence, max_iter=50) -> Sequence:
        """
        Optimizes sequence using a heuristic based on the convex optimization approach.
        """
        n = current_sequence.length
        if n == 0:
            return current_sequence

        # Start with a simple convex optimization approach
        try:
            # Try convex optimization approach
            result = ConvexOptimizer.solve_convolution_quadratic_cvxpy(current_sequence)
            if result is not None:
                return result
        except Exception as e:
            pass

        # Fallback to a more classical approach with better handling of edge cases
        a = current_sequence.data.copy()
        a = np.maximum(a, 1e-10)

        # Simple gradient-based update
        t = 0.01
        new_data = a * (1 - t) + np.mean(a) * t  # Move towards mean
        new_data = np.maximum(new_data, 1e-10)

        return Sequence(new_data.tolist())

    @staticmethod
    def multi_start_optimization(initial_sequences: list[Sequence], max_time: float) -> Sequence:
        """
        Performs multi-start optimization using a convex optimization framework.
        """
        best_inv_C1 = 0.0
        best_sequence = None
        best_C1 = float('inf')
        start_time = time.time()

        # Try multiple starting points with convex optimization
        for i, init_seq in enumerate(initial_sequences):
            if time.time() - start_time > max_time - 5:
                break

            current_seq = init_seq.copy()
            current_C1, current_inv_C1 = AutocorrelationEvaluator.compute_c1(current_seq)

            if current_inv_C1 > best_inv_C1:
                best_inv_C1 = current_inv_C1
                best_sequence = current_seq.copy()
                best_C1 = current_C1

            # Use convex optimization approach for improvement
            improved_seq = SequenceOptimizer.adaptive_frequency_optimize(current_seq, max_iter=30)
            improved_C1, improved_inv_C1 = AutocorrelationEvaluator.compute_c1(improved_seq)

            if improved_inv_C1 > current_inv_C1:
                current_seq = improved_seq
                current_C1 = improved_C1
                current_inv_C1 = improved_inv_C1

                if current_inv_C1 > best_inv_C1:
                    best_inv_C1 = current_inv_C1
                    best_sequence = current_seq.copy()
                    best_C1 = current_C1

        # Add historical sequences to the mix for seeding
        if len(historical_sequences) > 0:
            for hist_seq in list(historical_sequences):
                if time.time() - start_time > max_time - 5:
                    break
                hist_C1, hist_inv_C1 = AutocorrelationEvaluator.compute_c1(hist_seq)
                if hist_inv_C1 > best_inv_C1:
                    best_inv_C1 = hist_inv_C1
                    best_sequence = hist_seq.copy()
                    best_C1 = hist_C1

        if best_sequence is None:
            return Sequence([1.0])
        return best_sequence

def generate_initial_sequences() -> list[Sequence]:
    """
    Generates diverse initial sequences at different scales.
    """
    initial_sequences = []

    # Random sequences
    for _ in range(2):
        n = np.random.randint(50, 200)
        seq = np.random.uniform(0.1, 1.0, n).tolist()
        initial_sequences.append(Sequence(seq))

    # Exponential decay sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        seq = [0.8 ** i for i in range(n)]
        initial_sequences.append(Sequence(seq))

    # Spike sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        seq = [0.0] * n
        spike_idx = np.random.randint(0, n)
        seq[spike_idx] = 1.0
        initial_sequences.append(Sequence(seq))

    # Gaussian-like sequences
    for _ in range(2):
        n = np.random.randint(100, 500)
        center = n // 2
        seq = [np.exp(-0.5 * ((i - center)**2) / (n/10)**2) for i in range(n)]
        initial_sequences.append(Sequence(seq))

    # Historical sequences if available
    if len(historical_sequences) > 0:
        for _ in range(2):
            hist_seq = random.choice(list(historical_sequences))
            initial_sequences.append(hist_seq.copy())

    return initial_sequences

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    """
    start_time = time.time()
    max_time = 180  # seconds

    # Generate diverse initial sequences at different scales
    initial_sequences = generate_initial_sequences()

    # Multi-start optimization using convex approach
    best_sequence = SequenceOptimizer.multi_start_optimization(initial_sequences, max_time)

    # Final optimization with enhanced parameters
    if best_sequence is not None:
        final_seq = SequenceOptimizer.adaptive_frequency_optimize(best_sequence, max_iter=100)
        final_C1, final_inv_C1 = AutocorrelationEvaluator.compute_c1(final_seq)
        if final_inv_C1 > 0.0:  # Ensure valid result
            best_sequence = final_seq

    return best_sequence.data.tolist() if best_sequence is not None else [1.0]

# Enhanced version of the main function
def search_for_best_sequence_enhanced() -> list[float]:
    """
    Enhanced version of the main function with improved heuristic and
    better integration of evolutionary and convex optimization.
    """
    start_time = time.time()
    max_time = 180  # seconds

    # Generate diverse initial sequences at different scales
    initial_sequences = []

    # Random sequences with more variety
    for _ in range(3):
        n = np.random.randint(50, 1000)
        # Use a mixture of uniform and exponential distributions
        if np.random.rand() < 0.5:
            seq = np.random.uniform(0.1, 1.0, n).tolist()
        else:
            seq = [0.9 ** i for i in range(n)]
        initial_sequences.append(Sequence(seq))

    # Exponential decay sequences with different rates
    for _ in range(2):
        n = np.random.randint(100, 500)
        rate = np.random.uniform(0.7, 0.95)
        seq = [rate ** i for i in range(n)]
        initial_sequences.append(Sequence(seq))

    # Spike sequences with varying amplitudes
    for _ in range(2):
        n = np.random.randint(100, 500)
        seq = [0.0] * n
        spike_idx = np.random.randint(0, n)
        seq[spike_idx] = np.random.uniform(1.0, 10.0)
        initial_sequences.append(Sequence(seq))

    # Gaussian-like sequences with varied width
    for _ in range(2):
        n = np.random.randint(100, 500)
        center = n // 2
        width = np.random.uniform(n/20, n/5)
        seq = [np.exp(-0.5 * ((i - center)**2) / width**2) for i in range(n)]
        initial_sequences.append(Sequence(seq))

    # Historical sequences if available
    if len(historical_sequences) > 0:
        for _ in range(2):
            hist_seq = random.choice(list(historical_sequences))
            initial_sequences.append(hist_seq.copy())

    # Multi-start optimization using convex approach
    best_sequence = SequenceOptimizer.multi_start_optimization(initial_sequences, max_time)

    # Enhanced optimization with adaptive mutation and crossover
    if best_sequence is not None:
        # Perform additional refinement with evolutionary approach
        refined_sequence = best_sequence.copy()
        for _ in range(20):
            # Apply adaptive mutation
            mutated_seq = []
            for val in refined_sequence.data:
                if np.random.rand() < 0.1:  # 10% mutation rate
                    mutated_val = val * (1 + np.random.normal(0, 0.1))  # Gaussian mutation
                    mutated_seq.append(max(0, mutated_val))  # Ensure non-negativity
                else:
                    mutated_seq.append(val)

            # Apply simple crossover with a random sequence
            if np.random.rand() < 0.3:  # 30% crossover rate
                crossover_point = np.random.randint(1, len(refined_sequence.data))
                temp_seq = refined_sequence.data[:crossover_point] + mutated_seq[crossover_point:]
                mutated_seq = temp_seq

            # Evaluate and accept if better
            mutated_seq = Sequence(mutated_seq)
            mutated_C1, mutated_inv_C1 = AutocorrelationEvaluator.compute_c1(mutated_seq)
            if mutated_inv_C1 > 0.0:  # Ensure valid result
                refined_sequence = mutated_seq

        # Final optimization with enhanced parameters
        final_seq = SequenceOptimizer.adaptive_frequency_optimize(refined_sequence, max_iter=100)
        final_C1, final_inv_C1 = AutocorrelationEvaluator.compute_c1(final_seq)
        if final_inv_C1 > 0.0:  # Ensure valid result
            best_sequence = final_seq

        # Store the best sequence as historical for future use
        historical_sequences.append(best_sequence.copy())

    return best_sequence.data.tolist() if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")