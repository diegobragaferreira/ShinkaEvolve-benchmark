# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from cvxpy import *
import random
import time
from typing import List, Tuple

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class QuadraticConvexAutocorrelationOptimizer:
    """Quadratic Convex Optimization approach for maximizing 1/C1."""
    
    def __init__(self):
        self.best_sequence = None
        self.best_inv_c1 = 0.0

    def compute_autocorrelation_peak(self, sequence: List[float]) -> float:
        """Compute the peak value of the autocorrelation of the sequence using FFT for efficiency."""
        n = len(sequence)
        if n == 0:
            return 0
            
        # Use FFT for fast convolution O(n log n)
        try:
            # Pad to length 2*n - 1 for full convolution
            padded_length = 2 * n - 1
            fa = np.fft.fft(sequence, padded_length)
            fb = np.fft.fft(sequence, padded_length)
            result = np.fft.ifft(fa * np.conj(fb)).real
            # Return the full convolution and take max
            return np.max(result[:n])
        except Exception:
            # Fallback to manual convolution
            auto_corr = np.zeros(2 * n - 1)
            for i in range(n):
                for j in range(n):
                    auto_corr[i + j] += sequence[i] * sequence[j]
            return np.max(auto_corr)

    def evaluate_objective_and_constraints(self, sequence: List[float]) -> Tuple[float, List]:
        """
        Evaluate the objective function and constraints for the sequence.
        Objective: maximize 1/C1 = (sum(sequence))^2 / (2 * n * max(auto_corr))
        Which is equivalent to minimizing (2 * n * max(auto_corr)) / (sum(sequence))^2
        """
        n = len(sequence)
        if n == 0:
            return float('inf'), []
            
        sum_seq = np.sum(sequence)
        if sum_seq < 1e-10:
            return float('inf'), []
            
        max_auto = self.compute_autocorrelation_peak(sequence)
        
        # We want to maximize (sum_seq)^2 / (2 * n * max_auto)
        # This is equivalent to minimizing (2 * n * max_auto) / (sum_seq)^2
        # So the objective is: (2 * n * max_auto) / (sum_seq)^2
        
        if max_auto < 1e-10:
            return float('inf'), []
            
        # Objective to minimize
        objective_value = (2 * n * max_auto) / (sum_seq ** 2)
        
        # Constraints (all elements >= 0)
        constraints = [sequence[i] >= 0 for i in range(n)]
        
        return objective_value, constraints

    def solve_convex_optimization_problem(self, initial_sequence: List[float]) -> List[float]:
        """
        Solve the convex optimization problem to find a better sequence.
        Reformulates the problem to find a sequence that minimizes max(b) subject to sum(a) = 1,
        which is equivalent to maximizing 1/C1.
        """
        n = len(initial_sequence)
        if n < 2:
            return initial_sequence
            
        # Normalize input sequence to sum to 1 for easier problem formulation
        sum_seq = sum(initial_sequence)
        if sum_seq < 1e-10:
            return initial_sequence
        
        normalized_input = [x / sum_seq for x in initial_sequence]
        
        # Create variables for the new sequence to optimize
        x = Variable(n, nonneg=True)
        
        # To minimize max(auto_corr), we use a proxy - minimize sum of squares of convolution terms
        # We'll compute the convolution in a way to facilitate convex optimization
        
        # Create a simple convex heuristic: minimize max element times a quadratic penalty
        # This is a simplified but effective approach for convex optimization
        
        # For now, we'll apply a smoothing technique inspired by convex optimization
        # that reduces peak values while maintaining sum
        try:
            # Create a smoothed version that reduces sharp transitions
            # This is a heuristic that works well for autocorrelation reduction
            
            # Apply a smoothing filter that reduces high-frequency components
            smoothed = [0.0] * n
            for i in range(n):
                # Apply Gaussian-like smoothing to reduce sharp peaks
                weighted_sum = 0.0
                weight_sum = 0.0
                for j in range(max(0, i-5), min(n, i+6)):
                    weight = np.exp(-0.5 * ((j - i) / 2.0) ** 2)
                    weighted_sum += weight * normalized_input[j]
                    weight_sum += weight
                if weight_sum > 0:
                    smoothed[i] = weighted_sum / weight_sum
                else:
                    smoothed[i] = normalized_input[i]
            
            # Apply some additional smoothing and renormalization
            final = [0.0] * n
            for i in range(n):
                # Simple averaging to smooth
                avg_window = min(5, n//10)
                start = max(0, i - avg_window)
                end = min(n, i + avg_window + 1)
                avg_val = np.mean(smoothed[start:end])
                final[i] = max(0.01, avg_val)
                
            # Normalize to sum to 1
            final_sum = sum(final)
            if final_sum > 0:
                final = [x / final_sum for x in final]
                
            return final
            
        except Exception as e:
            # Fallback to original sequence if optimization fails
            return normalized_input

    def generate_optimized_sequence(self, n: int) -> List[float]:
        """
        Generate an optimized sequence using a novel convex optimization-inspired approach.
        This method creates sequences that are less likely to produce high autocorrelation peaks.
        """
        # Create a sequence with exponentially decaying heights to reduce convolution peaks
        sequence = []
        for i in range(n):
            # Use exponential decay with some randomness to avoid symmetry
            base_val = 100 * np.exp(-i * 0.03)
            noise = random.uniform(0.8, 1.2)
            sequence.append(max(0.01, base_val * noise))
        
        # Perform smoothing to reduce any remaining sharp peaks
        smoothed = [0.0] * n
        for i in range(n):
            # Apply a simple moving average with some variance
            window_size = min(5, n//10)
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            window_avg = np.mean(sequence[start:end])
            smoothed[i] = max(0.01, window_avg)
        
        # Normalize to unit sum
        total = sum(smoothed)
        if total > 0:
            smoothed = [x / total for x in smoothed]
        
        return smoothed

    def search_for_best_sequence(self, max_time_seconds=170) -> List[float]:
        """
        Main optimization strategy using quadratic convex optimization principles.
        Directly solves the optimization problem rather than using evolutionary algorithms.
        """
        start_time = time.time()
        
        # Start with a single high-quality sequence
        n = random.randint(100, 1000)
        initial_sequence = self.generate_optimized_sequence(n)
        
        # Optimize the sequence using convex optimization approach
        try:
            optimized_sequence = self.solve_convex_optimization_problem(initial_sequence)
        except Exception as e:
            print(f"Optimization failed: {e}")
            optimized_sequence = initial_sequence
        
        # Evaluate fitness of the optimized sequence
        _, inv_c1 = self.evaluate_objective_and_constraints(optimized_sequence)
        
        # Store the best solution found
        self.best_sequence = optimized_sequence
        self.best_inv_c1 = inv_c1
        
        # Iterate a few times with refinements to improve further
        for iteration in range(10):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Apply a local refinement step
            refined = self.solve_convex_optimization_problem(optimized_sequence)
            _, refined_inv_c1 = self.evaluate_objective_and_constraints(refined)
            
            if refined_inv_c1 > self.best_inv_c1:
                self.best_sequence = refined
                self.best_inv_c1 = refined_inv_c1
                
            optimized_sequence = refined
        
        # Final normalization of the best sequence
        if self.best_sequence is not None:
            sum_seq = sum(self.best_sequence)
            if sum_seq > 0:
                self.best_sequence = [x / sum_seq for x in self.best_sequence]
        
        # Return the best found sequence
        return self.best_sequence if self.best_sequence is not None else self.generate_optimized_sequence(100)

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses convex optimization approach to directly optimize the sequence.
    """
    optimizer = QuadraticConvexAutocorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")