# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from typing import List, Optional, Tuple
import numba
from numba import jit
import time

class AutocorrelationGradientEvolution:
    """
    An optimizer for finding step functions that maximize 1/C₁,
    thereby improving the upper bound on the first autocorrelation inequality constant.
    This version uses a hybrid approach combining evolutionary search with gradient-based refinement.
    """

    def __init__(self, max_sequence_length: int = 1000, min_sequence_length: int = 100):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.diversity_threshold = 0.1  # Threshold to trigger diversity maintenance
        self.max_diversity_stagnation = 20  # Max stagnation before diversity reset
        self.refinement_steps = 10  # Number of refinement steps after evolution
        self.seed = 42  # For deterministic results

    @staticmethod
    @jit(nopython=True)
    def fast_convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Fast convolution using Numba JIT compilation."""
        n1, n2 = len(arr1), len(arr2)
        result = np.zeros(n1 + n2 - 1)
        for i in range(n1):
            for j in range(n2):
                result[i + j] += arr1[i] * arr2[j]
        return result

    def compute_convolution(self, sequence: np.ndarray) -> np.ndarray:
        """Compute the autoconvolution of the sequence efficiently."""
        n = len(sequence)
        if n > 1000:  # For large sequences, use FFT
            padded_seq = np.pad(sequence, (0, n - 1), 'constant')
            # Using np.fft for convolution
            conv_result = np.fft.irfft(np.fft.rfft(padded_seq) * np.fft.rfft(sequence))
            return conv_result[:2*n - 1]
        else:
            # For smaller sequences, use direct computation for precision
            return self.fast_convolve(sequence, sequence)

    def compute_c1(self, sequence: np.ndarray) -> float:
        """Compute C₁ constant for the given sequence."""
        n = len(sequence)
        if n == 0:
            return float('inf')
        
        # Compute convolution
        conv = self.compute_convolution(sequence)
        
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

    def compute_gradient(self, sequence: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute approximate gradient of 1/C₁ using finite differences."""
        n = len(sequence)
        grad = np.zeros(n)
        base_inv_c1 = self.compute_inv_c1(sequence)
        
        for i in range(n):
            perturbed = sequence.copy()
            perturbed[i] += epsilon
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negative
            perturbed_inv_c1 = self.compute_inv_c1(perturbed)
            grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon
            
        return grad

    def gradient_refinement(self, sequence: np.ndarray, steps: int = 10, lr: float = 0.01) -> np.ndarray:
        """Perform gradient-based refinement on the sequence."""
        current = np.array(sequence, dtype=float)
        for _ in range(steps):
            grad = self.compute_gradient(current)
            # Adjust step size based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                step_size = lr / grad_norm
                current += step_size * grad
                current = np.maximum(current, 0)  # Keep non-negative
        return current.tolist()

    def get_diversity_score(self, population: List[List[float]]) -> float:
        """Calculate diversity score among population members."""
        if len(population) < 2:
            return 0.0
        avg_dist = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                avg_dist += dist
                count += 1
        return avg_dist / count if count > 0 else 0.0

    def generate_initial_sequence(self, n: int = None) -> List[float]:
        """Generate a good initial sequence."""
        if n is None:
            n = np.random.randint(self.min_sequence_length, self.max_sequence_length)
        # Generate a sequence with some structure to avoid poor starts
        sequence = np.random.exponential(scale=1.0, size=n).tolist()
        # Normalize to have reasonable total weight
        total = sum(sequence)
        if total > 0:
            sequence = [x / total * 100 for x in sequence]
        return sequence

    def run_evolution(self, max_iterations: int = 100) -> Tuple[List[float], float]:
        """Run the hybrid optimization process."""
        np.random.seed(self.seed)
        best_sequence = self.generate_initial_sequence()
        best_inv_c1 = self.compute_inv_c1(best_sequence)
        
        print(f'Initial inv_C1: {best_inv_c1:.6f}')
        
        # Population for diversity management
        population = [best_sequence]
        diversity_stagnation = 0
        
        # Evolution loop
        for iteration in range(max_iterations):
            # Generate new candidate via mutation
            try:
                # Create a mutated version
                mutated_seq = self.generate_initial_sequence()
                
                # Occasionally do gradient-based refinement on existing best
                if iteration % 5 == 0 and len(population) > 0:
                    representative = population[-1]  # Take last one as representative
                    refined = self.gradient_refinement(representative, self.refinement_steps, 0.01)
                    mutated_seq = refined
                
                candidate_inv_c1 = self.compute_inv_c1(mutated_seq)
                
                # Update best
                if candidate_inv_c1 > best_inv_c1:
                    best_sequence = mutated_seq
                    best_inv_c1 = candidate_inv_c1
                    print(f'Iteration {iteration}: New best inv_C1: {best_inv_c1:.6f}')
                    
                    # Early exit if we beat the benchmark
                    if best_inv_c1 > 1.0 / 1.5031:
                        print(f'Beaten benchmark at iteration {iteration}')
                        break
                        
                # Update population and diversity
                population.append(mutated_seq)
                if len(population) > 20:
                    population.pop(0)  # Keep only recent members
                    
                # Check diversity
                diversity = self.get_diversity_score(population)
                if diversity < self.diversity_threshold:
                    diversity_stagnation += 1
                    if diversity_stagnation > self.max_diversity_stagnation:
                        # Reset diversity if stagnation detected
                        population = [best_sequence]
                        diversity_stagnation = 0
                        print("Diversity reset triggered")
                else:
                    diversity_stagnation = 0
                    
            except Exception as e:
                print(f'Error in evolution iteration {iteration}: {e}')
                # Perturb slightly
                if best_sequence:
                    index = np.random.randint(len(best_sequence))
                    best_sequence[index] = max(0, best_sequence[index] + np.random.normal(0, 0.5))

        # Final refinement
        final_sequence = self.gradient_refinement(best_sequence, self.refinement_steps, 0.01)
        final_inv_c1 = self.compute_inv_c1(final_sequence)
        
        if final_inv_c1 > best_inv_c1:
            best_sequence = final_sequence
            best_inv_c1 = final_inv_c1
            
        return best_sequence, best_inv_c1

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = AutocorrelationGradientEvolution()
    best_sequence, _ = optimizer.run_evolution()
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
