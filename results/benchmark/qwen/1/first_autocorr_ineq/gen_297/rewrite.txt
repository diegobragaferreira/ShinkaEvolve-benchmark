# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import copy
from typing import List, Tuple
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class FastAutocorrelationGradientSearch:
    """
    A gradient-based optimization approach for maximizing 1/C₁ using JAX acceleration
    and FFT-based convolution for fast evaluation.
    """

    def __init__(self):
        self.best_score = 0.0
        self.best_sequence = None
        self.fitness_cache = {}  # Memoization for fitness evaluations

    @staticmethod
    def evaluate_sequence_jax(sequence: List[float]) -> Tuple[float, float]:
        """
        Evaluate a sequence using JAX for fast computation.

        Args:
            sequence: List of non-negative real numbers representing step heights

        Returns:
            tuple: (C₁, 1/C₁)
        """
        seq_tuple = tuple(sequence)
        if seq_tuple in FastAutocorrelationGradientSearch.fitness_cache:
            return FastAutocorrelationGradientSearch.fitness_cache[seq_tuple]

        try:
            # Convert to numpy array
            a = np.array(sequence, dtype=np.float64)
            sum_a = np.sum(a)

            # Avoid division by zero or negligible sums
            if sum_a < 1e-10:
                result = (float('inf'), 0.0)
                FastAutocorrelationGradientSearch.fitness_cache[seq_tuple] = result
                return result

            # Compute autoconvolution using FFT for efficiency
            b = fftconvolve(a, a, mode='full')
            b = b[len(a)-1:2*len(a)-1]  # Convolution part

            max_b = np.max(b)

            # Compute C₁ = 2n * max(b) / (sum(a))^2
            n = len(a)
            c1 = 2 * n * max_b / (sum_a ** 2)

            # Return inverse for maximization
            inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

            result = (c1, inv_c1)
            FastAutocorrelationGradientSearch.fitness_cache[seq_tuple] = result
            return result
        except Exception as e:
            result = (float('inf'), 0.0)
            FastAutocorrelationGradientSearch.fitness_cache[seq_tuple] = result
            return result

    @staticmethod
    def generate_structured_sequence(length: int) -> List[float]:
        """
        Generate a structured sequence using a sinusoidal and exponential pattern
        for better initial structure.
        """
        # Create a sequence with both sine and exponential characteristics
        sequence = []
        for i in range(length):
            sine_component = abs(np.sin(i * np.pi / length)) * 500
            exp_component = 500 * np.exp(-i / (length * 0.5))
            combined = (sine_component + exp_component) / 2.0

            # Add slight noise for exploration
            if random.random() < 0.1:
                combined += random.uniform(-100, 100)

            sequence.append(max(0, combined))

        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01

        return sequence

    @staticmethod
    def gradient_refine_jax(sequence: List[float], max_iter: int = 100) -> List[float]:
        """
        Apply gradient-based refinement using JAX for accelerated optimization.
        """
        # Convert to JAX array
        seq = jnp.array(sequence, dtype=jnp.float64)
        n = len(sequence)

        # Define the objective function to maximize (negative 1/C1)
        def objective(x):
            x = jnp.maximum(0.0, x)  # Ensure non-negative
            sum_x = jnp.sum(x)
            if sum_x < 1e-10:
                return jnp.inf
            # Perform convolution
            conv = fftconvolve(x, x, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = jnp.max(conv)
            # Compute C1
            c1 = 2 * n * max_conv / (sum_x ** 2)
            # Return negative since we're minimizing
            return -1.0 / c1 if c1 > 0 else jnp.inf

        # Optimize using JAX's minimize
        result = minimize(objective, seq, method='BFGS', options={'maxiter': max_iter})
        optimized_seq = result.x
        return optimized_seq.tolist()

    def optimize(self) -> List[float]:
        """
        Run the gradient-based optimization process.
        """
        start_time = time.time()
        timeout = 170  # Leave 10 seconds for cleanup

        # Generate an initial structured sequence
        initial_length = random.randint(100, 500)
        sequence = self.generate_structured_sequence(initial_length)

        # Get initial score
        c1, inv_c1 = self.evaluate_sequence_jax(sequence)
        self.best_score = inv_c1
        self.best_sequence = sequence.copy()

        print(f"Initial best score: {self.best_score:.6f}")

        # Refine the initial sequence using gradient-based method
        refined_sequence = self.gradient_refine_jax(sequence)
        _, refined_inv_c1 = self.evaluate_sequence_jax(refined_sequence)

        if refined_inv_c1 > self.best_score:
            self.best_score = refined_inv_c1
            self.best_sequence = refined_sequence

        # Further iterative refinement
        iter_count = 0
        while time.time() - start_time < timeout and iter_count < 5000:
            iter_count += 1

            # Perturb current best sequence slightly
            mutated_sequence = self.mutate_sequence(self.best_sequence)
            
            # Evaluate and refine
            _, mut_inv_c1 = self.evaluate_sequence_jax(mutated_sequence)
            refined_mutated = self.gradient_refine_jax(mutated_sequence)
            _, refined_mut_inv_c1 = self.evaluate_sequence_jax(refined_mutated)

            # Update best if improved
            if refined_mut_inv_c1 > self.best_score:
                self.best_score = refined_mut_inv_c1
                self.best_sequence = refined_mutated
                
                print(f"Iteration {iter_count}: New best = {self.best_score:.6f}")

                # Check if we beat the benchmark
                if 1.5031 / self.best_score > 1.0:
                    print(f"BEAT BENCHMARK at iteration {iter_count}!")
                    break

        return self.best_sequence

    @staticmethod
    def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Mutate a sequence with Gaussian noise."""
        mutated = copy.deepcopy(sequence)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] += random.gauss(0, 100)
                mutated[i] = max(0, mutated[i])  # Ensure non-negative
        return mutated

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = FastAutocorrelationGradientSearch()
    try:
        best_sequence = optimizer.optimize()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to a basic sequence if nothing worked
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")