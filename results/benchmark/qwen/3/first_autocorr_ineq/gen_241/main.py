# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random
import time
from typing import List, Tuple, Optional, Union

class AutocorrelationOptimizer:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.curvature_history = []
        self.hessian_approx = None

    def compute_convolution(self, sequence: List[float], use_fft: bool = True) -> np.ndarray:
        """Computes convolution of sequence with itself."""
        a = np.array(sequence)
        if use_fft and len(sequence) > 100:
            return fftconvolve(a, a, mode='full')[:2*len(sequence)-1]
        else:
            return np.convolve(a, a, mode='full')

    def compute_c1_constant(self, sequence: List[float]) -> Tuple[float, float]:
        """Computes C1 constant and 1/C1 value for a given sequence."""
        a = np.array(sequence)
        n = len(a)
        
        conv_result = self.compute_convolution(sequence)
        max_conv = np.max(conv_result)
        sum_a = np.sum(a)

        if sum_a < 0.01:
            return float('inf'), 0.0

        c1 = 2 * n * max_conv / (sum_a ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        return c1, inv_c1

    def solve_convolution_lp(self, f_sequence: List[float], rhs: float, n: int) -> Optional[List[float]]:
        """Solves the convolution LP for a given sequence and RHS."""
        try:
            c = -np.ones(n)
            a_ub = []
            b_ub = []

            # Build constraint matrix for convolution constraints
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

            # Solve the linear program
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                return result.x.tolist()
            else:
                return None

        except Exception as e:
            return None

    def get_good_direction_to_move_into(self, sequence: List[float], 
                                       max_iterations: int = 10) -> Optional[List[float]]:
        """Improve the sequence using LP optimization."""
        n = len(sequence)
        sum_sequence = np.sum(sequence)
        
        # Prevent division by zero
        if sum_sequence < 0.01:
            return None

        # Normalize with adaptive factor
        adaptive_factor = np.sqrt(2 * n)
        normalized_sequence = [x * adaptive_factor / sum_sequence for x in sequence]

        # Compute convolution constraints
        conv_result = self.compute_convolution(normalized_sequence)
        rhs = np.max(conv_result)

        # Try multiple times to solve LP
        for _ in range(max_iterations):
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs, n)
            if g_fun is not None:
                break
            else:
                # If LP fails, slightly modify constraints and retry
                rhs *= 1.01

        if g_fun is None:
            return None

        # Normalize the solution from LP
        sum_g = np.sum(g_fun)
        if sum_g < 0.01:
            return None

        normalized_g_fun = [x * adaptive_factor / sum_g for x in g_fun]

        # Apply small perturbation for exploration
        t = 0.05
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity and reasonable bounds
        new_sequence = [max(0, min(1000, x)) for x in new_sequence]

        return new_sequence

    def mutate_sequence(self, sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Apply random mutation to a sequence."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.1 * mutated[i]))
        return mutated

    def crossover_sequences(self, seq1: List[float], seq2: List[float]) -> List[float]:
        """Perform uniform crossover between two sequences."""
        child = []
        for i in range(min(len(seq1), len(seq2))):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])
        return child

    def search_for_best_sequence(self, max_time_seconds: int = 180) -> List[float]:
        """Main function to search for the best coefficient sequence using evolutionary approach."""
        start_time = time.time()
        
        # Initialize population with diverse sequences
        population = []
        for _ in range(20):
            n = random.randint(100, 500)
            individual = [random.random() * 100 for _ in range(n)]
            population.append(individual)

        best_sequence = None
        best_inv_c1 = 0.0

        generations = 50
        for generation in range(generations):
            if time.time() - start_time > max_time_seconds:
                break

            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                _, inv_c1 = self.compute_c1_constant(individual)
                fitness_scores.append((individual, inv_c1))

            # Sort by fitness (descending order)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Update best solution
            current_best, current_best_inv_c1 = fitness_scores[0]
            if current_best_inv_c1 > best_inv_c1:
                best_inv_c1 = current_best_inv_c1
                best_sequence = current_best.copy()

            # Select elite individuals
            elite_count = int(0.2 * len(population))
            elite_individuals = [ind for ind, _ in fitness_scores[:elite_count]]

            # Generate new population
            new_population = elite_individuals.copy()

            # Create offspring through crossover and mutation
            while len(new_population) < len(population):
                parent1 = random.choice(elite_individuals)
                parent2 = random.choice(elite_individuals)

                # Crossover
                child = self.crossover_sequences(parent1, parent2)

                # Mutation
                child = self.mutate_sequence(child)

                # Random initialization for diversity
                if random.random() < 0.1:
                    n = random.randint(100, 500)
                    child = [random.random() * 100 for _ in range(n)]

                new_population.append(child)

            population = new_population

        # Final optimization of best solution
        if best_sequence is not None:
            for _ in range(20):  # Additional fine-tuning iterations
                improved = self.get_good_direction_to_move_into(best_sequence)
                if improved is None:
                    break
                _, inv_c1_new = self.compute_c1_constant(improved)
                _, inv_c1_old = self.compute_c1_constant(best_sequence)
                if inv_c1_new > inv_c1_old:
                    best_sequence = improved
                else:
                    break

        return best_sequence if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")