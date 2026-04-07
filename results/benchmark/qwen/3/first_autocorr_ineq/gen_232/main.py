# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional
import warnings
from numba import jit

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)

@jit(nopython=True)
def fast_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast convolution using Numba JIT compilation for small to medium sequences."""
    n = len(a)
    m = len(b)
    result = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
    return result

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    # Dynamic threshold based on sequence characteristics
    n = len(a)
    m = len(b)
    product = n * m

    # Use FFT if either sequence is large OR if the product indicates high computational cost
    if n > 100 or m > 100 or product > 10000:
        return convolve_fft(a, b)
    else:
        return fast_convolution(a, b)

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

class ConvolutionOptimizer:
    """Handles the core optimization logic for solving convolution constraints."""

    def __init__(self):
        self._last_solution = None

    def solve_convolution_lp(self, f_sequence: List[float], rhs: float, n: int) -> Optional[List[float]]:
        """Solves the convolution LP for a given sequence and RHS with robustness."""
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

            # Solve the linear program with multiple fallbacks
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs',
                                    options={'maxiter': 1000, 'presolve': True})

            if not result.success:
                # Try revised simplex if highs fails
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex',
                                        options={'maxiter': 1000})

            if result.success:
                self._last_solution = result.x.tolist()
                return self._last_solution
            else:
                return None

        except Exception:
            return None

class GradientRefiner:
    """Refines sequences using gradient ascent and curvature awareness."""

    def __init__(self, optimizer: ConvolutionOptimizer):
        self.optimizer = optimizer

    def refine_sequence(self, sequence: List[float]) -> Optional[List[float]]:
        """Improve the sequence using LP optimization with curvature awareness."""
        n = len(sequence)

        # Normalize the sequence
        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        # Use adaptive normalization scaling factor
        adaptive_scale = np.sqrt(2 * n) / max(1.0, sum_sequence)
        normalized_sequence = [x * adaptive_scale for x in sequence]

        # Compute convolution constraints
        if n > 100:
            b = convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
        else:
            b = convolve_direct(np.array(normalized_sequence), np.array(normalized_sequence))

        rhs = np.max(b)

        # Try solving LP multiple times
        g_fun = None
        max_attempts = 5
        for attempt in range(max_attempts):
            g_fun = self.optimizer.solve_convolution_lp(normalized_sequence, rhs, n)
            if g_fun is not None:
                break
            rhs *= 1.01  # Slight increase in constraint if failed

        if g_fun is None:
            return None

        # Normalize the solution from LP
        sum_g = np.sum(g_fun)
        if sum_g < 0.01:
            return None

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

        # Apply curvature-aware directional bias correction
        if n > 10:
            # Create small perturbations to estimate Hessian with improved numerical stability
            # Use adaptive epsilon based on sequence values to prevent numerical issues
            base_epsilon = 1e-3
            epsilon = base_epsilon * max(1.0, np.mean(np.abs(normalized_sequence)))
            hessian_approx = np.zeros((n, n))

            # Improved finite difference approximation with better numerical handling
            for i in range(n):
                perturbed_seq = normalized_sequence.copy()
                perturbed_seq[i] += epsilon
                # Apply boundary corrections to avoid index errors with safer adjustment
                if i > 0:
                    perturbed_seq[i-1] -= epsilon / 2
                if i < n-1:
                    perturbed_seq[i+1] -= epsilon / 2

                # Recompute convolution with perturbed sequence using appropriate method
                try:
                    if n > 100:
                        b_perturbed = convolve_fft(np.array(perturbed_seq), np.array(perturbed_seq))
                    else:
                        b_perturbed = convolve_direct(np.array(perturbed_seq), np.array(perturbed_seq))

                    # Ensure we have valid convolution results
                    if len(b_perturbed) > 0:
                        second_derivative = (np.max(b_perturbed) - np.max(b)) / (epsilon**2)
                        hessian_approx[i, i] = max(0, second_derivative)  # Ensure non-negative
                    else:
                        hessian_approx[i, i] = 0.0
                except Exception:
                    # Fallback if computation fails
                    hessian_approx[i, i] = 0.0

            # Apply curvature correction with better stability checks
            curvature_correction = np.dot(hessian_approx, normalized_g_fun)

            # Normalize curvature correction to prevent blowup
            norm_correction = np.linalg.norm(curvature_correction)
            if norm_correction > 0:
                curvature_correction = curvature_correction / (1.0 + norm_correction)
            else:
                curvature_correction = np.zeros_like(curvature_correction)

            # Adjust the direction with curvature bias, with bounded adjustment
            correction_magnitude = 0.1 * min(1.0, norm_correction / (1.0 + np.mean(np.abs(normalized_g_fun))))
            corrected_direction = np.array(normalized_g_fun) + correction_magnitude * curvature_correction
            normalized_g_fun = corrected_direction.tolist()

        # Adaptive step size
        current_c1, _ = compute_c1_constant(sequence)
        t = min(0.1, max(0.01, 0.05 * (1.0 - min(1.0, current_c1 / 1.5))))
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity and reasonable bounds
        new_sequence = [max(0, min(1000, x)) for x in new_sequence]

        return new_sequence

class EvolutionaryStrategy:
    """Manages the evolutionary search process with adaptive parameters."""

    def __init__(self, refiner: GradientRefiner):
        self.refiner = refiner
        # Optimized evolutionary parameters for better convergence
        self.population_size = 100
        self.generations = 120
        self.elite_fraction = 0.3
        self.patience_limit = 20
        self.mutation_rate_start = 0.15
        self.mutation_rate_end = 0.02
        self.initial_length_range = (120, 400)  # Narrower range for more focused search

    def initialize_population(self) -> List[List[float]]:
        """Generate initial population with diverse initialization strategies."""
        population = []
        for _ in range(self.population_size):
            n = random.randint(*self.initial_length_range)
            init_type = random.choice(['uniform', 'gaussian', 'sparse', 'historical'])
            if init_type == 'uniform':
                individual = [random.uniform(0.1, 1.0) for _ in range(n)]
            elif init_type == 'gaussian':
                individual = [max(0, random.gauss(0.5, 0.2)) for _ in range(n)]
            elif init_type == 'sparse':
                individual = [random.uniform(0.1, 1.0) if random.random() < 0.3 else 0.0 for _ in range(n)]
            else:  # historical
                individual = self._initialize_historical(n)
            population.append(individual)
        return population

    def _initialize_historical(self, length: int) -> List[float]:
        """Initialize sequence with historical knowledge."""
        sequence = []
        peak_position = length // 2
        peak_height = 1.0

        for i in range(length):
            distance_from_peak = abs(i - peak_position)
            decay_factor = np.exp(-distance_from_peak / (length / 4))
            sequence.append(peak_height * decay_factor)

        total_sum = sum(sequence)
        if total_sum > 0:
            sequence = [x / total_sum for x in sequence]

        return [max(0.01, x) for x in sequence]

    def evaluate_fitness(self, population: List[List[float]]) -> List[Tuple[List[float], float]]:
        """Evaluate the fitness for all individuals in the population."""
        fitness_scores = []
        for individual in population:
            _, inv_c1 = compute_c1_constant(individual)
            fitness_scores.append((individual, inv_c1))
        return fitness_scores

    def select_elite(self, fitness_scores: List[Tuple[List[float], float]]) -> List[List[float]]:
        """Select the elite individuals based on fitness."""
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        elite_count = int(self.elite_fraction * self.population_size)
        elite_individuals = [ind for ind, _ in fitness_scores[:elite_count]]
        return elite_individuals

    def crossover_and_mutate(self, elite_individuals: List[List[float]], generation: int) -> List[List[float]]:
        """Create new individuals through crossover and mutation."""
        new_population = elite_individuals.copy()
        current_mutation_rate = self.mutation_rate_start - (self.mutation_rate_start - self.mutation_rate_end) * (generation / self.generations)

        while len(new_population) < self.population_size:
            parent1 = random.choice(elite_individuals)
            parent2 = random.choice(elite_individuals)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child, current_mutation_rate)

            # Random initialization for diversity
            if random.random() < 0.2:
                n = random.randint(*self.initial_length_range)
                init_type = random.choice(['uniform', 'gaussian', 'sparse', 'historical'])
                if init_type == 'uniform':
                    child = [random.uniform(0.1, 1.0) for _ in range(n)]
                elif init_type == 'gaussian':
                    child = [max(0, random.gauss(0.5, 0.2)) for _ in range(n)]
                elif init_type == 'sparse':
                    child = [random.uniform(0.1, 1.0) if random.random() < 0.3 else 0.0 for _ in range(n)]
                else:  # historical
                    child = self._initialize_historical(n)

            new_population.append(child)
        return new_population[:self.population_size]

    def _crossover(self, seq1: List[float], seq2: List[float]) -> List[float]:
        """Perform uniform crossover between two sequences."""
        min_len = min(len(seq1), len(seq2))
        child = []
        for i in range(min_len):
            if random.random() < 0.5:
                child.append(seq1[i])
            else:
                child.append(seq2[i])

        # Handle length differences
        if len(seq1) > len(seq2):
            child.extend(seq1[min_len:])
        elif len(seq2) > len(seq1):
            child.extend(seq2[min_len:])
        return child

    def _mutate(self, sequence: List[float], mutation_rate: float) -> List[float]:
        """Apply random mutation to a sequence."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutation_scale = 0.3 * mutated[i]
                mutated[i] = max(0, mutated[i] + np.random.normal(0, mutation_scale))
        return mutated

    def run_evolution(self) -> List[float]:
        """Run the evolutionary search process."""
        start_time = time.time()
        population = self.initialize_population()
        best_sequence = None
        best_inv_c1 = 0.0
        patience_counter = 0
        last_improvement = 0

        for generation in range(self.generations):
            if time.time() - start_time > 180:
                break

            # Evaluate fitness for all individuals
            fitness_scores = self.evaluate_fitness(population)

            # Update best solution
            current_best, current_best_inv_c1 = max(fitness_scores, key=lambda x: x[1])
            if current_best_inv_c1 > best_inv_c1:
                best_inv_c1 = current_best_inv_c1
                best_sequence = current_best.copy()
                patience_counter = 0
                last_improvement = generation
            else:
                patience_counter += 1

            # Early stopping conditions
            if patience_counter > self.patience_limit:
                break
            if generation - last_improvement > self.patience_limit * 2:
                break

            # Select elite individuals
            elite_individuals = self.select_elite(fitness_scores)

            # Generate new population
            population = self.crossover_and_mutate(elite_individuals, generation)

        # Final refinement of best solution
        if best_sequence is not None:
            prev_inv_c1 = 0.0
            for _ in range(20):  # More fine-tuning iterations
                improved = self.refiner.refine_sequence(best_sequence)
                if improved is None:
                    break
                _, inv_c1_new = compute_c1_constant(improved)
                _, inv_c1_old = compute_c1_constant(best_sequence)
                if inv_c1_new > inv_c1_old:
                    best_sequence = improved
                    prev_inv_c1 = inv_c1_new
                else:
                    break

        return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Main function to find the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    optimizer = ConvolutionOptimizer()
    refiner = GradientRefiner(optimizer)
    strategy = EvolutionaryStrategy(refiner)

    return strategy.run_evolution()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")