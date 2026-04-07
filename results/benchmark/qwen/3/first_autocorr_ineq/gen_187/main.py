# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationOptimizer:
    """An optimized class for finding step function sequences that maximize 1/C1."""
    
    def __init__(self, max_time_seconds: int = 180):
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        
    def convolve_fft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Efficient FFT-based convolution for large sequences."""
        return fftconvolve(a, b, mode='full')[:2*len(a)-1]
    
    def compute_c1_constant(self, sequence: List[float]) -> Tuple[float, float]:
        """Compute C1 constant and 1/C1 value for a given sequence."""
        a = np.array(sequence)
        n = len(a)

        # Use FFT for efficiency when sequence is large
        if n > 100:
            b = self.convolve_fft(a, a)
        else:
            b = np.convolve(a, a, mode='full')[:2*n-1]

        max_conv = np.max(b)
        sum_a = np.sum(a)

        if sum_a < 0.01:
            return float('inf'), 0.0

        c1 = 2 * n * max_conv / (sum_a ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        return c1, inv_c1

    def solve_convolution_lp(self, f_sequence: List[float], rhs: float) -> Optional[List[float]]:
        """Solves the convolution LP for a given sequence and RHS with multiple fallback methods."""
        try:
            n = len(f_sequence)
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

            # Solve the linear program with multiple methods as fallback
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'presolve': True, 'maxiter': 1000})

            if not result.success:
                # Try different method if highs fails
                try:
                    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex', options={'maxiter': 1000})
                except:
                    pass

            if result.success:
                return result.x.tolist()
            else:
                return None

        except Exception as e:
            # Fallback to simpler method
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point', options={'maxiter': 500})
                if result.success:
                    return result.x.tolist()
            except:
                pass
            return None

    def get_good_direction_to_move_into(
        self, sequence: List[float], historical_sequences: List[List[float]] = []
    ) -> Optional[List[float]]:
        """Improve the sequence using evolutionary strategy and LP optimization."""
        n = len(sequence)

        # Normalize the sequence
        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

        # Compute convolution constraints
        if n > 100:
            b = self.convolve_fft(np.array(normalized_sequence), np.array(normalized_sequence))
        else:
            b = np.convolve(np.array(normalized_sequence), np.array(normalized_sequence), mode='full')[:2*n-1]

        rhs = np.max(b)

        # Try multiple times to solve LP
        g_fun = None
        for _ in range(10):
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
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

        normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

        # Apply adaptive perturbation based on convergence rate and current C1
        current_c1, _ = self.compute_c1_constant(sequence)
        t = min(0.1, max(0.01, 0.05 * (1.0 - min(1.0, current_c1 / 1.5))))

        # Inject diversity from historical sequences if available
        if historical_sequences:
            # Sample a historical sequence
            hist_seq = random.choice(historical_sequences)
            # Ensure same length, pad or truncate
            if len(hist_seq) < n:
                hist_seq = hist_seq + [0.0] * (n - len(hist_seq))
            elif len(hist_seq) > n:
                hist_seq = hist_seq[:n]

            # Mix with current direction
            mix_ratio = 0.1
            for i in range(n):
                normalized_g_fun[i] = (1 - mix_ratio) * normalized_g_fun[i] + mix_ratio * hist_seq[i]

        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity and reasonable bounds
        new_sequence = [max(0, min(1000, x)) for x in new_sequence]

        return new_sequence

    def mutate_sequence(self, sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Apply random mutation to a sequence with variance scaling."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Use larger variance for better exploration
                mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.5 * mutated[i]))
        return mutated

    def crossover_sequences(self, seq1: List[float], seq2: List[float]) -> List[float]:
        """Perform uniform crossover between two sequences."""
        # Ensure both sequences are of same length
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

    def adaptive_search_for_best_sequence(self) -> List[float]:
        """Main function to search for the best coefficient sequence using improved evolutionary approach."""
        self.start_time = time.time()
        initial_length_range = (100, 500)
        population_size = 20
        generations = 50
        elite_fraction = 0.2
        patience_limit = 5

        # Initialize population with diverse sequences
        population = []
        historical_sequences = []  # Store good sequences from previous runs
        for _ in range(population_size):
            n = random.randint(*initial_length_range)
            # Use different initialization strategies for diversity
            init_type = random.choice(['uniform', 'gaussian', 'sparse'])
            if init_type == 'uniform':
                individual = [random.uniform(0.1, 1.0) for _ in range(n)]
            elif init_type == 'gaussian':
                individual = [max(0, random.gauss(0.5, 0.2)) for _ in range(n)]
            else:  # sparse
                individual = [random.uniform(0.1, 1.0) if random.random() < 0.3 else 0.0 for _ in range(n)]
            population.append(individual)

        # Pre-initialize with historical data if available
        if historical_sequences:
            for i in range(min(len(historical_sequences), population_size // 4)):
                # Replace some random individuals with historical sequences
                idx = random.randint(0, population_size - 1)
                # Ensure same length
                if len(historical_sequences[i]) < len(population[idx]):
                    population[idx] = historical_sequences[i] + [0.0] * (len(population[idx]) - len(historical_sequences[i]))
                elif len(historical_sequences[i]) > len(population[idx]):
                    population[idx] = historical_sequences[i][:len(population[idx])]
                else:
                    population[idx] = historical_sequences[i].copy()

        best_sequence = None
        best_inv_c1 = 0.0
        patience_counter = 0
        last_improvement = 0

        for generation in range(generations):
            if time.time() - self.start_time > self.max_time_seconds:
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
                patience_counter = 0  # Reset patience counter
                last_improvement = generation
                # Save good sequences for diversity injection
                historical_sequences.append(current_best.copy())
                # Keep only top historical sequences
                historical_sequences = sorted(historical_sequences, key=self.compute_c1_constant, reverse=True)[:10]
            else:
                patience_counter += 1

            # Early stopping if no improvement for too long
            if patience_counter > patience_limit:
                break

            # Check if we're stagnating - stop early if so
            if generation - last_improvement > patience_limit * 2:
                break

            # Select elite individuals
            elite_count = int(elite_fraction * population_size)
            elite_individuals = [ind for ind, _ in fitness_scores[:elite_count]]

            # Generate new population
            new_population = elite_individuals.copy()

            # Create offspring through crossover and mutation
            while len(new_population) < population_size:
                parent1 = random.choice(elite_individuals)
                parent2 = random.choice(elite_individuals)

                # Crossover with history injection
                child = self.crossover_sequences(parent1, parent2)

                # Inject historical knowledge into child
                if historical_sequences and random.random() < 0.3:
                    # Average with a random historical sequence
                    hist_seq = random.choice(historical_sequences)
                    avg_child = []
                    for i in range(len(child)):
                        if i < len(hist_seq):
                            avg_child.append((child[i] + hist_seq[i]) / 2)
                        else:
                            avg_child.append(child[i])
                    child = avg_child

                # Mutation with historical influence
                child = self.mutate_sequence(child)

                # Occasionally start fresh with historical good sequences
                if random.random() < 0.1:
                    n = random.randint(*initial_length_range)
                    # Use historical sequence if available
                    if historical_sequences:
                        # Pick a good historical sequence and perturb it
                        hist_seq = random.choice(historical_sequences)
                        # Trim or extend to desired length
                        if len(hist_seq) < n:
                            child = hist_seq + [0.0] * (n - len(hist_seq))
                        elif len(hist_seq) > n:
                            child = hist_seq[:n]
                        else:
                            child = hist_seq.copy()

                        # Apply some mutation to avoid local minimum
                        child = self.mutate_sequence(child, 0.2)
                    else:
                        # Fallback to standard initialization if no history
                        init_type = random.choice(['uniform', 'gaussian', 'sparse'])
                        if init_type == 'uniform':
                            child = [random.uniform(0.1, 1.0) for _ in range(n)]
                        elif init_type == 'gaussian':
                            child = [max(0, random.gauss(0.5, 0.2)) for _ in range(n)]
                        else:  # sparse
                            child = [random.uniform(0.1, 1.0) if random.random() < 0.3 else 0.0 for _ in range(n)]

                new_population.append(child)

            population = new_population

        # Final optimization of best solution
        if best_sequence is not None:
            prev_inv_c1 = 0.0
            for _ in range(20):  # Additional fine-tuning iterations
                improved = self.get_good_direction_to_move_into(best_sequence, historical_sequences=historical_sequences)
                if improved is None:
                    break
                _, inv_c1_new = self.compute_c1_constant(improved)
                _, inv_c1_old = self.compute_c1_constant(best_sequence)
                if inv_c1_new > inv_c1_old:
                    best_sequence = improved
                    prev_inv_c1 = inv_c1_new
                else:
                    break

        return best_sequence if best_sequence is not None else [1.0]

    def search_for_best_sequence(self) -> List[float]:
        """Function to search for the best coefficient sequence."""
        return self.adaptive_search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")