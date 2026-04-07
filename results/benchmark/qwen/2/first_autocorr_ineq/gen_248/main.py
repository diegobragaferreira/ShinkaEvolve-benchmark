# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import warnings
import time
from typing import List, Tuple, Optional
import heapq

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class ConvolutionAnalyzer:
    """Handles convolution computations and spectral analysis."""
    
    @staticmethod
    def compute_convolution_fft(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency."""
        n = len(seq1)
        padded_length = 2 * n - 1
        fft_seq1 = fft(seq1, padded_length)
        fft_seq2 = fft(seq2, padded_length)
        conv_result = ifft(fft_seq1 * np.conj(fft_seq2)).real
        return conv_result[:n]
    
    @staticmethod
    def compute_c1(sequence: List[float]) -> float:
        """Compute C1 value for a given sequence."""
        if len(sequence) < 1:
            return float('inf')
        
        sequence = np.array(sequence)
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')
            
        conv_result = ConvolutionAnalyzer.compute_convolution_fft(sequence, sequence)
        max_conv = np.max(conv_result)
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

class SpectralGuidance:
    """Provides spectral analysis and guidance for sequence improvement."""
    
    @staticmethod
    def analyze_frequency(sequence: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform spectral analysis for frequency characteristics."""
        n = len(sequence)
        if n < 1:
            return np.array([]), np.array([])
        
        fft_coeffs = fft(sequence)
        power_spectrum = np.abs(fft_coeffs)**2
        freq_indices = np.arange(n)
        return freq_indices, power_spectrum
    
    @staticmethod
    def generate_guidance(sequence: List[float]) -> List[float]:
        """Generate a spectral-guided direction for sequence modification."""
        try:
            n = len(sequence)
            if n < 1:
                return sequence

            freq_indices, power_spectrum = SpectralGuidance.analyze_frequency(sequence)
            
            # Identify dominant frequencies
            threshold = np.percentile(power_spectrum, 75)
            dominant_freqs = freq_indices[power_spectrum > threshold]
            
            guided_sequence = np.array(sequence).copy()
            
            if len(dominant_freqs) > 0:
                mask = np.zeros(n)
                mask[dominant_freqs] = 1.0
                
                seq_fft = fft(guided_sequence)
                filtered_fft = seq_fft * mask
                guided_sequence = np.real(ifft(filtered_fft))
                
                # Enforce non-negativity
                guided_sequence = np.maximum(guided_sequence, 0)
            
            return guided_sequence.tolist()
            
        except Exception as e:
            warnings.warn(f"Spectral guidance error: {str(e)}")
            return sequence

class StepFunctionOptimizer:
    """Main optimizer for finding optimal step functions."""
    
    def __init__(self, max_iterations: int = 1000, timeout_seconds: int = 180):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self.performance_history = []
        
    def compute_inv_c1(self, sequence: List[float]) -> float:
        """Compute inverse of C1 (objective to maximize)."""
        c1 = ConvolutionAnalyzer.compute_c1(sequence)
        return 1.0 / c1 if c1 > 0 else 0.0

    def solve_convolution_lp(self, f_sequence: np.ndarray, rhs: float) -> Optional[np.ndarray]:
        """Solves the convolution LP with enhanced constraints handling."""
        try:
            n = len(f_sequence)
            if n < 1:
                return None

            # Precompute constraint matrix efficiently
            num_constraints = 2 * n - 1
            a_ub = np.zeros((num_constraints, n))
            b_ub = np.zeros(num_constraints)

            # Vectorized constraint generation for performance
            for k in range(num_constraints):
                coefficients = np.zeros(n)
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        coefficients[j] = f_sequence[i]
                a_ub[k] = coefficients
                b_ub[k] = rhs

            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)

            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            c = -np.ones(n)

            # Solve with multiple fallback methods
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                return result.x
            else:
                try:
                    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
                    if result.success:
                        return result.x
                except:
                    pass
                return None

        except Exception as e:
            warnings.warn(f"LP solver error: {str(e)}")
            return None

    def get_good_direction(self, sequence: List[float], iteration: int) -> Optional[List[float]]:
        """Returns the direction to move into the sequence with spectral guidance."""
        try:
            n = len(sequence)
            if n < 1:
                return None

            # Apply spectral guidance
            guided_seq = SpectralGuidance.generate_guidance(sequence)
            
            sum_sequence = np.sum(guided_seq)
            if sum_sequence < 1e-10:
                return None

            normalized_sequence = np.array(guided_seq) * np.sqrt(2 * n) / sum_sequence

            conv_result = ConvolutionAnalyzer.compute_convolution_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(conv_result)

            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)

            if g_fun is None:
                return None

            sum_g = np.sum(g_fun)
            if sum_g < 1e-10:
                return None

            normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g

            # Adaptive step size with faster decay
            t = 0.01 / (1.0 + 0.001 * n) * (1 - np.exp(-0.1 * n))

            new_sequence = (1 - t) * np.array(guided_seq) + t * normalized_g_fun

            new_sequence = np.maximum(new_sequence, 0)

            return new_sequence.tolist()

        except Exception as e:
            warnings.warn(f"Direction calculation error: {str(e)}")
            return None

    def create_structured_sequence(self, n: int) -> List[float]:
        """Create a structured sequence with blend of patterns."""
        exp_decay = np.exp(-np.linspace(0, 3, n))
        cosine_env = 0.5 + 0.5 * np.cos(np.linspace(0, 2*np.pi, n))
        random_part = np.random.rand(n)
        
        base_seq = 0.4 * exp_decay + 0.3 * cosine_env + 0.3 * random_part
        
        base_seq = base_seq / np.sum(base_seq) * 10
        base_seq = np.clip(base_seq, 0, 1000)

        return base_seq.tolist()
    
    def mutate_sequence(self, sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Apply random mutation with dynamic adjustment."""
        mutated = sequence.copy()
        n = len(mutated)

        num_mutations = max(1, int(n * mutation_rate))

        for _ in range(num_mutations):
            idx = random.randint(0, n - 1)
            change_factor = random.uniform(0.7, 1.3)
            mutated[idx] *= change_factor
            mutated[idx] = max(0, mutated[idx])

        return mutated

    def evolve_population(self, population: List[List[float]]) -> List[List[float]]:
        """Evolve population using tournament selection and crossover."""
        if len(population) < 2:
            return population
            
        # Evaluate fitness
        fitnesses = [self.compute_inv_c1(seq) for seq in population]
        
        # Preserve elite individuals
        elite_indices = np.argsort(fitnesses)[-min(5, len(population)):]
        elite_individuals = [population[i] for i in elite_indices]
        
        # Create new population
        new_population = elite_individuals[:]
        
        # Fill rest with offspring
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)

            # Crossover
            child = self.crossover_sequences(parent1, parent2)
            
            # Mutation
            child = self.mutate_sequence(child, 0.1)
            
            # Ensure minimum sum
            if sum(child) < 0.01:
                child[0] = 0.1
                
            new_population.append(child)
            
        return new_population

    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], k: int = 3) -> List[float]:
        """Select an individual using tournament selection."""
        if len(population) < k:
            selected_idx = np.argmax(fitnesses)
            return population[selected_idx]

        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]

    def crossover_sequences(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Perform crossover between two sequences."""
        min_len = min(len(parent1), len(parent2))
        
        # Uniform crossover
        child = []
        for i in range(min_len):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        
        # Extend with remaining elements
        if len(parent1) > min_len:
            child.extend(parent1[min_len:])
        elif len(parent2) > min_len:
            child.extend(parent2[min_len:])
            
        return child

    def optimize_sequence(self) -> List[float]:
        """Main optimization loop."""
        start_time = time.time()
        
        # Initialize population
        population_size = 30
        population = []
        for _ in range(population_size):
            n = random.randint(50, 900)
            sequence = self.create_structured_sequence(n)
            population.append(sequence)

        # Evolution loop
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds - 5:
                break
            
            # Evolve population
            population = self.evolve_population(population)
            
            # Evaluate and track best
            fitnesses = [self.compute_inv_c1(seq) for seq in population]
            current_best_fitness = max(fitnesses)
            
            if current_best_fitness > self.best_inv_c1:
                self.best_inv_c1 = current_best_fitness
                best_idx = np.argmax(fitnesses)
                self.best_sequence = population[best_idx].copy()
            
            self.performance_history.append(current_best_fitness)

        # Final refinement stage
        if self.best_sequence is not None:
            for _ in range(10):  # Few refinement steps
                if time.time() - start_time > self.timeout_seconds - 2:
                    break
                refined = self.get_good_direction(self.best_sequence, iteration)
                if refined is not None:
                    self.best_sequence = refined
                else:
                    break

        return self.best_sequence if self.best_sequence is not None else [random.random() for _ in range(100)]

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = StepFunctionOptimizer()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")