# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
from scipy.fft import fft, ifft
import time
from collections import deque
import random
import threading

# Configuration parameters
MAX_ITERATIONS = 1000
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TIME_LIMIT_SECONDS = 170
MIN_SEQ_LENGTH = 50
MAX_SEQ_LENGTH = 500
ADAPTIVE_DECAY_RATE = 0.95
ELITE_SIZE = 10

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class AutoCorrelationOptimizer:
    def __init__(self):
        self.history = deque(maxlen=10)
        self.best_sequence = None
        self.best_score = float('-inf')
        self.elite_sequences = []
        
    def convolve_fft(self, a, b):
        """Compute convolution using FFT for better performance."""
        n = len(a)
        # Zero-pad to avoid circular convolution effects
        padded_length = 2 * n - 1
        fa = fft(a, padded_length)
        fb = fft(b, padded_length)
        result = ifft(fa * fb).real
        return result[:n]

    def compute_c1(self, sequence):
        """Compute C1 for a given sequence."""
        if len(sequence) == 0 or abs(sum(sequence)) < 1e-10:
            return float('inf')
        convolved = self.convolve_fft(sequence, sequence)
        max_conv = np.max(convolved)
        sum_seq = sum(sequence)
        return (2 * len(sequence) * max_conv) / (sum_seq * sum_seq)

    def evaluate_sequence(self, sequence):
        """Evaluate a sequence by computing 1/C1."""
        c1 = self.compute_c1(sequence)
        if c1 == float('inf') or c1 > 1e10:
            return float('-inf')
        return 1.0 / c1  # We want to maximize 1/C1

    def generate_random_sequence(self, length=None):
        """Generate a random sequence with structure."""
        if length is None:
            length = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
        seq = np.random.uniform(0, 100, length)
        # Add some structure
        if np.random.random() < 0.3:
            idxs = np.random.choice(length, size=min(5, length//4), replace=False)
            seq[idxs] *= np.random.uniform(5, 20)
        return seq.tolist()

    def create_structured_sequence(self, length):
        """Create structured sequences such as step functions."""
        seq = [1.0] * length
        if np.random.random() < 0.5:
            # Create a step function
            split_point = np.random.randint(1, length-1)
            seq[split_point:] = [0.0] * (length - split_point)
        return seq

    def compute_gradient(self, sequence, epsilon=1e-5):
        """Compute gradient using finite differences."""
        seq_array = np.array(sequence, dtype=float)
        n = len(seq_array)

        # Compute base convolution
        convolved = self.convolve_fft(seq_array, seq_array)
        base_max = np.max(convolved)

        # Compute gradients for each element
        grad_dir = np.zeros(n)
        for i in range(n):
            temp_seq = seq_array.copy()
            temp_seq[i] += epsilon
            temp_conv = self.convolve_fft(temp_seq, temp_seq)
            temp_max = np.max(temp_conv)
            grad_dir[i] = -(temp_max - base_max) / epsilon  # Negative because we want to minimize

        return grad_dir

    def gradient_ascent_step(self, sequence, iteration=0):
        """Perform a gradient ascent step to improve the sequence."""
        try:
            # Convert to numpy array
            seq_array = np.array(sequence, dtype=float)
            n = len(seq_array)

            # Adaptive step size with exponential decay
            step_size = 0.01 * (ADAPTIVE_DECAY_RATE ** iteration)
            
            # Compute gradient using finite differences
            grad_dir = self.compute_gradient(sequence)
            
            # Apply gradient update
            updated_seq = seq_array + step_size * grad_dir
            updated_seq = np.maximum(updated_seq, 0.0)  # Ensure non-negativity
            
            # Normalize
            sum_updated = np.sum(updated_seq)
            if sum_updated > 0.01:
                updated_seq = updated_seq / sum_updated
            else:
                updated_seq = updated_seq + 0.01
                updated_seq = updated_seq / np.sum(updated_seq)
            
            return updated_seq.tolist()
        except Exception:
            # Fallback to random perturbation if gradient fails
            new_sequence = sequence.copy()
            for i in range(len(new_sequence)):
                if random.random() < 0.1:
                    new_sequence[i] = max(0, new_sequence[i] + random.uniform(-10, 10))
            return new_sequence

    def mutate_sequence(self, sequence, fitness_gradient=None):
        """Mutate a sequence based on its fitness gradient."""
        mutated = sequence.copy()
        length = len(mutated)
        
        # Adapt mutation based on fitness landscape
        if fitness_gradient is not None and abs(fitness_gradient) > 0.01:
            # If gradient suggests improvement, increase mutation magnitude
            mutation_scale = 1.5
        else:
            mutation_scale = 1.0
            
        # Apply mutations with variable rate
        for i in range(length):
            if np.random.random() < MUTATION_RATE * mutation_scale:
                # Adjust element by a random amount
                delta = np.random.uniform(-10, 10) * mutation_scale
                mutated[i] = max(0, mutated[i] + delta)
        
        return mutated

    def crossover_sequences(self, seq1, seq2):
        """Perform crossover between two sequences."""
        length = min(len(seq1), len(seq2))
        if length < 2:
            return seq1.copy()
            
        # Single-point crossover
        crossover_point = np.random.randint(1, length)
        child1 = seq1[:crossover_point] + seq2[crossover_point:]
        child2 = seq2[:crossover_point] + seq1[crossover_point:]
        
        # Ensure lengths match
        if len(child1) != len(seq1):
            child1 = child1 + [0.0] * (len(seq1) - len(child1))
        if len(child2) != len(seq2):
            child2 = child2 + [0.0] * (len(seq2) - len(child2))
            
        return child1 if np.random.random() < 0.5 else child2

    def fitness_landscape_guided_evolution(self, initial_sequences):
        """Perform evolutionary optimization guided by fitness landscape."""
        population = initial_sequences
        for iter_num in range(MAX_ITERATIONS):
            if time.time() - start_time > TIME_LIMIT_SECONDS:
                break
                
            # Evaluate all individuals
            fitness_scores = [self.evaluate_sequence(ind) for ind in population]
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_score:
                self.best_score = fitness_scores[best_idx]
                self.best_sequence = population[best_idx].copy()
                
            # Update elite sequences
            for i, (ind, fitness) in enumerate(zip(population, fitness_scores)):
                if len(self.elite_sequences) < ELITE_SIZE or fitness > self.elite_sequences[0][0]:
                    self.elite_sequences.append((fitness, ind.copy()))
                    self.elite_sequences.sort(key=lambda x: x[0], reverse=True)
                    self.elite_sequences = self.elite_sequences[:ELITE_SIZE]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            best_individuals = [population[i] for i in sorted_indices[:POPULATION_SIZE//2]]
            
            # Generate new population
            new_population = []
            
            # Elitism: keep best individuals
            new_population.extend(best_individuals)
            
            # Crossover and mutation
            while len(new_population) < POPULATION_SIZE:
                # Select parents
                parent1 = random.choice(best_individuals)
                parent2 = random.choice(best_individuals)
                
                # Crossover
                if np.random.random() < CROSSOVER_RATE:
                    child = self.crossover_sequences(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self.mutate_sequence(child)
                
                new_population.append(child)
            
            population = new_population[:POPULATION_SIZE]
            
        return self.best_sequence if self.best_sequence is not None else initial_sequences[0]

    def optimize_with_de(self, initial_seq):
        """Optimize using differential evolution."""
        bounds = [(0.0, 1000.0)] * len(initial_seq)
        try:
            result = differential_evolution(
                lambda s: -self.evaluate_sequence(s),  # Negative because DE minimizes
                bounds,
                maxiter=30,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=random.randint(0, 1000),
                polish=True
            )
            if result.success:
                return result.x.tolist()
        except Exception:
            pass
        return initial_seq

    def search_for_best_sequence(self):
        """Main function to search for the best coefficient sequence."""
        global start_time
        start_time = time.time()
        
        # Create diverse initial population
        initial_sequences = []
        
        # Add structured sequences
        for _ in range(5):
            n = np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            seq = self.create_structured_sequence(n)
            initial_sequences.append(seq)
        
        # Add random sequences
        for _ in range(5):
            seq = self.generate_random_sequence()
            initial_sequences.append(seq)
            
        # Add known good structures
        initial_sequences.append([1.0] * 100)  # Uniform
        initial_sequences.append([1.0] * 50 + [0.0] * 50)  # Step function
        
        # Run evolutionary optimization
        final_sequence = self.fitness_landscape_guided_evolution(initial_sequences)
        
        # Perform final optimization with DE
        final_sequence = self.optimize_with_de(final_sequence)
        
        # Ensure non-negative values
        final_sequence = [max(0, x) for x in final_sequence]
        
        # Final gradient ascent
        final_sequence = self.gradient_ascent_step(final_sequence, iteration=MAX_ITERATIONS)
        
        return final_sequence

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    optimizer = AutoCorrelationOptimizer()
    return optimizer.search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")