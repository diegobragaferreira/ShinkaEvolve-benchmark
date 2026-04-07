# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import time
import random
import multiprocessing as mp
from functools import partial
import copy
from typing import List, Tuple, Optional
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class FastAutocorrelationEvolutionarySearch:
    """
    An optimized hybrid optimizer combining evolutionary search with adaptive gradient-based refinement
    to maximize 1/C₁ for step function sequences.
    """

    def __init__(self, pop_size: int = 50, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.best_score = 0.0
        self.best_sequence = None
        self.fitness_cache = {}  # Memoization for fitness evaluations
        self.diversity_history = []

    @staticmethod
    def evaluate_sequence(sequence: List[float]) -> Tuple[float, float, float, float]:
        """
        Evaluate a sequence and return its performance metrics.
        
        Args:
            sequence: List of non-negative real numbers representing step heights
            
        Returns:
            tuple: (C₁, 1/C₁, max_convolution_value, sum_of_sequence)
        """
        # Check cache first
        seq_tuple = tuple(sequence)
        if seq_tuple in FastAutocorrelationEvolutionarySearch.fitness_cache:
            return FastAutocorrelationEvolutionarySearch.fitness_cache[seq_tuple]
        
        try:
            # Convert to numpy array
            a = np.array(sequence)
            sum_a = np.sum(a)
            
            # Avoid division by zero or negligible sums
            if sum_a < 1e-10:
                result = (float('inf'), 0.0, 0.0, sum_a)
                FastAutocorrelationEvolutionarySearch.fitness_cache[seq_tuple] = result
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
            
            # Add secondary fitness regularization to penalize high variance in convolution profile
            conv_variance = np.var(b)
            regularized_inv_c1 = inv_c1 / (1 + 0.01 * conv_variance)
            
            result = (c1, regularized_inv_c1, max_b, sum_a)
            FastAutocorrelationEvolutionarySearch.fitness_cache[seq_tuple] = result
            return result
        except Exception as e:
            result = (float('inf'), 0.0, 0.0, 0.0)
            FastAutocorrelationEvolutionarySearch.fitness_cache[seq_tuple] = result
            return result

    @staticmethod
    def evaluate_population_parallel(population: List[List[float]], 
                                   chunk_size: int = 10) -> List[float]:
        """
        Evaluate a list of sequences in parallel.
        
        Args:
            population: List of sequences to evaluate
            chunk_size: Number of sequences per worker
            
        Returns:
            List of performance metrics (1/C₁) for each sequence
        """
        # Split the population into chunks
        chunks = [population[i:i+chunk_size] for i in range(0, len(population), chunk_size)]
        
        # Use multiprocessing to evaluate chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(FastAutocorrelationEvolutionarySearch._evaluate_chunk, chunks)
        
        # Flatten results
        flattened_results = [item for sublist in results for item in sublist]
        return flattened_results

    @staticmethod
    def _evaluate_chunk(chunk: List[List[float]]) -> List[float]:
        """Helper function to evaluate a chunk of sequences."""
        return [FastAutocorrelationEvolutionarySearch.evaluate_sequence(seq)[1] for seq in chunk]

    @staticmethod
    def generate_multiscale_sequence() -> List[float]:
        """Generate a structured sequence using multi-scale pattern."""
        # Sample sequence length from a skewed distribution favoring smaller sizes
        length = random.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        # Create a base sinusoidal pattern
        sequence = [abs(np.sin(i * np.pi / length)) * 500 for i in range(length)]
        # Add some high frequency noise for exploration
        for i in range(length):
            if random.random() < 0.2:
                sequence[i] += random.uniform(-100, 100)
        # Ensure non-negative
        sequence = [max(0, x) for x in sequence]
        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01
        return sequence

    @staticmethod
    def generate_structured_sequence() -> List[float]:
        """Generate a sequence using a combination of known effective patterns."""
        # Prefer shorter sequences for faster evaluation
        length = random.choice([50, 100, 150, 200, 250])
        
        # Mix of sinusoidal and exponential decay
        sequence = []
        for i in range(length):
            if random.random() < 0.5:
                sequence.append(abs(np.sin(i * np.pi / length)) * 500)
            else:
                sequence.append(np.exp(-i / (length / 5)) * 500)
                
        # Ensure non-negative
        sequence = [max(0, x) for x in sequence]
        # Ensure sum is meaningful
        if sum(sequence) < 0.01:
            sequence[random.randint(0, length-1)] += 0.01
        return sequence

    @staticmethod
    def mutate_sequence(sequence: List[float], mu: float = 0, sigma: float = 100, 
                       indpb: float = 0.1) -> List[float]:
        """Mutate a sequence with Gaussian noise."""
        mutated = copy.deepcopy(sequence)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                mutated[i] = max(0, mutated[i])  # Ensure non-negative
        return mutated

    @staticmethod
    def crossover_sequences(seq1: List[float], seq2: List[float], 
                          cxpb: float = 0.5) -> Tuple[List[float], List[float]]:
        """Perform uniform crossover between two sequences."""
        child1, child2 = copy.deepcopy(seq1), copy.deepcopy(seq2)
        for i in range(len(child1)):
            if random.random() < cxpb:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    @staticmethod
    def gradient_refine(sequence: List[float], steps: int = 50, lr: float = 1e-4) -> List[float]:
        """Apply gradient-based refinement to improve the sequence with momentum."""
        seq = np.array(sequence, dtype=float)
        n = len(seq)
        
        # Normalize to avoid numerical issues
        sum_seq = np.sum(seq)
        if sum_seq < 1e-10:
            seq += 1e-5
            sum_seq = np.sum(seq)
        seq /= sum_seq
        
        # Initialize momentum for smoother updates
        momentum = np.zeros_like(seq)
        beta = 0.9  # Momentum factor
        
        for step in range(steps):
            # Compute convolution
            conv = fftconvolve(seq, seq, mode='full')
            conv = conv[n-1:2*n-1]
            max_conv = np.max(conv)
            
            # Compute gradient estimate using finite differences
            grad = np.zeros_like(seq)
            epsilon = 1e-6
            for i in range(n):
                perturbed = seq.copy()
                perturbed[i] += epsilon
                perturbed /= np.sum(perturbed)
                
                pert_conv = fftconvolve(perturbed, perturbed, mode='full')
                pert_conv = pert_conv[n-1:2*n-1]
                max_pert = np.max(pert_conv)
                
                grad[i] = (max_pert - max_conv) / epsilon
            
            # Apply momentum to gradient
            momentum = beta * momentum + (1 - beta) * grad
            
            # Update using gradient ascent with momentum
            seq += lr * momentum
            seq = np.maximum(seq, 0)  # Ensure non-negative
            
            # Renormalize
            sum_seq = np.sum(seq)
            if sum_seq > 0:
                seq /= sum_seq
                
        return seq.tolist()

    @staticmethod
    def compute_diversity(population: List[List[float]]) -> float:
        """Compute population diversity using standard deviation of fitnesses."""
        fitnesses = [FastAutocorrelationEvolutionarySearch.evaluate_sequence(seq)[1] for seq in population]
        return np.std(fitnesses) if len(fitnesses) > 1 else 0.0

    def adaptive_optimize(self) -> List[float]:
        """Run the adaptive hybrid evolutionary-gradient optimization process."""
        start_time = time.time()
        timeout = 170  # Leave 10 seconds for cleanup
        
        # Initialize population with mixed strategies
        population = []
        for i in range(self.pop_size):
            if i < self.pop_size // 2:
                population.append(self.generate_multiscale_sequence())
            else:
                population.append(self.generate_structured_sequence())
        
        # Evaluate initial population
        fitnesses = self.evaluate_population_parallel(population)
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        self.best_score = fitnesses[best_idx]
        self.best_sequence = copy.deepcopy(population[best_idx])
        
        print(f"Initial best score: {self.best_score:.6f}")
        
        # Track fitness history for convergence detection
        fitness_history = [self.best_score]
        stagnation_counter = 0
        
        # Begin evolution
        for gen in range(self.generations):
            if time.time() - start_time > timeout:
                break
                
            # Track diversity for adaptive strategies
            diversity = self.compute_diversity(population)
            self.diversity_history.append(diversity)
            
            # Dynamic tournament size based on diversity and generation
            if gen < 10 or diversity > 0.1:
                tournament_size = 5
            elif gen > 30 or diversity < 0.01:
                tournament_size = 3
            else:
                tournament_size = 4
            
            # Selection using dynamic tournament selection
            selected = self.dynamic_tournament_selection(population, fitnesses, tournament_size)
            
            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover_sequences(selected[i], selected[i+1])
                    # Apply adaptive mutation
                    child1 = self.adaptive_mutate_sequence(child1, gen, diversity)
                    child2 = self.adaptive_mutate_sequence(child2, gen, diversity)
                    offspring.extend([child1, child2])
                else:
                    # Handle odd population size
                    offspring.append(self.adaptive_mutate_sequence(selected[i], gen, diversity))
            
            # Keep offspring size consistent
            offspring = offspring[:self.pop_size]
            
            # Apply gradient refinement to top individuals (adaptive)
            top_indices = np.argsort(fitnesses)[-min(10, len(offspring)):][::-1]
            refine_count = max(1, int(0.3 * len(top_indices)))  # 30% of top individuals
            for i in range(refine_count):
                idx = top_indices[i]
                offspring[idx] = self.gradient_refine(offspring[idx])
            
            # Evaluate offspring
            offspring_fitnesses = self.evaluate_population_parallel(offspring)
            
            # Replace population with offspring
            population = offspring
            fitnesses = offspring_fitnesses
            
            # Update best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_sequence = copy.deepcopy(population[best_idx])
                stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                stagnation_counter += 1  # Increment if no improvement
            
            # Update fitness history
            fitness_history.append(self.best_score)
            if len(fitness_history) > 10:
                fitness_history.pop(0)  # Keep only last 10 values

            print(f"Gen {gen}: Best = {self.best_score:.6f}, Diversity = {diversity:.6f}")
            
            # Adaptive termination check
            if gen > 10:
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < 1e-8:
                    print("Fitness plateau detected, stopping early")
                    break

                if stagnation_counter > 5:
                    print("Population converged and stagnated, stopping early")
                    break
                    
        return self.best_sequence

    @staticmethod
    def dynamic_tournament_selection(population: List[List[float]], 
                                    fitnesses: List[float], 
                                    tournament_size: int = 3) -> List[List[float]]:
        """Select individuals using dynamic tournament selection with variable size."""
        selected = []
        for _ in range(len(population)):
            # Select randomly from population
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(copy.deepcopy(population[winner_index]))
        return selected

    @staticmethod
    def adaptive_mutate_sequence(sequence: List[float], generation: int, diversity: float) -> List[float]:
        """Apply adaptive mutation with generation and diversity awareness."""
        mutated = copy.deepcopy(sequence)
        n = len(mutated)
        
        # Adaptive mutation rate
        mutation_rate = 0.15 if generation < 10 else 0.1
        mutation_rate = max(0.01, mutation_rate - diversity * 0.05)
        
        # Adaptive mutation strength
        base_strength = 100.0
        strength_factor = max(0.1, 1.0 - generation / 50.0)
        mutation_strength = base_strength * strength_factor
        
        for i in range(n):
            if random.random() < mutation_rate:
                # Use a mix of Gaussian and Cauchy for adaptive behavior
                if random.random() < 0.3:
                    # Heavy-tailed Cauchy for exploration in early stages
                    mutated[i] += np.random.standard_cauchy() * mutation_strength
                else:
                    # Gaussian for fine-tuning
                    mutated[i] += random.gauss(0, mutation_strength / 10.0)
                mutated[i] = max(0, mutated[i])  # Ensure non-negative
        return mutated

    def optimize(self) -> List[float]:
        """Execute the optimization procedure."""
        # Run adaptive optimization routine
        return self.adaptive_optimize()

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    optimizer = FastAutocorrelationEvolutionarySearch(pop_size=50, generations=50)
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
