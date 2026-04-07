# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from joblib import Parallel, delayed
import warnings
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class FitnessEvaluator:
    """Handles all fitness computation and caching."""
    
    def __init__(self):
        self.cache: Dict[Tuple[float, ...], Tuple[float, float, float, float]] = {}
        
    def compute_c1(self, sequence: List[float]) -> float:
        """Compute C₁ for a given sequence using FFT for efficiency."""
        if len(sequence) == 0:
            return float('inf')
        
        # Convert to numpy array
        a = np.array(sequence)
        
        # Compute autoconvolution using FFT for efficiency
        conv = signal.fftconvolve(a, a, mode='full')
        conv = conv[len(a)-1:]  # Take the relevant part
        
        # Max convolution value
        max_conv = np.max(conv)
        
        # Sum of sequence squared
        sum_sq = np.sum(a)**2
        
        if sum_sq == 0:
            return float('inf')
        
        # Compute C₁
        c1 = 2 * len(a) * max_conv / sum_sq
        return c1
    
    def compute_inv_c1(self, sequence: List[float]) -> float:
        """Compute 1/C₁ for a given sequence."""
        seq_tuple = tuple(sequence)
        if seq_tuple in self.cache:
            return self.cache[seq_tuple][1]
            
        c1 = self.compute_c1(sequence)
        if c1 == 0 or np.isnan(c1):
            result = 0.0
        else:
            result = 1.0 / c1
            
        self.cache[seq_tuple] = (c1, result, max_conv, sum_sq) if 'max_conv' in locals() else (c1, result, 0.0, sum_sq)
        return result

class SequenceGenerator:
    """Handles generation of initial and mutated sequences."""
    
    @staticmethod
    def generate_structured_sequence(min_length: int = 10, max_length: int = 1000, max_height: float = 1000.0) -> List[float]:
        """Generate a structured sequence using exponential decay for better initialization."""
        n = random.randint(min_length, max_length)
        sequence = []
        for i in range(n):
            base_val = max(0.01, max_height * np.exp(-i * 0.05))
            noise = random.uniform(0.9, 1.1)
            sequence.append(base_val * noise)
        return sequence
    
    @staticmethod
    def generate_memory_based_sequence(elite_sequences: List[List[float]], n: int) -> List[float]:
        """Generate a sequence based on learned patterns from elite sequences."""
        if not elite_sequences:
            return SequenceGenerator.generate_structured_sequence(n=n)
        
        # Take the average of elite sequences to form a base
        avg_sequence = np.mean(elite_sequences, axis=0)
        # Add some noise to prevent overfitting
        noise = [random.uniform(-0.1, 0.1) for _ in range(n)]
        base_seq = [max(0.01, val * (1 + noise[i])) for i, val in enumerate(avg_sequence)]
        return base_seq

class PopulationManager:
    """Manages population creation, selection, and evolutionary operations."""
    
    def __init__(self, population_size: int, elite_size: int = 10):
        self.population_size = population_size
        self.elite_size = elite_size
    
    def generate_population(self, min_n: int = 50, max_n: int = 1000, elite_sequences: List[List[float]] = None) -> List[List[float]]:
        """Generate diverse initial population with structured sequences."""
        population = []
        for _ in range(self.population_size):
            n = random.randint(min_n, max_n)
            # Use a mix of structured and memory-based sequences
            if elite_sequences and random.random() < 0.3:
                # Memory-based sequence
                individual = SequenceGenerator.generate_memory_based_sequence(elite_sequences, n)
            elif random.random() < 0.7:
                # Structured sequence
                individual = SequenceGenerator.generate_structured_sequence(n)
            else:
                # Random sequence
                individual = [random.uniform(0.1, 100) for _ in range(n)]
            population.append(individual)
        return population
    
    def mutate_sequence(self, sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Apply mutation to sequence with multiplicative Gaussian perturbation."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply multiplicative Gaussian perturbation
                perturbation = random.gauss(1, 0.1)
                mutated[i] *= abs(perturbation)  # Ensure non-negative
                mutated[i] = max(0.01, mutated[i])
        return mutated
    
    def crossover_sequences(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Perform crossover between two sequences."""
        min_len = min(len(parent1), len(parent2))
        crossover_point = random.randint(1, min_len - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def adaptive_tournament_selection(self, population: List[List[float]], 
                                    fitness_scores: List[float], diversity: float) -> List[float]:
        """Perform adaptive tournament selection based on population diversity."""
        # Adjust tournament size based on diversity
        if diversity > 0.1:
            tournament_size = 7
        elif diversity < 0.05:
            tournament_size = 3
        else:
            tournament_size = 5
        
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Select the best individual from tournament
        best_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[best_idx]
    
    def adaptive_mutation_rate(self, population_fitnesses: List[float]) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        if len(population_fitnesses) < 2:
            return 0.1

        std_dev = np.std(population_fitnesses)
        avg_fitness = np.mean(population_fitnesses)

        # Higher diversity = higher mutation rate
        if avg_fitness > 0:
            mutation_rate = min(0.3, max(0.01, 0.1 + std_dev / avg_fitness))
        else:
            mutation_rate = 0.1

        return mutation_rate

class LocalSearchEngine:
    """Handles local refinement strategies for sequence improvement."""
    
    @staticmethod
    def local_search_refinement(sequence: List[float], max_iterations: int = 10) -> List[float]:
        """Apply local search refinement to improve sequence."""
        best_seq = sequence.copy()
        best_fitness = 1.0 / (LocalSearchEngine.compute_c1_if_valid(sequence) or float('inf'))
        
        for _ in range(max_iterations):
            # Try small perturbations
            mutant = LocalSearchEngine._mutate_sequence_small(best_seq, 0.05)
            mutant_fitness = 1.0 / (LocalSearchEngine.compute_c1_if_valid(mutant) or float('inf'))

            if mutant_fitness > best_fitness:
                best_seq = mutant
                best_fitness = mutant_fitness

        return best_seq
    
    @staticmethod
    def multi_start_local_search(sequence: List[float], num_starts: int = 5) -> List[float]:
        """Perform multi-start local search to find better local optima."""
        best_seq = sequence.copy()
        best_fitness = 1.0 / (LocalSearchEngine.compute_c1_if_valid(sequence) or float('inf'))

        for _ in range(num_starts):
            # Perturb the sequence slightly
            perturbed = LocalSearchEngine._mutate_sequence_small(sequence, 0.05)
            refined = LocalSearchEngine.local_search_refinement(perturbed, 10)
            refined_fitness = 1.0 / (LocalSearchEngine.compute_c1_if_valid(refined) or float('inf'))
            
            if refined_fitness > best_fitness:
                best_seq = refined
                best_fitness = refined_fitness

        return best_seq
    
    @staticmethod
    def _mutate_sequence_small(sequence: List[float], mutation_rate: float) -> List[float]:
        """Small mutation helper for local search."""
        mutated = sequence.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply multiplicative Gaussian perturbation
                perturbation = random.gauss(1, 0.05)
                mutated[i] *= abs(perturbation)  # Ensure non-negative
                mutated[i] = max(0.01, mutated[i])
        return mutated
    
    @staticmethod
    def compute_c1_if_valid(sequence: List[float]) -> Optional[float]:
        """Compute C1 if the sequence is valid, otherwise return None."""
        try:
            if len(sequence) == 0:
                return None
            a = np.array(sequence)
            sum_sq = np.sum(a)**2
            if sum_sq < 1e-10:
                return None
            
            conv = signal.fftconvolve(a, a, mode='full')
            conv = conv[len(a)-1:] 
            max_conv = np.max(conv)
            return 2 * len(a) * max_conv / sum_sq
        except Exception:
            return None

class AutocorrelationOptimizer:
    """Main optimizer class that orchestrates the entire process."""
    
    def __init__(self, 
                 pop_size: int = 50,
                 generations: int = 100,
                 elite_size: int = 10,
                 max_stagnation: int = 20):
        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.max_stagnation = max_stagnation
        self.fitness_evaluator = FitnessEvaluator()
        self.population_manager = PopulationManager(pop_size, elite_size)
        self.local_search_engine = LocalSearchEngine()
        self.best_score = 0.0
        self.best_individual = None
        self.elite_history = []  # Track elite sequences across generations
    
    def run_evolution(self) -> List[float]:
        """Run the evolutionary optimization process."""
        start_time = time.time()
        max_time = 175  # Leave some time for cleanup

        # Initialize population with structured sequences
        population = self.population_manager.generate_population(elite_sequences=self.elite_history)

        best_solution = None
        best_fitness = 0.0
        stagnation_counter = 0
        fitness_history = []

        for generation in range(self.generations):
            # Check time limit
            if time.time() - start_time > max_time:
                break

            # Evaluate fitness for all individuals in parallel
            fitness_scores = Parallel(n_jobs=-1)(
                delayed(self.fitness_evaluator.compute_inv_c1)(individual) 
                for individual in population
            )

            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_solution = population[best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Update best score and individual
            if best_fitness > self.best_score:
                self.best_score = best_fitness
                self.best_individual = best_solution.copy()

            fitness_history.append(best_fitness)
            
            # Check for stagnation using multi-generational trend analysis
            if len(fitness_history) > 10:
                recent_improvement = np.mean(fitness_history[-10:]) - np.mean(fitness_history[:-10])
                if recent_improvement < 0.0001:
                    stagnation_counter += 1
                    if stagnation_counter >= self.max_stagnation:
                        # Reset with new diverse population
                        population = self.population_manager.generate_population(elite_sequences=self.elite_history)
                        stagnation_counter = 0
            else:
                stagnation_counter = 0

            # Calculate adaptive mutation rate
            mutation_rate = self.population_manager.adaptive_mutation_rate(fitness_scores)

            # Selection: keep top individuals
            sorted_indices = np.argsort(fitness_scores)[::-1][:self.elite_size]
            elite = [population[i] for i in sorted_indices]
            self.elite_history.append(copy.deepcopy(elite))

            # Apply multi-start local search to elite members
            refined_elite = []
            for ind in elite:
                refined = self.local_search_engine.multi_start_local_search(ind)
                refined_elite.append(refined)
            elite = refined_elite

            # Create new population through selection, crossover, and mutation
            new_population = elite.copy()

            while len(new_population) < self.pop_size:
                # Adaptive tournament selection
                parents = [
                    self.population_manager.adaptive_tournament_selection(
                        population, 
                        fitness_scores, 
                        np.std(fitness_scores) / max(1e-10, np.mean(fitness_scores))
                    ) 
                    for _ in range(2)
                ]
                child = self.population_manager.crossover_sequences(parents[0], parents[1])
                mutated_child = self.population_manager.mutate_sequence(child, mutation_rate)
                new_population.append(mutated_child)

            population = new_population

        # Final multi-start local search on best solution
        if best_solution is not None:
            best_solution = self.local_search_engine.multi_start_local_search(best_solution, 10)

        return best_solution if best_solution is not None else SequenceGenerator.generate_structured_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize optimizer
    optimizer = AutocorrelationOptimizer(
        pop_size=50,
        generations=100,
        elite_size=10,
        max_stagnation=20
    )
    
    # Run optimization
    best_sequence = optimizer.run_evolution()

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")