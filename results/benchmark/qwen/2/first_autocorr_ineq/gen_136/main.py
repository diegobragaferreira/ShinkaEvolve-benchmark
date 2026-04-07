# EVOLVE-BLOCK-START
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from scipy.fft import fft, ifft
import random
import time
import warnings
from typing import List, Tuple, Optional
import math

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

class AutocorrelationGradientEvolution:
    def __init__(self, max_iterations: int = 1000, timeout_seconds: int = 180):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.best_sequence = None
        self.best_inv_c1 = 0.0
        self.elite_sequences = []
        self.performance_history = []
        self.iteration_count = 0
        
    def compute_convolution_fft(self, seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
        """Compute convolution using FFT for efficiency."""
        n = len(seq1)
        padded_length = 2 * n - 1
        fft_seq1 = fft(seq1, padded_length)
        fft_seq2 = fft(seq2, padded_length)
        conv_result = ifft(fft_seq1 * np.conj(fft_seq2)).real
        return conv_result[:n]
    
    def compute_c1(self, sequence: List[float]) -> float:
        """Compute C1 value for a given sequence."""
        if len(sequence) < 1:
            return float('inf')
        
        sequence = np.array(sequence)
        sum_a = np.sum(sequence)
        if sum_a < 1e-10:
            return float('inf')
            
        conv_result = self.compute_convolution_fft(sequence, sequence)
        max_conv = np.max(conv_result)
        n = len(sequence)
        c1 = (2 * n * max_conv) / (sum_a ** 2)
        return c1

    def compute_inv_c1(self, sequence: List[float]) -> float:
        """Compute inverse of C1 (the value we want to maximize)."""
        c1 = self.compute_c1(sequence)
        return 1.0 / c1 if c1 > 0 else 0.0

    def spectral_initialization(self, n: int) -> List[float]:
        """Initialize sequence using spectral analysis for better structure."""
        # Create a structured sequence with exponential decay and cosine components
        exp_decay = np.exp(-np.linspace(0, 3, n))
        cosine_env = 0.5 + 0.5 * np.cos(np.linspace(0, 2*np.pi, n))
        random_part = np.random.rand(n)
        
        # Blend patterns with different weights
        base_seq = 0.4 * exp_decay + 0.3 * cosine_env + 0.3 * random_part
        
        # Normalize and scale
        base_seq = base_seq / np.sum(base_seq) * 10
        
        # Ensure non-negativity and bounds
        base_seq = np.clip(base_seq, 0, 1000)
        
        return base_seq.tolist()
    
    def compute_gradient_descent_step(self, sequence: List[float], steps: int = 50) -> Optional[List[float]]:
        """Optimize sequence using gradient descent with PyTorch."""
        try:
            n = len(sequence)
            if n < 1:
                return None
                
            # Convert to PyTorch tensor with gradient tracking
            seq_tensor = torch.tensor(sequence, dtype=torch.float32, requires_grad=True)
            
            # Set up optimizer with adaptive learning rate
            optimizer = optim.Adam([seq_tensor], lr=0.01)
            
            # Optimize over several steps with decreasing learning rate
            for i in range(steps):
                optimizer.zero_grad()
                
                # Compute sum and autoconvolution
                sum_a = seq_tensor.sum()
                if sum_a < 1e-10:
                    return None
                    
                # Compute autoconvolution
                conv_result = torch.nn.functional.conv1d(
                    seq_tensor.view(1, 1, -1), 
                    seq_tensor.view(1, 1, -1), 
                    padding=n-1
                ).squeeze()[:n]
                
                # Max convolution value
                max_conv = conv_result.max()
                
                # Compute C1
                c1 = 2 * n * max_conv / (sum_a ** 2)
                
                # We want to maximize 1/C1, so minimize -1/C1
                loss = -1.0 / c1
                
                # Compute gradients
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Ensure non-negativity
                with torch.no_grad():
                    seq_tensor.clamp_(min=0)
                    
                # Adaptive learning rate decay
                if i % 10 == 0 and i > 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
            
            # Convert back to list
            optimized_sequence = seq_tensor.detach().numpy().tolist()
            return optimized_sequence
            
        except Exception as e:
            warnings.warn(f"Error in gradient descent: {str(e)}")
            return None

    def get_good_direction_to_move_into(self, sequence: List[float], iteration: int) -> Optional[List[float]]:
        """Returns the direction to move into the sequence using gradient guidance."""
        try:
            n = len(sequence)
            if n < 1:
                return None

            # Apply spectral guidance to get a better starting point
            guided_seq = self.spectral_initialization(n)
            
            # Normalize the sequence appropriately for gradient descent
            sum_sequence = np.sum(guided_seq)
            if sum_sequence < 1e-10:
                return None

            # Normalize sequence
            normalized_sequence = np.array(guided_seq) * np.sqrt(2 * n) / sum_sequence

            # Compute convolution using FFT for efficiency
            conv_result = self.compute_convolution_fft(normalized_sequence, normalized_sequence)
            rhs = np.max(conv_result)

            # Solve the LP problem to enhance gradient guidance
            # Not directly used for gradient computation but provides structural insight
            g_fun = self.solve_convolution_lp(normalized_sequence, rhs)
            
            if g_fun is None:
                return None

            # Normalize the result again
            sum_g = np.sum(g_fun)
            if sum_g < 1e-10:
                return None

            normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g

            # Use adaptive step size based on iteration and sequence complexity
            t = 0.01 / (1.0 + 0.001 * n) * (1.0 - math.exp(-iteration / 50))  # Exponential decay

            # Create new sequence with adaptive mixing
            new_sequence = (1 - t) * np.array(guided_seq) + t * normalized_g_fun

            # Ensure non-negativity
            new_sequence = np.maximum(new_sequence, 0)

            return new_sequence.tolist()

        except Exception as e:
            warnings.warn(f"Error in get_good_direction_to_move_into: {str(e)}")
            return None

    def solve_convolution_lp(self, f_sequence, rhs):
        """Solves the convolution LP for a given sequence and RHS."""
        try:
            n = len(f_sequence)
            if n < 1:
                return None

            # Constraint matrix A_ub such that A_ub * x <= b_ub
            num_constraints = 2 * n - 1
            a_ub = np.zeros((num_constraints, n))
            b_ub = np.zeros(num_constraints)

            # Fill constraint matrix efficiently
            for k in range(num_constraints):
                for i in range(n):
                    j = k - i
                    if 0 <= j < n:
                        a_ub[k, j] = f_sequence[i]
                b_ub[k] = rhs

            # Add non-negativity constraints
            a_ub_nonneg = -np.eye(n)
            b_ub_nonneg = np.zeros(n)

            # Combine constraints
            a_ub = np.vstack([a_ub, a_ub_nonneg])
            b_ub = np.hstack([b_ub, b_ub_nonneg])

            # Define objective function (minimize -sum x, i.e., maximize sum x)
            c = -np.ones(n)

            # Solve linear programming problem
            from scipy import optimize
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

            if result.success:
                return result.x
            else:
                # Try with another solver method
                try:
                    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
                    if result.success:
                        return result.x
                except:
                    pass
                # If everything fails, return None
                return None

        except Exception as e:
            warnings.warn(f"Error in solve_convolution_lp: {str(e)}")
            return None

    def mutate_sequence(self, sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
        """Apply random mutation to a sequence."""
        mutated = sequence.copy()
        n = len(mutated)

        # Determine number of mutations based on sequence length and rate
        num_mutations = max(1, int(n * mutation_rate))

        for _ in range(num_mutations):
            idx = random.randint(0, n - 1)
            # Small random change
            change_factor = random.uniform(0.8, 1.2)
            mutated[idx] *= change_factor
            mutated[idx] = max(0, mutated[idx])  # Ensure non-negative

        return mutated

    def evolve_population(self, population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
        """Evolve a population using tournament selection and crossover."""
        if len(population) < 2:
            return population

        # Select elite individuals
        elite_indices = np.argsort(fitnesses)[-min(10, len(population)):]
        elite_individuals = [population[i] for i in elite_indices]

        # Create new population
        new_population = elite_individuals[:]

        # Fill rest of population through selection, crossover, and mutation
        while len(new_population) < len(population):
            # Tournament selection for parents
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)

            # Crossover: weighted average
            child = []
            min_len = min(len(parent1), len(parent2))
            for i in range(min_len):
                # Blend parents with some randomness
                blend_factor = random.uniform(0.3, 0.7)
                val = blend_factor * parent1[i] + (1 - blend_factor) * parent2[i]
                child.append(val)

            # Mutation
            child = self.mutate_sequence(child, 0.1)

            # Ensure minimum sum requirement
            if sum(child) < 0.01:
                child[0] = 0.1

            new_population.append(child)

        return new_population

    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], k: int = 3) -> List[float]:
        """Select an individual from population using tournament selection."""
        if len(population) < k:
            selected_idx = np.argmax(fitnesses)
            return population[selected_idx]

        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]

    def fallback_strategy(self, current_sequence: List[float]) -> List[float]:
        """Multi-tiered fallback strategy to prevent stagnation."""
        # Tier 1: Try mirrored sequence
        mirrored_seq = current_sequence[::-1]
        mirrored_inv_c1 = self.compute_inv_c1(mirrored_seq)
        if mirrored_inv_c1 > self.compute_inv_c1(current_sequence):
            return mirrored_seq
        
        # Tier 2: Apply bounded random perturbations
        perturbed_seq = [
            max(0, x + random.uniform(-0.1, 0.1) * x) if x > 0 else random.uniform(0, 1)
            for x in current_sequence
        ]
        perturbed_inv_c1 = self.compute_inv_c1(perturbed_seq)
        if perturbed_inv_c1 > self.compute_inv_c1(current_sequence):
            return perturbed_seq
        
        # Tier 3: Resort to randomization
        return [random.random() for _ in range(len(current_sequence))]

    def optimize_sequence(self) -> List[float]:
        """Main optimization loop combining gradient and evolutionary approaches."""
        start_time = time.time()
        
        # Initialize population with diverse sequences
        population_size = 20
        population = []
        for _ in range(population_size):
            n = random.randint(100, 1000)
            sequence = self.spectral_initialization(n)
            population.append(sequence)
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.timeout_seconds:
                break
                
            # Evaluate fitness for current population
            fitnesses = [self.compute_inv_c1(seq) for seq in population]
            
            # Track best performance
            current_best_fitness = max(fitnesses)
            if current_best_fitness > self.best_inv_c1:
                self.best_inv_c1 = current_best_fitness
                best_idx = np.argmax(fitnesses)
                self.best_sequence = population[best_idx].copy()
            
            # Preserve elites
            elite_indices = np.argsort(fitnesses)[-min(5, len(population)):]
            elite_individuals = [population[i] for i in elite_indices]
            
            # Evolve population
            population = self.evolve_population(population, fitnesses)
            
            # Apply gradient descent to top individuals
            for i in range(min(3, len(elite_individuals))):
                refined = self.compute_gradient_descent_step(elite_individuals[i], 30)
                if refined is not None:
                    population.append(refined)
            
            # Prevent stagnation with fallbacks
            if len(self.performance_history) > 10:
                recent_performance = self.performance_history[-10:]
                if abs(max(recent_performance) - min(recent_performance)) < 1e-8:
                    # Likely plateaued, introduce diversity
                    for i in range(min(5, len(population))):
                        population[i] = self.fallback_strategy(population[i])
            
            self.performance_history.append(current_best_fitness)
            
            self.iteration_count = iteration
            
        return self.best_sequence if self.best_sequence is not None else [random.random() for _ in range(100)]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    optimizer = AutocorrelationGradientEvolution()
    return optimizer.optimize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")