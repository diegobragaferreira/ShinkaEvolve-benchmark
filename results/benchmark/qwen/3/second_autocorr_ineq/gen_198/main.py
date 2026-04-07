# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum

class OptimizationStrategy(Enum):
    GRADIENT_BASED = "gradient"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"

@dataclass
class OptimizationResult:
    best_function: List[float]
    best_c2: float
    eval_time: float
    benchmark_ratio: float

class NormCalculator:
    """Handles all norm computations for C2 calculation"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """Compute the three norms needed for C2 calculation"""
        # Create step function with appropriate spacing
        n = len(f_values)
        if n == 0:
            return 0.0, 0.0, 0.0

        # Generate step positions [-1/4, 1/4] - these are the left edges of steps
        dx = 0.5 / n  # Width of each step

        # Convert to numpy array and ensure non-negative
        f_array = np.array(f_values, dtype=np.float64)
        f_array = np.maximum(f_array, 0.0)  # Clip negative values

        # Manual computation of the convolution sum for step functions
        g = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                # In convolution, the value at index i+j comes from f[i] * f[j]
                g[i + j] += f_array[i] * f_array[j]

        # Scale by step width for proper normalization
        g = g * dx

        # Compute norms
        g_abs = np.abs(g)

        # L2 norm squared (sum of squares times dx)
        g2_squared = np.sum(g_abs**2) * dx

        # L1 norm (sum of absolute values times dx)
        g1 = np.sum(g_abs) * dx

        # L-infinity norm
        g_inf = np.max(g_abs)

        return g2_squared, g1, g_inf

    @staticmethod
    def calculate_c2(f_values: List[float]) -> float:
        """Calculate C2 value for given step function"""
        g2_squared, g1, g_inf = NormCalculator.compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if g1 <= 1e-15 or g_inf <= 1e-15:
            return 0.0

        return g2_squared / (g1 * g_inf)

class FunctionBuilder:
    """Creates and manipulates step function representations"""
    
    @staticmethod
    def create_uniform_function(n: int) -> List[float]:
        """Create a uniform step function"""
        return [1.0] * n
    
    @staticmethod
    def create_peak_function(n: int, num_peaks: int = 3) -> List[float]:
        """Create a function with strategic peaks"""
        func = [0.0] * n
        for i in range(num_peaks):
            pos = int((i + 1) * n / (num_peaks + 1))
            func[pos] = 1.0 + random.random() * 2.0
        return func
    
    @staticmethod
    def create_bump_function(n: int, num_bumps: int = 5) -> List[float]:
        """Create a function with multiple bumps"""
        func = np.zeros(n)
        for i in range(num_bumps):
            center = int(random.random() * n)
            width = max(1, int(random.random() * n / 10))
            amplitude = 0.5 + random.random() * 1.5
            for j in range(max(0, center - width), min(n, center + width)):
                func[j] += amplitude * np.exp(-0.5 * ((j - center) / width)**2)
        return func.tolist()
    
    @staticmethod
    def create_adaptive_initial(n: int) -> List[float]:
        """Create an adaptive initial function based on mathematical heuristics"""
        # Create a function that combines peaks and smooth transitions
        func = np.zeros(n)
        
        # Add several peaks with different characteristics
        num_peaks = max(3, min(10, n // 20))
        for i in range(num_peaks):
            center = int(random.random() * n)
            width = max(1, int(random.random() * n / 15))
            amplitude = 0.5 + random.random() * 2.0
            
            # Create a Gaussian-like bump
            x = np.arange(n)
            gaussian = amplitude * np.exp(-0.5 * ((x - center) / width)**2)
            func += gaussian
            
        # Ensure all values are non-negative
        func = np.maximum(func, 0.0)
        
        # Add some random noise to avoid getting stuck in local minima
        noise = np.random.normal(0, 0.05, n)
        func = np.maximum(func + noise, 0.0)
        
        return func.tolist()

class LocalSearchOptimizer:
    """Handles various local search refinement strategies"""
    
    @staticmethod
    def finite_difference_gradient(f_values: List[float], epsilon: float = 1e-4) -> List[float]:
        """Estimate gradient using finite differences"""
        n = len(f_values)
        if n == 0:
            return [0.0] * n
            
        grad = np.zeros(n)
        base_c2 = NormCalculator.calculate_c2(f_values)
        
        for i in range(n):
            # Create perturbed versions
            f_plus = f_values.copy()
            f_minus = f_values.copy()
            
            f_plus[i] = max(0.0, f_values[i] + epsilon)
            f_minus[i] = max(0.0, f_values[i] - epsilon)
            
            plus_c2 = NormCalculator.calculate_c2(f_plus)
            minus_c2 = NormCalculator.calculate_c2(f_minus)
            
            grad[i] = (plus_c2 - minus_c2) / (2 * epsilon)
            
        return grad.tolist()
    
    @staticmethod
    def coordinate_wise_refinement(f_values: List[float], max_iter: int = 50) -> List[float]:
        """Refine function using coordinate-wise updates"""
        current = np.array(f_values, dtype=np.float64)
        n = len(current)
        
        if n == 0:
            return f_values
        
        best_c2 = NormCalculator.calculate_c2(f_values)
        best_solution = current.copy()
        
        # Simple gradient descent with adaptive step size
        for _ in range(max_iter):
            # Estimate gradient
            grad = LocalSearchOptimizer.finite_difference_gradient(current.tolist())
            grad = np.array(grad)
            
            # Adaptive step size based on gradient magnitude
            grad_mag = np.linalg.norm(grad)
            if grad_mag > 1e-10:
                step_size = 0.01 / (1.0 + grad_mag)
            else:
                step_size = 0.01
                
            # Update solution
            new_solution = current - step_size * grad
            new_solution = np.maximum(new_solution, 0.0)
            
            new_c2 = NormCalculator.calculate_c2(new_solution.tolist())
            
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_solution = new_solution.copy()
                current = new_solution.copy()
        
        return best_solution.tolist()
    
    @staticmethod
    def differential_evolution_refinement(f_values: List[float], max_iter: int = 20) -> List[float]:
        """Use differential evolution for local refinement"""
        try:
            # Convert to numpy array
            solution_array = np.array(f_values)

            # Define bounds for each parameter
            bounds = [(0.0, 3.0) for _ in range(len(solution_array))]

            # Objective function for local search
            def objective(x):
                return -NormCalculator.calculate_c2(x.tolist())  # Negative because we minimize

            # Use differential evolution for local refinement
            result = differential_evolution(objective, bounds, maxiter=max_iter//5,
                                          popsize=10, seed=42, disp=False)

            # Return refined solution with non-negative values
            refined = np.maximum(result.x, 0)
            return refined.tolist()
        except:
            # If local search fails, return original solution
            return f_values

class AdaptiveGeneticOptimizer:
    """Main evolutionary optimization class with adaptive strategies"""
    
    def __init__(self, pop_size: int = 20, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = 2
        self.mutation_rate = 0.3
        self.final_refinement_generations = 10
    
    def initialize_population(self) -> List[List[float]]:
        """Initialize diverse population using adaptive strategies"""
        population = []
        
        # Add structured initial functions
        # 1. Uniform function
        population.append(FunctionBuilder.create_uniform_function(500))
        
        # 2. Peak function
        population.append(FunctionBuilder.create_peak_function(500))
        
        # 3. Bump function
        population.append(FunctionBuilder.create_bump_function(500))
        
        # 4. Adaptive initial
        population.append(FunctionBuilder.create_adaptive_initial(500))
        
        # Fill rest with random variations
        for _ in range(self.pop_size - 4):
            n = random.randint(200, 1000)
            # Use a mix of initialization strategies
            strategy = random.choice([0, 1, 2, 3])
            if strategy == 0:
                individual = FunctionBuilder.create_uniform_function(n)
            elif strategy == 1:
                individual = FunctionBuilder.create_peak_function(n)
            elif strategy == 2:
                individual = FunctionBuilder.create_bump_function(n)
            else:
                individual = FunctionBuilder.create_adaptive_initial(n)
            
            # Add some noise
            individual = [max(0.0, x + random.gauss(0, 0.05)) for x in individual]
            
            population.append(individual)
            
        return population
    
    def select_parents(self, population: List[List[float]], fitnesses: List[float]) -> List[List[float]]:
        """Tournament selection for parent selection"""
        selected = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = min(3, len(population))
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index].copy())
        return selected
    
    def adaptive_crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Create offspring through adaptive crossover"""
        n1, n2 = len(parent1), len(parent2)
        n = max(n1, n2)

        # Adaptive crossover - use different strategies based on similarity
        if abs(n1 - n2) / max(n1, n2) > 0.2:  # Different sizes
            # Use the longer parent as main template
            longer, shorter = (parent1, parent2) if n1 > n2 else (parent2, parent1)
            offspring = longer.copy()
            # Fill gaps with shorter parent data
            for i in range(min(n1, n2)):
                if i < len(shorter):
                    offspring[i] = shorter[i]
        else:
            # Uniform crossover
            offspring = []
            for i in range(n):
                if i < n1 and i < n2 and random.random() < 0.5:
                    offspring.append(parent1[i])
                elif i < n1:
                    offspring.append(parent1[i])
                elif i < n2:
                    offspring.append(parent2[i])
                else:
                    offspring.append(0.0)
        
        return offspring
    
    def adaptive_mutation(self, individual: List[float], generation: int) -> List[float]:
        """Apply adaptive mutation with dynamic rate"""
        child = individual.copy()
        n = len(child)

        if n == 0:
            return child

        # Adaptive mutation rate that decreases over generations
        initial_mutation_rate = 0.3
        final_mutation_rate = 0.05
        mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (generation / self.generations)
        mutation_rate = max(final_mutation_rate, mutation_rate)

        # Apply Gaussian noise to each element
        for i in range(n):
            # Add Gaussian noise scaled by adaptive rate and current value
            noise = np.random.normal(0, mutation_rate * max(1e-6, child[i]))
            child[i] = max(0, child[i] + noise)  # Keep non-negative

        return child
    
    def hybrid_refinement(self, solution: List[float]) -> List[float]:
        """Apply hybrid refinement strategies"""
        # Try coordinate-wise refinement first
        refined = LocalSearchOptimizer.coordinate_wise_refinement(solution)
        c2_refined = NormCalculator.calculate_c2(refined)
        
        # Then try differential evolution
        de_refined = LocalSearchOptimizer.differential_evolution_refinement(solution)
        c2_de = NormCalculator.calculate_c2(de_refined)
        
        # Return better of the two
        return refined if c2_refined > c2_de else de_refined
    
    def optimize(self) -> OptimizationResult:
        """Main optimization loop"""
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        
        best_fitness_history = []
        best_fitness = -float('inf')
        best_individual = None
        no_improvement_count = 0
        patience_limit = 15
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [NormCalculator.calculate_c2(individual) for individual in population]

            # Track best fitness
            current_best = max(fitnesses)
            best_fitness_history.append(current_best)

            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[fitnesses.index(current_best)].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Print progress every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: Best C2 = {best_fitness:.6f}")

            # Early stopping condition
            if no_improvement_count >= patience_limit:
                print(f"Early stopping at generation {gen} due to no improvement")
                break

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]

            # Keep elite
            elite = population[:self.elite_size]

            # Apply hybrid refinement to elite in final generations
            if gen >= self.generations - self.final_refinement_generations:
                for i in range(len(elite)):
                    if i < len(elite):  # Safety check
                        refined = self.hybrid_refinement(elite[i])
                        refined_fitness = NormCalculator.calculate_c2(refined)
                        if refined_fitness > NormCalculator.calculate_c2(elite[i]):
                            elite[i] = refined

            # Select parents
            parents = self.select_parents(population, fitnesses)

            # Create new population through crossover and mutation
            new_population = elite.copy()

            while len(new_population) < self.pop_size:
                # Select two parents
                p1_idx, p2_idx = random.sample(range(len(parents)), 2)
                p1, p2 = parents[p1_idx], parents[p2_idx]

                # Crossover
                offspring = self.adaptive_crossover(p1, p2)

                # Mutation with adaptive rate
                mutated_offspring = self.adaptive_mutation(offspring, gen)

                new_population.append(mutated_offspring)

            population = new_population[:self.pop_size]

        # Final evaluation
        final_fitnesses = [NormCalculator.calculate_c2(individual) for individual in population]
        best_final_individual = population[np.argmax(final_fitnesses)]
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Calculate benchmark ratio
        benchmark_ratio = NormCalculator.calculate_c2(best_final_individual) / 0.962
        
        return OptimizationResult(
            best_function=best_final_individual,
            best_c2=NormCalculator.calculate_c2(best_final_individual),
            eval_time=eval_time,
            benchmark_ratio=benchmark_ratio
        )

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    try:
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Create and run optimizer
        optimizer = AdaptiveGeneticOptimizer(pop_size=20, generations=50)
        result = optimizer.optimize()
        
        return result.best_function
    except Exception as e:
        # Fallback to basic approach if optimization fails
        print(f"Optimization failed with error: {e}, using fallback")
        return [1.0] * 500  # Return simple uniform function as fallback

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")