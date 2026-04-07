# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time
from collections import deque
from scipy.ndimage import gaussian_filter1d

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class SpectralAutoconvolutionEvaluator:
    """Handles all autoconvolution norm computations with optimized numerical methods"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the autoconvolution g = f*f and its norms efficiently.
        Returns (||g||₂², ||g||₁, ||g||∞)
        """
        if not f_values or len(f_values) < 2:
            return 0.0, 0.0, 0.0

        # Create step function on [-1/4, 1/4] with equal spacing
        n = len(f_values)
        
        # Step size in x domain [-1/4, 1/4]
        dx = 0.5 / (n - 1) if n > 1 else 0.5

        # Compute autoconvolution using numpy's convolution
        g = signal.convolve(f_values, f_values, mode='full')
        
        # Extract the central portion representing the actual convolution on [-1/2, 1/2]
        # For two functions of length n on [-1/4, 1/4], convolution produces 2*n-1 points
        center_start = len(g) // 2 - (n - 1)
        center_end = center_start + (2 * n - 1)
        g = g[center_start:center_end]

        # Compute the three norms
        # ||g||∞ = max of |g|
        norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0

        # ||g||₁ = sum of |g| * dx
        norm_1 = np.sum(np.abs(g)) * dx if len(g) > 1 else 0.0

        # ||g||₂² = ∫ g² dx using trapezoidal-like integration
        if len(g) <= 1:
            norm_2_squared = 0.0
        else:
            # Use piecewise linear integration for g^2
            # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
            norm_2_squared = 0.0
            for i in range(len(g)-1):
                y1, y2 = g[i], g[i+1]
                norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)

        return norm_2_squared, norm_1, norm_inf

    @classmethod
    def compute_c2(cls, f_values: List[float]) -> float:
        """Compute the C2 value for given step function."""
        norm_2_squared, norm_1, norm_inf = cls.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2

class PeakBasedFunctionGenerator:
    """Generates functions based on strategically placed peaks in frequency domain"""
    
    def __init__(self):
        self.min_length = 200
        self.max_length = 2000
        
    def generate_peak_parameters(self, n_peaks: int = None) -> List[Tuple[float, float, float]]:
        """Generate peak parameters (position, amplitude, width) for Gaussian peaks"""
        if n_peaks is None:
            n_peaks = np.random.randint(3, 15)
            
        peaks = []
        for i in range(n_peaks):
            # Vary peak widths to produce different autoconvolution profiles
            peak_width = np.random.uniform(0.01, 0.06)
            
            # Position peaks more strategically in the middle of domain to avoid edge effects
            # Distribute them logarithmically to cover different scales
            if i == 0:
                peak_position = 0.0  # Center peak
            else:
                # Distribute log-uniformly across the space, avoiding very near center
                log_min = np.log(0.05)  # Avoiding too close to center
                log_max = np.log(0.3)   # Avoiding too far from center
                log_pos = np.random.uniform(log_min, log_max)
                peak_position = np.random.choice([-1, 1]) * np.exp(log_pos)
            
            # Amplitude based on inverse distance from center to encourage flatness
            distance_from_center = abs(peak_position)
            if distance_from_center < 0.05:
                peak_amplitude = np.random.uniform(1.0, 3.0)
            elif distance_from_center < 0.15:
                peak_amplitude = np.random.uniform(0.8, 2.0)
            else:
                peak_amplitude = np.random.uniform(0.5, 1.5)
            
            peaks.append((peak_position, peak_amplitude, peak_width))
        
        return peaks
    
    def create_function_from_peaks(self, peak_params: List[Tuple[float, float, float]], 
                                 length: int = None) -> List[float]:
        """Create function from peak parameters"""
        if length is None:
            length = np.random.randint(self.min_length, self.max_length)
            
        x = np.linspace(-0.25, 0.25, length)
        f_values = np.zeros(length)
        
        for pos, amp, width in peak_params:
            # Create Gaussian peak
            gaussian = amp * np.exp(-0.5 * ((x - pos) / width)**2)
            f_values += gaussian
            
        # Ensure non-negative values
        f_values = np.clip(f_values, 0, None)
        
        # Apply gentle smoothing to avoid sharp transitions that may hurt autoconvolution
        if length > 100:
            f_values = gaussian_filter1d(f_values, sigma=max(0.5, length/200.0))
        
        # Normalize to reasonable magnitude
        if np.max(f_values) > 1e-6:
            f_values = f_values / np.max(f_values) * 2.0
            
        return f_values.tolist()

class MultiScalePeakOptimizer:
    """Optimizes peak parameters using multi-scale approach"""
    
    def __init__(self, population_size: int = 40, max_generations: int = 150):
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_size = max(1, population_size // 4)
        self.generator = PeakBasedFunctionGenerator()
        
    def generate_initial_population(self) -> List[List[Tuple[float, float, float]]]:
        """Create diverse initial population of peak parameter sets"""
        population = []
        
        # Strategy 1: Balanced approach with varied peak counts and parameters
        for _ in range(self.population_size // 2):
            n_peaks = np.random.randint(3, 15)
            peak_params = self.generator.generate_peak_parameters(n_peaks)
            population.append(peak_params)
        
        # Strategy 2: Sparse approach for exploration
        for _ in range(self.population_size // 4):
            n_peaks = np.random.randint(1, 4)  # Fewer peaks for sparse exploration
            peak_params = self.generator.generate_peak_parameters(n_peaks)
            population.append(peak_params)
            
        # Strategy 3: Dense approach for exploitation
        for _ in range(self.population_size // 4):
            n_peaks = np.random.randint(10, 25)  # More peaks for dense representation
            peak_params = self.generator.generate_peak_parameters(n_peaks)
            population.append(peak_params)
            
        return population
    
    def crossover_peaks(self, parent1: List[Tuple[float, float, float]], 
                       parent2: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], 
                                                                          List[Tuple[float, float, float]]]:
        """Crossover two peak parameter sets to create offspring"""
        # Combine peaks from both parents with some randomization
        combined = parent1 + parent2
        np.random.shuffle(combined)
        
        # Select subset for children
        split_point = len(combined) // 2
        child1 = combined[:split_point]
        child2 = combined[split_point:]
        
        return child1, child2
    
    def mutate_peak_parameters(self, peak_params: List[Tuple[float, float, float]], 
                              mutation_rate: float = 0.2, generation: int = 0) -> List[Tuple[float, float, float]]:
        """Mutate peak parameters with adaptive strategy"""
        mutated = []
        
        for pos, amp, width in peak_params:
            if random.random() < mutation_rate:
                # Mutate position
                if random.random() < 0.5:
                    pos += np.random.normal(0, 0.01) * (0.5 if generation < 50 else 0.2)
                    pos = np.clip(pos, -0.24, 0.24)  # Keep in valid range
                    
                # Mutate amplitude
                if random.random() < 0.5:
                    amp *= np.random.uniform(0.8, 1.2)
                    amp = max(0.1, amp)  # Ensure non-negative
                    
                # Mutate width
                if random.random() < 0.5:
                    width *= np.random.uniform(0.7, 1.3)
                    width = np.clip(width, 0.005, 0.1)  # Keep reasonable range
                    
            mutated.append((pos, amp, width))
            
        # Occasionally add/remove peaks for structural diversity
        if random.random() < 0.1 and len(mutated) > 1:
            # Remove a peak
            mutated.pop(random.randint(0, len(mutated)-1))
        elif random.random() < 0.1 and len(mutated) < 30:  # Don't exceed limit
            # Add a new peak
            new_peak = self.generator.generate_peak_parameters(1)[0]
            mutated.append(new_peak)
            
        return mutated
    
    def evaluate_population(self, peak_populations: List[List[Tuple[float, float, float]]]) -> Tuple[List[List[float]], np.ndarray]:
        """Convert peak populations to functions and evaluate their C2 values"""
        functions = []
        fitnesses = []
        
        for peak_params in peak_populations:
            # Convert to function
            try:
                func = self.generator.create_function_from_peaks(peak_params)
                functions.append(func)
                c2 = SpectralAutoconvolutionEvaluator.compute_c2(func)
                fitnesses.append(c2)
            except Exception:
                functions.append([0.0] * np.random.randint(100, 1000))
                fitnesses.append(0.0)
                
        return functions, np.array(fitnesses)
    
    def optimize(self, max_time_seconds: int = 85) -> List[float]:
        """Main optimization routine"""
        start_time = time.time()
        
        # Initialize population
        peak_population = self.generate_initial_population()
        functions, fitnesses = self.evaluate_population(peak_population)
        
        # Track best solution
        best_fitness = np.max(fitnesses)
        best_peak_params = peak_population[np.argmax(fitnesses)].copy()
        best_function = functions[np.argmax(fitnesses)].copy()
        recent_improvements = deque(maxlen=10)
        recent_improvements.append(best_fitness)
        
        # Evolution loop
        generation = 0
        stagnation_counter = 0
        max_stagnation = 40
        last_best_fitness = 0.0
        
        while generation < self.max_generations and (time.time() - start_time) < max_time_seconds - 1:
            # Update best solution
            current_best_idx = np.argmax(fitnesses)
            current_fitness = fitnesses[current_best_idx]
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_peak_params = peak_population[current_best_idx].copy()
                best_function = functions[current_best_idx].copy()
                recent_improvements.append(current_fitness)
            
            # Check for stagnation
            if abs(current_fitness - last_best_fitness) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = current_fitness
                
            # Early stop if stagnating too much
            if stagnation_counter >= max_stagnation:
                break
            
            # Selection and reproduction with elitism
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [peak_population[i] for i in sorted_indices[:self.elite_size]]
            
            # Generate offspring using tournament selection and crossover
            offspring = []
            tournament_size = 3
            
            while len(offspring) < self.population_size - self.elite_size:
                # Tournament selection
                tournament_indices = random.sample(range(len(peak_population)), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                
                tournament_indices = random.sample(range(len(peak_population)), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitnesses)]
                
                # Crossover
                child1, child2 = self.crossover_peaks(
                    peak_population[parent1_idx], 
                    peak_population[parent2_idx]
                )
                
                # Mutation
                child1 = self.mutate_peak_parameters(child1, generation=generation)
                child2 = self.mutate_peak_parameters(child2, generation=generation)
                
                offspring.extend([child1, child2])
            
            # Combine elites and offspring
            peak_population = elites + offspring[:self.population_size - self.elite_size]
            
            # Evaluate new population
            functions, fitnesses = self.evaluate_population(peak_population)
            
            # Adjust adaptive parameters based on progress
            if generation > 100:
                mutation_rate = 0.15 + 0.05 * random.random()
            else:
                mutation_rate = 0.2 + 0.1 * random.random()
            
            generation += 1
        
        # Final local search refinement
        if best_function is not None and best_fitness > 0.9:
            # Perform a focused local refinement around the best peak configuration
            refined_peak_params = self.mutate_peak_parameters(best_peak_params, 
                                                           mutation_rate=0.02, 
                                                           generation=generation)
            refined_function = self.generator.create_function_from_peaks(refined_peak_params)
            refined_c2 = SpectralAutoconvolutionEvaluator.compute_c2(refined_function)
            
            if refined_c2 > best_fitness:
                best_function = refined_function
                
        return best_function if best_function is not None else []

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - entry point"""
    try:
        optimizer = MultiScalePeakOptimizer(population_size=40, max_generations=150)
        f_values = optimizer.optimize(max_time_seconds=85)
        return f_values
    except Exception as e:
        # Fallback to random generation if anything fails
        print(f"Error in optimization: {e}")
        f_values = [np.random.random()] * np.random.randint(100, 1000)
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")