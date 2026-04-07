# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import differential_evolution
import time
from numba import jit, prange
import random
from typing import List, Tuple

# JIT compiled core functions for performance
@jit(nopython=True)
def compute_autoconvolution_fast(f: np.ndarray) -> np.ndarray:
    """Fast autoconvolution computation using numba."""
    n = len(f)
    g = np.zeros(2*n - 1, dtype=np.float64)
    
    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]
    
    return g[n-1:]  # Return positive lags only

@jit(nopython=True)
def compute_norms_fast(g: np.ndarray) -> Tuple[float, float, float]:
    """Fast norm computations using numba."""
    n = len(g)
    
    # Compute norms
    norm_1 = 0.0
    norm_2_sq = 0.0
    norm_inf = 0.0
    
    for i in range(n):
        abs_g = abs(g[i])
        norm_1 += abs_g
        norm_2_sq += abs_g * abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g
    
    return norm_1, norm_2_sq, norm_inf

@jit(nopython=True)
def compute_c2_fast(norm_1: float, norm_2_sq: float, norm_inf: float) -> float:
    """Fast C2 computation using numba."""
    if norm_1 < 1e-12 or norm_inf < 1e-12:
        return 0.0
    return norm_2_sq / (norm_1 * norm_inf)

class SpectralPeakConstructor:
    """Constructs functions in frequency domain with targeted peak properties."""
    
    @staticmethod
    def create_frequency_template(n_steps: int, seed: int) -> np.ndarray:
        """Create a frequency domain template that encourages good autoconvolution."""
        np.random.seed(seed)
        
        # Create complex frequency spectrum with strategic peaks
        spectrum = np.zeros(n_steps, dtype=complex)
        
        # Add fundamental component that creates overall structure
        spectrum[0] = 1.0  # DC component
        
        # Add multiple peaks that are logarithmically distributed
        # This targets different frequency scales to encourage rich autoconvolution
        
        # Logarithmic peak distribution
        log_min = np.log(1.0)  # Starting at 1 Hz
        log_max = np.log(10.0)  # Ending at 10 Hz (for 1000-point spectrum)
        num_peaks = 8
        
        peak_frequencies = np.logspace(log_min, log_max, num_peaks)
        
        # Add peaks with careful amplitude scaling
        for i, freq in enumerate(peak_frequencies):
            if freq < n_steps // 2:
                # Get integer frequency index
                idx = int(freq)
                if idx < n_steps // 2:
                    # Add peak with amplitude that decreases with frequency
                    amp = 1.0 / (1.0 + freq * 0.2)
                    # Add some randomness to peak characteristics
                    phase = np.random.random() * 2 * np.pi
                    spectrum[idx] = amp * np.exp(1j * phase)
                    
                    # Add conjugate pair for real output
                    if n_steps - idx < n_steps and idx != 0:
                        spectrum[n_steps - idx] = np.conj(spectrum[idx])
        
        # Ensure symmetry for real-valued output
        # DC component should be real
        if n_steps > 0:
            spectrum[0] = abs(spectrum[0])
        
        # Create the full symmetric spectrum
        for i in range(1, n_steps//2):
            if n_steps - i < n_steps:
                spectrum[n_steps - i] = np.conj(spectrum[i])
        
        # Convert to time domain
        try:
            f_time = np.real(ifft(spectrum))
        except:
            f_time = np.ones(n_steps) * 0.5
            
        return f_time
    
    @staticmethod
    def create_balanced_peaks(n_steps: int, seed: int) -> np.ndarray:
        """Create function with balanced, well-distributed peaks that avoid sharp interference."""
        np.random.seed(seed)
        
        # Create base frequency structure with logarithmic peak distribution
        base_spectrum = np.zeros(n_steps, dtype=complex)
        
        # Use logarithmic spacing for peak distribution
        num_peaks = 12
        log_freqs = np.logspace(np.log10(1.0), np.log10(n_steps/4), num_peaks)
        
        # Add peaks with decreasing amplitudes but strategic positions
        for i, freq in enumerate(log_freqs):
            if freq < n_steps // 2 and freq > 0:
                idx = int(freq)
                if idx < n_steps // 2:
                    # Amplitude decreases with frequency and peak index
                    amp = 1.0 / (1.0 + i * 0.2 + freq * 0.05)
                    # Add some randomness for diversity
                    amp *= (0.8 + np.random.random() * 0.4)
                    phase = np.random.random() * 2 * np.pi
                    
                    base_spectrum[idx] = amp * np.exp(1j * phase)
                    
                    # Conjugate pair
                    if n_steps - idx < n_steps:
                        base_spectrum[n_steps - idx] = np.conj(base_spectrum[idx])
        
        # Ensure DC component is real and appropriately scaled
        base_spectrum[0] = np.abs(base_spectrum[0])
        
        # Apply inverse FFT
        try:
            result = np.real(ifft(base_spectrum))
        except:
            result = np.ones(n_steps) * 0.5
            
        # Enforce non-negativity and normalization
        result = np.maximum(result, 0)
        if np.sum(result) > 0:
            result = result / np.sum(result) * 3.0  # Scale appropriately
            
        return result

class TimeLimitedOptimizer:
    """Implements time-aware optimization with proper termination checks."""
    
    def __init__(self, seed: int = 42, max_time: float = 90.0):
        self.seed = seed
        self.max_time = max_time
        self.start_time = time.time()
        
        np.random.seed(seed)
        random.seed(seed)
        
    def time_remaining(self) -> float:
        """Check remaining time."""
        return self.max_time - (time.time() - self.start_time)
    
    def is_expired(self) -> bool:
        """Check if time budget is exhausted."""
        return self.time_remaining() <= 0.1

class SpectralEvolutionOptimizer:
    """Implements an evolutionary strategy focused on spectral domain optimization."""
    
    def __init__(self, seed: int = 42, max_time: float = 90.0):
        self.optimizer = TimeLimitedOptimizer(seed, max_time)
        
    def fast_evaluate(self, f: np.ndarray) -> Tuple[float, np.ndarray]:
        """Fast evaluation using compiled functions."""
        try:
            # Fast autoconvolution
            g = compute_autoconvolution_fast(f)
            
            # Fast norm computations
            norm_1, norm_2_sq, norm_inf = compute_norms_fast(g)
            
            # C2 computation
            c2 = compute_c2_fast(norm_1, norm_2_sq, norm_inf)
            
            return c2, g
        except Exception:
            return 0.0, np.array([0.0])
    
    def generate_initial_population(self, n_steps: int, population_size: int) -> List[np.ndarray]:
        """Generate diverse initial spectrum-based functions."""
        population = []
        
        for i in range(population_size):
            if self.optimizer.time_remaining() < 1.0:
                break
                
            # Alternate between different construction strategies
            if i % 3 == 0:
                # Frequency template approach
                func = SpectralPeakConstructor.create_frequency_template(n_steps, self.optimizer.seed + i)
            elif i % 3 == 1:
                # Balanced peaks approach
                func = SpectralPeakConstructor.create_balanced_peaks(n_steps, self.optimizer.seed + i)
            else:
                # Hybrid approach
                func = np.ones(n_steps) * 0.5 + np.random.normal(0, 0.1, n_steps)
                func = np.maximum(func, 0)
                
            # Add mild noise for diversity
            noise_level = 0.005
            noisy_func = func + np.random.normal(0, noise_level, n_steps)
            noisy_func = np.maximum(noisy_func, 0)
            
            population.append(noisy_func)
            
        return population
    
    def local_refinement(self, f: np.ndarray, n_steps: int) -> np.ndarray:
        """Perform local refinement around good solution."""
        if self.optimizer.time_remaining() < 2.0:
            return f
            
        try:
            # Simple hill climbing on a subset of points
            refined_f = f.copy()
            max_iterations = min(30, int(self.optimizer.time_remaining() * 1.5))
            
            for iteration in range(max_iterations):
                if self.optimizer.time_remaining() < 0.5:
                    break
                    
                # Select random subset for refinement
                indices = np.random.choice(len(f), size=max(1, len(f) // 10), replace=False)
                
                for idx in indices:
                    old_val = refined_f[idx]
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.005)
                    refined_f[idx] += perturbation
                    refined_f[idx] = max(0, refined_f[idx])
            
            return refined_f
        except:
            return f
    
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        # Early exit if not enough time
        if self.optimizer.time_remaining() < 5.0:
            return [0.5] * 1000
        
        # Determine number of steps (larger for better resolution)
        n_steps = np.random.randint(3000, 7000)
        
        # Early exit if insufficient time
        if self.optimizer.time_remaining() < 10.0:
            return [0.5] * 1000
        
        # Initialize population
        population_size = min(30, max(5, int(self.optimizer.time_remaining() * 0.07)))
        population = self.generate_initial_population(n_steps, population_size)
        
        if not population:
            # Fallback to default
            return [0.5] * n_steps
            
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            c2, _ = self.fast_evaluate(individual)
            fitness_scores.append(c2)
        
        # Evolution loop
        max_generations = min(80, max(20, int(self.optimizer.time_remaining() * 0.15)))
        generation = 0
        
        while generation < max_generations and not self.optimizer.is_expired():
            # Select parent for reproduction (tournament selection)
            tournament_size = 3
            if len(population) >= tournament_size:
                parent_idx = np.random.choice(len(population), size=tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in parent_idx]
                selected_parent_idx = parent_idx[np.argmax(tournament_fitness)]
                selected_parent = population[selected_parent_idx]
            else:
                selected_parent = population[0]
            
            # Create offspring through mutation
            offspring = self.create_offspring(selected_parent)
            
            # Evaluate offspring
            offspring_c2, _ = self.fast_evaluate(offspring)
            
            # Replace worst individual if offspring is better
            if len(fitness_scores) > 0:
                worst_idx = np.argmin(fitness_scores)
                if offspring_c2 > fitness_scores[worst_idx]:
                    population[worst_idx] = offspring
                    fitness_scores[worst_idx] = offspring_c2
            
            # Local refinement on best individual periodically
            if generation % 4 == 0:
                best_individual = population[np.argmax(fitness_scores)] if len(population) > 0 else None
                if best_individual is not None:
                    refined = self.local_refinement(best_individual, n_steps)
                    refined_c2, _ = self.fast_evaluate(refined)
                    if refined_c2 > max(fitness_scores):
                        # Replace best with refined version
                        best_idx = np.argmax(fitness_scores)
                        population[best_idx] = refined
                        fitness_scores[best_idx] = refined_c2
                    
            generation += 1
        
        # Final selection
        if len(population) > 0:
            final_solution = population[np.argmax(fitness_scores)]
        else:
            final_solution = np.array([0.5] * n_steps)
        
        # Final refinement
        refined_final = self.local_refinement(final_solution, n_steps)
        final_c2, _ = self.fast_evaluate(refined_final)
        
        # Ensure final solution is valid
        result = np.maximum(refined_final, 0).tolist()
        
        # Add small noise for robustness
        noise_level = 0.003
        noisy_result = np.array(result) + np.random.normal(0, noise_level, len(result))
        noisy_result = np.maximum(noisy_result, 0)
        
        return noisy_result.tolist()
    
    def create_offspring(self, parent: np.ndarray) -> np.ndarray:
        """Create offspring by mutating a parent function."""
        try:
            # Convert to frequency domain
            spectrum = fft(parent)
            
            # Mutate the spectrum
            mutated_spectrum = self.mutate_spectrum(spectrum)
            
            # Convert back to time domain
            offspring = np.real(ifft(mutated_spectrum))
            offspring = np.maximum(offspring, 0)
            
            # Normalize if needed
            if np.sum(offspring) > 0:
                offspring = offspring / np.sum(offspring) * 5.0
                
            return offspring
        except:
            # Fallback to simple mutation if spectral operations fail
            offspring = parent.copy()
            for i in range(len(offspring)):
                if np.random.random() < 0.05:
                    offspring[i] += np.random.normal(0, 0.01)
                    offspring[i] = max(0, offspring[i])
            return offspring
    
    def mutate_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Mutate spectrum by modifying frequency components."""
        mutated_spectrum = spectrum.copy()
        
        # Select components to modify
        n_components = len(mutated_spectrum)
        n_mutations = max(1, int(n_components * 0.1))
        
        for _ in range(n_mutations):
            if np.random.random() < 0.7:  # 70% chance of modifying existing component
                # Modify an existing component
                idx = np.random.randint(0, n_components)
                # Apply scaling mutation
                scale_factor = np.random.uniform(0.8, 1.2)
                mutated_spectrum[idx] *= scale_factor
                
                # Maintain symmetry for real-valued output
                if idx != 0 and idx != n_components - idx and idx < n_components // 2:
                    mutated_spectrum[n_components - idx] = np.conj(mutated_spectrum[idx])
            else:
                # Add new component
                if len(mutated_spectrum) > 0:
                    idx = np.random.randint(1, len(mutated_spectrum) // 2)  # Avoid DC and Nyquist
                    amplitude = np.random.uniform(0.1, 1.5)
                    phase = np.random.random() * 2 * np.pi
                    mutated_spectrum[idx] = amplitude * np.exp(1j * phase)
                    if idx != len(mutated_spectrum) - idx:
                        mutated_spectrum[len(mutated_spectrum) - idx] = np.conj(mutated_spectrum[idx])
        
        # Ensure DC component remains real
        if len(mutated_spectrum) > 0:
            mutated_spectrum[0] = abs(mutated_spectrum[0])
            
        return mutated_spectrum

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    optimizer = SpectralEvolutionOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")