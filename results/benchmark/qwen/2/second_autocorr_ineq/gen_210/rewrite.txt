# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import differential_evolution
import random
import time
from numba import jit, prange
from typing import List, Tuple

# Core JIT-compiled functions for performance
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

class SpectralFunctionGenerator:
    """Generates functions in spectral domain that promote good autoconvolution properties."""
    
    @staticmethod
    def generate_spectral_template(n_steps: int, seed: int) -> np.ndarray:
        """Generate a spectral template designed to produce favorable autoconvolution."""
        np.random.seed(seed)
        
        # Create a frequency domain representation with specific characteristics
        # This template encourages smooth, well-behaved autoconvolution results
        frequencies = fftfreq(n_steps, 0.5/n_steps)
        
        # Generate a complex spectrum with structured peaks
        spectrum = np.zeros(n_steps, dtype=complex)
        
        # Add several strategically placed peaks in frequency domain
        # These peaks are chosen to promote good properties in time domain
        peak_frequencies = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        peak_amplitudes = [1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Create peaks with some randomness to avoid overly structured functions
        for freq, amp in zip(peak_frequencies, peak_amplitudes):
            # Convert frequency to index
            if freq == 0.0:
                # DC component
                spectrum[0] = amp
            else:
                # Find closest frequency bin
                idx = int(freq * n_steps / 0.5)
                if 0 < idx < n_steps//2:
                    # Add peaks with some frequency variation
                    freq_variation = np.random.uniform(-0.5, 0.5)
                    actual_idx = max(1, min(n_steps//2 - 1, idx + int(freq_variation)))
                    spectrum[actual_idx] = amp * (0.8 + np.random.random() * 0.4)
                    if actual_idx != n_steps - actual_idx:
                        spectrum[n_steps - actual_idx] = np.conj(spectrum[actual_idx])
        
        # Ensure symmetry for real-valued output
        # DC component should be real
        if n_steps > 0:
            spectrum[0] = abs(spectrum[0]).real
            
        # Symmetrize the spectrum
        for i in range(1, n_steps//2):
            if n_steps - i < n_steps:
                spectrum[n_steps - i] = np.conj(spectrum[i])
        
        # Inverse FFT to get time domain function
        try:
            f = np.real(ifft(spectrum))
        except:
            # Fallback to simple structure if FFT fails
            f = np.ones(n_steps) * 0.5
            
        return f
    
    @staticmethod
    def generate_better_template(n_steps: int, seed: int) -> np.ndarray:
        """Generate a more sophisticated template with improved spectral characteristics."""
        np.random.seed(seed)
        
        # Start with a base spectrum that has good autoconvolution properties
        base_spectrum = np.zeros(n_steps, dtype=complex)
        
        # Create a richer set of spectral components
        # Use a combination of sinusoids and exponential decays
        x_freq = np.arange(n_steps) / n_steps
        
        # Add multiple frequency bands with decreasing amplitudes
        # This promotes energy spread and reduces peakiness
        for i in range(1, min(20, n_steps//2)):
            # Frequency spacing: logarithmically increasing
            freq = i * 1.5
            # Amplitude decay: exponential with a random factor
            amp = 0.8 * np.exp(-0.05 * i) * (0.7 + np.random.random() * 0.6)
            
            # Add both positive and negative frequencies
            if i < n_steps//2:
                base_spectrum[i] = amp * np.exp(2j * np.pi * np.random.random())  # Complex phase
                if n_steps - i < n_steps:
                    base_spectrum[n_steps - i] = np.conj(base_spectrum[i])
        
        # Ensure DC component is real
        base_spectrum[0] = np.abs(base_spectrum[0])
        
        # Apply inverse FFT
        try:
            f = np.real(ifft(base_spectrum))
        except:
            f = np.ones(n_steps) * 0.5
            
        # Enforce non-negativity
        f = np.maximum(f, 0)
        
        # Normalize if possible
        if np.sum(f) > 0:
            f = f / np.sum(f) * 5.0
            
        return f

class SpectralEvolutionOptimizer:
    """Evolutionary optimizer working in spectral domain."""
    
    def __init__(self, seed: int = 42, max_time: float = 90.0):
        self.seed = seed
        self.max_time = max_time
        self.start_time = time.time()
        
        np.random.seed(seed)
        random.seed(seed)
        
    def time_remaining(self) -> float:
        """Check remaining time."""
        return self.max_time - (time.time() - self.start_time)
    
    def _evaluate_function(self, f: np.ndarray) -> Tuple[float, np.ndarray]:
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
    
    def _generate_initial_population(self, n_steps: int, population_size: int) -> List[np.ndarray]:
        """Generate initial population using spectral templates."""
        population = []
        
        # Generate diverse initial spectra
        for i in range(population_size):
            if self.time_remaining() < 1.0:
                break
                
            template = SpectralFunctionGenerator.generate_better_template(n_steps, self.seed + i)
            
            # Add some random noise to introduce variation
            noise_level = 0.01
            noisy_template = template + np.random.normal(0, noise_level, n_steps) * template
            noisy_template = np.maximum(noisy_template, 0)
            
            population.append(noisy_template)
            
        return population
    
    def _mutate_spectrum(self, spectrum: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutate spectrum by modifying frequency components.""" 
        mutated_spectrum = spectrum.copy()
        
        # Apply mutations to a fraction of components
        n_components = len(mutated_spectrum)
        n_mutations = max(1, int(n_components * mutation_rate))
        
        for _ in range(n_mutations):
            if np.random.random() < 0.7:  # 70% chance of modifying existing component
                # Modify an existing component
                idx = np.random.randint(0, n_components)
                # Apply small random change
                mutation_strength = np.random.uniform(0.5, 1.5)
                mutated_spectrum[idx] *= mutation_strength
                
                # Maintain symmetry for real-valued output
                if idx != 0 and idx != n_components - idx and idx < n_components // 2:
                    mutated_spectrum[n_components - idx] = np.conj(mutated_spectrum[idx])
            else:
                # Add new component
                if len(mutated_spectrum) > 0:
                    idx = np.random.randint(1, len(mutated_spectrum) // 2)  # Avoid DC and Nyquist
                    amplitude = np.random.uniform(0.1, 2.0)
                    mutated_spectrum[idx] = amplitude * np.exp(1j * np.random.random() * 2 * np.pi)
                    if idx != len(mutated_spectrum) - idx:
                        mutated_spectrum[len(mutated_spectrum) - idx] = np.conj(mutated_spectrum[idx])
        
        # Ensure DC component remains real
        if len(mutated_spectrum) > 0:
            mutated_spectrum[0] = abs(mutated_spectrum[0])
            
        return mutated_spectrum
    
    def _create_offspring(self, parent: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Create offspring by mutating a parent function."""
        # Convert to frequency domain
        try:
            spectrum = fft(parent)
            
            # Mutate the spectrum
            mutated_spectrum = self._mutate_spectrum(spectrum, mutation_rate)
            
            # Convert back to time domain
            offspring = np.real(ifft(mutated_spectrum))
            offspring = np.maximum(offspring, 0)
            
            # Normalize if needed
            if np.sum(offspring) > 0:
                offspring = offspring / np.sum(offspring) * 10
                
            return offspring
        except:
            # Fallback to simple mutation if spectral operations fail
            offspring = parent.copy()
            for i in range(len(offspring)):
                if np.random.random() < 0.05:
                    offspring[i] += np.random.normal(0, 0.02)
                    offspring[i] = max(0, offspring[i])
            return offspring
    
    def _select_best(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Select the best individual from population."""
        if not population:
            return np.ones(1000) * 0.5
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
    
    def _local_refinement(self, f: np.ndarray, n_steps: int) -> np.ndarray:
        """Perform local refinement around good solution."""
        if self.time_remaining() < 2.0:
            return f
            
        try:
            # Simple hill climbing on a subset of points
            refined_f = f.copy()
            max_iterations = min(30, int(self.time_remaining() * 1.5))
            
            for iteration in range(max_iterations):
                if self.time_remaining() < 0.5:
                    break
                    
                # Select random subset for refinement
                indices = np.random.choice(len(f), size=max(1, len(f) // 20), replace=False)
                
                for idx in indices:
                    old_val = refined_f[idx]
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.01)
                    refined_f[idx] += perturbation
                    refined_f[idx] = max(0, refined_f[idx])
            
            return refined_f
        except:
            return f
    
    def construct_function(self) -> List[float]:
        """Main function to construct step-function with high C2 value."""
        # Early exit if not enough time
        if self.time_remaining() < 5.0:
            return [0.5] * 1000
        
        # Determine number of steps
        n_steps = np.random.randint(2000, 8000)
        
        # Early exit if insufficient time
        if self.time_remaining() < 10.0:
            return [0.5] * 1000
        
        # Initialize population
        population_size = min(20, max(4, int(self.time_remaining() * 0.05)))
        population = self._generate_initial_population(n_steps, population_size)
        
        if not population:
            # Fallback to default
            return [0.5] * n_steps
            
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            c2, _ = self._evaluate_function(individual)
            fitness_scores.append(c2)
        
        # Evolution loop
        max_generations = min(50, max(10, int(self.time_remaining() * 0.1)))
        generation = 0
        
        while generation < max_generations and self.time_remaining() > 5.0:
            # Select parent for reproduction (tournament selection)
            tournament_size = 3
            parent_idx = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in parent_idx]
            selected_parent_idx = parent_idx[np.argmax(tournament_fitness)]
            selected_parent = population[selected_parent_idx]
            
            # Create offspring through mutation
            offspring = self._create_offspring(selected_parent, mutation_rate=0.15)
            
            # Evaluate offspring
            offspring_c2, _ = self._evaluate_function(offspring)
            
            # Replace worst individual if offspring is better
            worst_idx = np.argmin(fitness_scores)
            if offspring_c2 > fitness_scores[worst_idx]:
                population[worst_idx] = offspring
                fitness_scores[worst_idx] = offspring_c2
            
            # Local refinement on best individual periodically
            if generation % 5 == 0:
                best_individual = self._select_best(population, fitness_scores)
                refined = self._local_refinement(best_individual, n_steps)
                refined_c2, _ = self._evaluate_function(refined)
                if refined_c2 > max(fitness_scores):
                    # Replace best with refined version
                    best_idx = np.argmax(fitness_scores)
                    population[best_idx] = refined
                    fitness_scores[best_idx] = refined_c2
                    
            generation += 1
        
        # Final selection
        final_solution = self._select_best(population, fitness_scores)
        
        # Final refinement
        refined_final = self._local_refinement(final_solution, n_steps)
        final_c2, _ = self._evaluate_function(refined_final)
        
        # Ensure final solution is valid
        result = np.maximum(refined_final, 0).tolist()
        
        # Add small noise for robustness
        noise_level = 0.005
        noisy_result = np.array(result) + np.random.normal(0, noise_level, len(result))
        noisy_result = np.maximum(noisy_result, 0)
        
        return noisy_result.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    optimizer = SpectralEvolutionOptimizer()
    return optimizer.construct_function()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")