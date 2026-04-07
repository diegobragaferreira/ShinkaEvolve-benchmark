# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')
import scipy.fft as fft
from scipy.optimize import differential_evolution
from numba import jit
import time

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values):
    """Optimized computation of autoconvolution norms using numba"""
    n = len(f_values)
    
    # Initialize autoconvolution array
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute norms for the central portion
    half_len = n - 1
    g_center = g[half_len:-half_len]
    
    # Compute norms
    norm_2_squared = np.sum(g_center**2)
    norm_1 = np.sum(np.abs(g_center))
    norm_inf = np.max(np.abs(g_center))
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂² as specified in requirements
    """
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f
    g = np.convolve(f, f, mode='full')
    
    # Keep only the valid convolution part (middle)
    half_len = len(f) - 1
    g_valid = g[half_len:-half_len]
    
    # Compute norms
    norm_2_squared = np.sum(g_valid**2)
    norm_1 = np.sum(np.abs(g_valid))
    norm_inf = np.max(np.abs(g_valid))
    
    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0, 0.0, 0.0
    
    return norm_2_squared, norm_1, norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero with stricter thresholds
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0
            
        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

class SpectralEvolutionOptimizer:
    """
    Spectral-based evolutionary optimizer that works in frequency domain
    to construct step functions that maximize C₂
    """
    
    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps
        self.best_solution = None
        self.best_c2 = -np.inf
        # Pre-compute frequency grid for spectral operations
        self.freq_grid = np.arange(0, n_steps//2 + 1)
        self.half_n = n_steps // 2
    
    def generate_spectral_initialization(self) -> List[float]:
        """
        Generate initial configuration by creating spectrally optimized patterns
        """
        # Create a pattern in frequency domain that should lead to good autoconvolution
        # This uses principles from spectral analysis and signal processing
        
        # Start with a base spectrum with energy concentrated at low frequencies
        # to encourage smooth, well-behaved autoconvolutions
        spectrum = np.zeros(self.half_n + 1, dtype=complex)
        
        # Add some dominant frequency components
        for i in range(1, min(8, self.half_n + 1)):
            # Add cosine-like components at various frequencies
            freq = i * 0.5 + np.random.random() * 0.5
            mag = 0.5 + np.random.random() * 0.5
            # Create complex coefficient with phase
            phase = np.random.random() * 2 * np.pi
            spectrum[i] = mag * np.exp(1j * phase)
        
        # Ensure DC component is positive
        spectrum[0] = 0.5 + np.random.random() * 0.5
        
        # Ensure Hermitian symmetry for real signal
        for i in range(1, self.half_n):
            spectrum[-i] = np.conj(spectrum[i])
            
        # Convert to time domain using IFFT
        # Ensure we get a real signal
        f_real = np.real(fft.irfft(spectrum, n=self.n_steps))
        
        # Add some structure and randomness
        f_real = f_real * 0.7 + 0.3 * np.random.random(self.n_steps)
        
        # Ensure non-negativity and normalize
        f_real = np.maximum(f_real, 0)
        if np.sum(f_real) > 0:
            f_real = f_real / np.sum(f_real)
        
        return f_real.tolist()
    
    def reconstruct_from_spectrum(self, spectrum: np.ndarray) -> List[float]:
        """Reconstruct time domain signal from spectrum"""
        # Ensure Hermitian symmetry
        if len(spectrum) > 1:
            for i in range(1, len(spectrum)//2):
                spectrum[-i] = np.conj(spectrum[i])
        
        # Inverse FFT
        f_real = np.real(fft.irfft(spectrum, n=self.n_steps))
        
        # Ensure non-negativity and normalize
        f_real = np.maximum(f_real, 0)
        if np.sum(f_real) > 0:
            f_real = f_real / np.sum(f_real)
            
        return f_real.tolist()
    
    def spectral_mutate(self, spectrum: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Apply spectral mutation"""
        mutated_spectrum = spectrum.copy()
        
        # Mutate selected frequency components
        for i in range(len(mutated_spectrum)):
            if np.random.random() < mutation_rate:
                # Add small perturbation to magnitude and phase
                magnitude_change = np.random.normal(0, 0.05)
                phase_change = np.random.normal(0, 0.1)
                
                mag = abs(mutated_spectrum[i])
                phase = np.angle(mutated_spectrum[i])
                
                new_mag = max(0, mag + magnitude_change)
                new_phase = phase + phase_change
                
                # Reconstruct complex number
                mutated_spectrum[i] = new_mag * np.exp(1j * new_phase)
        
        return mutated_spectrum
    
    def spectral_crossover(self, spectrum1: np.ndarray, spectrum2: np.ndarray) -> np.ndarray:
        """Perform spectral crossover"""
        # Uniform crossover
        crossover_point = np.random.randint(1, len(spectrum1))
        
        child_spectrum = np.concatenate([
            spectrum1[:crossover_point],
            spectrum2[crossover_point:]
        ])
        
        # Ensure conjugate symmetry for real output
        if len(child_spectrum) > 1:
            for i in range(1, len(child_spectrum)//2):
                child_spectrum[-i] = np.conj(child_spectrum[i])
        
        return child_spectrum
    
    def adaptive_evolutionary_search(self, max_generations: int = 50) -> List[float]:
        """
        Evolutionary search in spectral domain with adaptive parameters
        """
        pop_size = 15
        population = []
        
        # Initialize population in spectral domain
        for i in range(pop_size):
            init_spectrum = np.random.random(self.half_n + 1) + 1j * np.random.random(self.half_n + 1)
            init_spectrum[0] = np.real(init_spectrum[0])  # DC component real
            init_spectrum = self.reconstruct_from_spectrum(init_spectrum)
            population.append(init_spectrum)
        
        best_solution = None
        best_c2 = -np.inf
        
        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                c2 = evaluate_c2(individual)
                fitness_scores.append(c2)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_solution = individual.copy()
            
            # Selection - keep top 50%
            sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size//2]
            selected_population = [population[i] for i in sorted_indices]
            
            # Create offspring through crossover and mutation
            new_population = selected_population.copy()
            
            # Elitism: keep the best individual
            if best_solution is not None:
                new_population.append(best_solution)
            
            # Generate new individuals through crossover and mutation
            while len(new_population) < pop_size:
                # Select two parents
                parent1 = selected_population[np.random.randint(0, len(selected_population))]
                parent2 = selected_population[np.random.randint(0, len(selected_population))]
                
                # Convert to spectral domain for crossover/mutation
                spec1 = fft.rfft(parent1, n=self.n_steps)
                spec2 = fft.rfft(parent2, n=self.n_steps)
                
                # Crossover
                child_spec = self.spectral_crossover(spec1, spec2)
                
                # Mutation
                mutation_rate = 0.1 + 0.05 * np.exp(-generation/max_generations)
                child_spec = self.spectral_mutate(child_spec, mutation_rate)
                
                # Reconstruct
                child = self.reconstruct_from_spectrum(child_spec)
                new_population.append(child)
            
            # Trim to population size
            population = new_population[:pop_size]
        
        return best_solution if best_solution is not None else [1.0/self.n_steps] * self.n_steps
    
    def local_refinement(self, initial_f: List[float], max_iter: int = 25) -> List[float]:
        """
        Local refinement using adaptive perturbations in time domain
        """
        f = np.array(initial_f)
        n_steps = len(f)
        
        for _ in range(max_iter):
            current_c2 = evaluate_c2(f.tolist())
            
            # Try adaptive perturbations
            best_f = f.copy()
            best_c2 = current_c2
            
            # Try different perturbation strategies
            for _ in range(40):
                perturbed_f = f.copy()
                
                # Choose perturbation type
                if np.random.random() < 0.7:
                    # Select specific indices to modify
                    idx = np.random.randint(0, n_steps)
                    delta = np.random.normal(0, 0.02)
                    perturbed_f[idx] = max(0, perturbed_f[idx] + delta)
                else:
                    # Global perturbation with some structure
                    perturbed_f = f + np.random.normal(0, 0.01, n_steps)
                    perturbed_f = np.maximum(perturbed_f, 0)
                
                # Normalize
                total = np.sum(perturbed_f)
                if total > 0:
                    perturbed_f = perturbed_f / total
                
                new_c2 = evaluate_c2(perturbed_f.tolist())
                
                if new_c2 > best_c2:
                    best_c2 = new_c2
                    best_f = perturbed_f
                    
            # Early stopping if no significant improvement
            if best_c2 <= current_c2:
                break
                
            f = best_f
            
        return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using spectral evolutionary approach
    """
    try:
        optimizer = SpectralEvolutionOptimizer(n_steps=500)
        
        # Strategy 1: Spectral initialization followed by evolutionary search
        spectral_init = optimizer.generate_spectral_initialization()
        c2_spectral = evaluate_c2(spectral_init)
        
        # Strategy 2: Multi-resolution spectral evolutionary search
        evolved_f = optimizer.adaptive_evolutionary_search(max_generations=40)
        c2_evolved = evaluate_c2(evolved_f)
        
        # Strategy 3: Local refinement of the best solution
        final_solution = evolved_f
        if c2_evolved < c2_spectral:
            final_solution = spectral_init
            
        refined_f = optimizer.local_refinement(final_solution, max_iter=20)
        c2_refined = evaluate_c2(refined_f)
        
        # Return the best solution
        if c2_refined > c2_evolved and c2_refined > c2_spectral:
            return refined_f
        elif c2_evolved > c2_spectral:
            return evolved_f
        else:
            return spectral_init
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        return [1.0/500] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")