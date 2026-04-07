# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import warnings
from numba import jit
import time
import random
import logging
from typing import List, Tuple
import math
from collections import deque
import copy

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast computation of autoconvolution norms using Numba JIT compilation.
    """
    n = len(f_values)
    if n < 1:
        return 0.0, 0.0, 0.0

    # Convert to numpy array for fast operations
    f = np.array(f_values, dtype=np.float64)
    
    # Create the step function on [-1/4, 1/4] with equal spacing
    dx = 0.5 / (n - 1) if n > 1 else 0.5
    
    # Precompute convolution manually for efficiency
    # Autoconvolution g[k] = sum f[i] * f[k-i] for valid indices
    g = np.zeros(2 * n - 1)
    
    # Manual convolution loop (optimized for autoconvolution)
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < len(g):
                g[k] += f[i] * f[j]
    
    # Keep only the middle part (proper autoconvolution)
    g_middle = g[n-1:2*n-1]
    
    # Create x-axis for g (interval [-0.5, 0.5])
    g_x = np.linspace(-0.5, 0.5, len(g_middle))
    
    # Compute the required norms
    # ||g||₂² (L2 norm squared)
    # Using trapezoidal integration approximation 
    g_sq = g_middle * g_middle
    area = 0.0
    for i in range(len(g_middle) - 1):
        h = g_x[i+1] - g_x[i]
        area += h * (g_sq[i] + g_sq[i+1]) / 2
    
    norm_2_sq = area

    # ||g||₁ (L1 norm) - approximate via summation
    norm_1 = np.sum(np.abs(g_middle)) * dx  # dx is the step size

    # ||g||∞ (infinity norm)
    norm_inf = np.max(np.abs(g_middle))

    return norm_2_sq, norm_1, norm_inf

class SpectralOptimizer:
    """Handles spectral domain optimization operations"""
    
    def __init__(self, n_points: int = 2000):
        self.n_points = n_points
    
    def generate_initial_spectrum(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial spectral representation with strategic peaks"""
        if seed is not None:
            np.random.seed(seed)
        
        # Create frequency domain representation with strategic peaks
        magnitudes = np.zeros(self.n_points)
        
        # Create multiple clusters of frequency components to promote smooth autoconvolution
        # Cluster centers in log-frequency space
        n_clusters = 5
        cluster_centers = np.logspace(np.log10(1), np.log10(self.n_points//2), n_clusters)
        
        # Each cluster contributes multiple components
        for i, center in enumerate(cluster_centers):
            n_components = np.random.randint(2, 6)
            for j in range(n_components):
                # Spread components within cluster
                freq_offset = np.random.normal(0, center * 0.15)
                freq_idx = int(center + freq_offset)
                if 1 <= freq_idx < self.n_points//2:
                    # Add energy with gamma distribution for varied strengths
                    strength = np.random.gamma(2.5, 1.2)
                    magnitudes[freq_idx] = strength
                    if freq_idx > 0:
                        magnitudes[-freq_idx] = strength  # Conjugate symmetry
                        
        # Add some low frequency content for smoothness
        magnitudes[0] = np.random.gamma(1.5, 2.5)  # DC component
        
        # Add phase information
        phases = np.random.uniform(0, 2*np.pi, self.n_points)
        
        return magnitudes, phases
    
    def reconstruct_time_domain(self, magnitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Convert spectral representation back to time domain"""
        # Create complex spectrum
        spectrum = magnitudes * np.exp(1j * phases)
        
        # Convert back to time domain
        f_real = np.real(ifft(spectrum))
        
        # Ensure non-negativity
        f_real = np.maximum(f_real, 0.0)
        
        # Normalize
        max_val = np.max(f_real)
        if max_val > 0:
            f_real = f_real / (max_val * 1.8)
        
        return f_real
    
    def evolve_spectrum(self, magnitudes: np.ndarray, phases: np.ndarray, 
                       iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve spectrum through frequency domain modifications"""
        evolved_mags = magnitudes.copy()
        evolved_phases = phases.copy()
        
        for _ in range(iterations):
            # Randomly select components to modify
            n_modify = np.random.randint(1, max(1, self.n_points // 15))
            indices = np.random.choice(self.n_points, n_modify, replace=False)
            
            for idx in indices:
                if np.random.random() < 0.7:  # 70% chance to modify magnitude
                    # Perturb magnitude with bounded Gaussian
                    perturbation = np.random.normal(0, 0.15)
                    new_mag = evolved_mags[idx] * np.exp(perturbation)
                    evolved_mags[idx] = np.clip(new_mag, 0, 1000.0)
                
                if np.random.random() < 0.5:  # 50% chance to modify phase
                    # Perturb phase with small amount
                    perturbation = np.random.normal(0, 0.15)
                    evolved_phases[idx] += perturbation
                    # Keep phases in [-pi, pi]
                    evolved_phases[idx] = evolved_phases[idx] % (2 * np.pi)
        
        return evolved_mags, evolved_phases
    
    def crossover_spectra(self, magnitudes1: np.ndarray, phases1: np.ndarray,
                         magnitudes2: np.ndarray, phases2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover operation in frequency domain"""
        # Blend frequency components
        alpha = np.random.beta(2, 2)  # Beta distribution for more balanced blending
        blended_mags = alpha * magnitudes1 + (1 - alpha) * magnitudes2
        blended_phases = alpha * phases1 + (1 - alpha) * phases2
        
        return blended_mags, blended_phases

class PopulationManager:
    """Manages evolutionary population dynamics"""
    
    def __init__(self, population_size: int = 25, n_points: int = 2000):
        self.population_size = population_size
        self.n_points = n_points
        self.optimizer = SpectralOptimizer(n_points)
        self.stagnation_count = 0
        self.best_fitness_history = deque(maxlen=10)
    
    def initialize_population(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Initialize population with diverse spectral functions"""
        population = []
        
        for i in range(self.population_size):
            # Use different seeds for variety
            seed = i * 1000 + int(time.time())
            magnitudes, phases = self.optimizer.generate_initial_spectrum(seed)
            
            # Convert to time domain
            time_domain = self.optimizer.reconstruct_time_domain(magnitudes, phases)
            
            population.append((magnitudes, phases, time_domain.tolist()))
        
        return population
    
    def tournament_selection(self, population_data: List[Tuple[np.ndarray, np.ndarray, List[float], float]], 
                           tournament_size: int = 4) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Perform tournament selection"""
        tournament_indices = np.random.choice(len(population_data), tournament_size, replace=False)
        tournament_fitness = [population_data[i][3] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population_data[winner_idx][0], population_data[winner_idx][1], population_data[winner_idx][2]
    
    def evolve_population(self, population_data: List[Tuple[np.ndarray, np.ndarray, List[float], float]], 
                         generation: int) -> List[Tuple[np.ndarray, np.ndarray, List[float], float]]:
        """Evolve the population using selection, crossover and mutation"""
        # Sort by fitness
        sorted_indices = sorted(range(len(population_data)), key=lambda i: population_data[i][3], reverse=True)
        
        # Keep elites
        elite_count = max(1, self.population_size // 4)
        elites = [population_data[i] for i in sorted_indices[:elite_count]]
        
        # Generate offspring
        offspring = []
        
        while len(offspring) < self.population_size - elite_count:
            # Selection
            parent1_mag, parent1_phase, parent1_time = self.tournament_selection(population_data)
            parent2_mag, parent2_phase, parent2_time = self.tournament_selection(population_data)
            
            # Crossover
            if len(parent1_time) == len(parent2_time):
                child_mag, child_phase = self.optimizer.crossover_spectra(
                    parent1_mag, parent1_phase, parent2_mag, parent2_phase
                )
            else:
                # Fallback to simple averaging
                child_mag = (parent1_mag + parent2_mag) / 2
                child_phase = (parent1_phase + parent2_phase) / 2
            
            # Mutation
            child_mag, child_phase = self.optimizer.evolve_spectrum(child_mag, child_phase, iterations=3)
            
            # Reconstruct and add to offspring
            child_time = self.optimizer.reconstruct_time_domain(child_mag, child_phase)
            offspring.append((child_mag, child_phase, child_time.tolist()))
        
        # Combine elites and offspring
        new_population = elites + offspring
        return new_population

def compute_c2_with_fallback(f_values: List[float]) -> float:
    """Compute C2 with comprehensive error handling"""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
        
        # Avoid division by zero with numerical stability
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return float(c2)
    except Exception as e:
        warnings.warn(f"C2 computation failed: {str(e)}")
        return 0.0

def adaptive_spectral_evolution(max_time_seconds: int = 85) -> List[float]:
    """Main evolutionary algorithm in spectral domain with adaptive strategies"""
    start_time = time.time()
    
    # Initialize population manager
    pop_manager = PopulationManager(population_size=25, n_points=2000)
    population = pop_manager.initialize_population()
    
    best_fitness = -1
    best_individual = None
    
    generation = 0
    max_generations = 200
    stagnation_limit = 30
    
    while generation < max_generations and (time.time() - start_time) < max_time_seconds - 2:
        # Evaluate fitness for current population
        population_with_fitness = []
        
        for magnitudes, phases, time_domain in population:
            fitness = compute_c2_with_fallback(time_domain)
            population_with_fitness.append((magnitudes, phases, time_domain, fitness))
        
        # Track best
        current_best_idx = max(range(len(population_with_fitness)), 
                              key=lambda i: population_with_fitness[i][3])
        current_best_fitness = population_with_fitness[current_best_idx][3]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population_with_fitness[current_best_idx][2].copy()
            pop_manager.stagnation_count = 0
            pop_manager.best_fitness_history.append(best_fitness)
        else:
            pop_manager.stagnation_count += 1
            
        # Check for stagnation
        if pop_manager.stagnation_count > stagnation_limit:
            # Introduce diversity by generating new individuals
            for i in range(pop_manager.population_size // 5):
                seed = int(time.time() * 1000 + i + generation) % (2**32)
                magnitudes, phases = pop_manager.optimizer.generate_initial_spectrum(seed)
                time_domain = pop_manager.optimizer.reconstruct_time_domain(magnitudes, phases)
                # Replace worst individual
                worst_idx = min(range(len(population_with_fitness)), 
                               key=lambda i: population_with_fitness[i][3])
                population[worst_idx] = (magnitudes, phases, time_domain.tolist())
            pop_manager.stagnation_count = 0
        
        # Evolve population
        population = pop_manager.evolve_population(population_with_fitness, generation)
        generation += 1
    
    # Return best individual found
    return best_individual if best_individual is not None else [0.5] * 1000

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value using spectral evolutionary approach.
    """
    try:
        # Use spectral evolutionary approach
        f_values = adaptive_spectral_evolution(max_time_seconds=85)
        return f_values
    except Exception as e:
        # Fallback to simple uniform distribution
        warnings.warn(f"Spectral evolution failed: {str(e)}")
        return [0.5] * 1000

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")