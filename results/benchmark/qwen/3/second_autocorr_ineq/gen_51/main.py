# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import random
from numba import njit
import time

@njit
def compute_autoconvolution_norms_fourier(fourier_coeffs):
    """
    Compute autoconvolution norms using Fourier domain approach for efficiency.
    """
    # Convert to time domain to get the actual step function
    n = len(fourier_coeffs)
    # Use inverse FFT to get the real part of the signal
    time_domain = np.real(ifft(fourier_coeffs))
    
    # Ensure non-negative values (clip negative values to zero)
    f_values = np.maximum(time_domain, 0)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Compute norms using trapezoidal integration for ||g||₂²
    g2_sq = 0.0
    dx = 1.0 / len(f_values)  # Assuming normalized domain
    
    # Trapezoidal rule for integral g(t)² dt
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)
    
    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))
    
    return g2_sq, g1, ginf

@njit
def compute_c2_fourier(fourier_coeffs):
    """Compute C₂ using Fourier domain approach"""
    g2_sq, g1, ginf = compute_autoconvolution_norms_fourier(fourier_coeffs)
    
    if g1 == 0 or ginf == 0:
        return 0.0
    
    return g2_sq / (g1 * ginf)

def construct_function() -> list[float]:
    """
    Construct a step function optimized for maximizing C₂ using Fourier-guided evolution.
    """
    # Set parameters
    n_coefficients = 100  # Number of Fourier coefficients to optimize
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_count = 5
    
    # Initialize population with random complex Fourier coefficients
    population = []
    for _ in range(population_size):
        # Random complex coefficients with reasonable magnitudes
        coeffs = np.array([complex(random.uniform(-1, 1), random.uniform(-1, 1)) 
                          for _ in range(n_coefficients)])
        population.append(coeffs)
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Evolutionary process
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for coeffs in population:
            fitness = compute_c2_fourier(coeffs)
            fitness_scores.append((fitness, coeffs))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Check for best solution
        current_best = fitness_scores[0][0]
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = fitness_scores[0][1].copy()
        
        # Create new population
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(elite_count):
            new_population.append(fitness_scores[i][1].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores)
            parent2 = tournament_selection(fitness_scores)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Convert best individual back to step function representation
    n_steps = 2 * n_coefficients - 1  # This will be our step function length
    if n_steps > 10000:
        n_steps = 10000
    
    # Reconstruct final step function from best Fourier coefficients
    reconstructed_signal = np.real(ifft(best_individual))
    # Ensure proper length and non-negative values
    f_values = np.maximum(reconstructed_signal[:n_steps], 0)
    
    # Normalize to avoid extreme values that might cause numerical issues
    total_sum = np.sum(f_values)
    if total_sum > 0:
        f_values = f_values / total_sum * n_steps
    
    return f_values.tolist()

def tournament_selection(fitness_scores, tournament_size=3):
    """Select an individual using tournament selection"""
    tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    return max(tournament, key=lambda x: x[0])[1]

def crossover(parent1, parent2):
    """Perform uniform crossover between two Fourier coefficient vectors"""
    n = len(parent1)
    child1 = np.array(parent1, copy=True)
    child2 = np.array(parent2, copy=True)
    
    # Uniform crossover
    for i in range(n):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    
    return child1, child2

def mutate(individual, mutation_rate):
    """Apply mutation to a Fourier coefficient vector"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Add Gaussian noise to real and imaginary parts separately
            individual[i] += complex(
                random.gauss(0, 0.1 * abs(individual[i].real)),
                random.gauss(0, 0.1 * abs(individual[i].imag))
            )

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")