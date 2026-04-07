# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
import time
import math
from numba import jit, njit

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 120
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 10
MAX_STEPS = 10000
MIN_STEPS = 500
SEED = 42
FFT_SIZE = 2048  # FFT size for efficient convolution

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

@njit
def compute_fourier_convolution_norms_jit(fourier_coeffs, n_steps):
    """
    Compute autoconvolution norms using FFT in the frequency domain.
    This provides significant speedup over spatial domain computation.
    """
    # Create frequency domain representation
    # We work with complex Fourier coefficients
    fft_size = FFT_SIZE
    freq_domain = np.zeros(fft_size, dtype=np.complex128)
    
    # Fill in the frequency coefficients (assuming real input)
    # We use only the first half since input is real
    if n_steps <= fft_size//2:
        freq_domain[:n_steps//2 + 1] = fourier_coeffs[:n_steps//2 + 1]
        # Conjugate symmetry for real signals
        if n_steps % 2 == 0:
            freq_domain[fft_size - n_steps//2:] = np.conj(fourier_coeffs[:n_steps//2][::-1])
        else:
            freq_domain[fft_size - (n_steps//2 + 1):] = np.conj(fourier_coeffs[1:n_steps//2 + 1][::-1])
    else:
        # Truncate or pad
        half_size = fft_size // 2
        if len(fourier_coeffs) >= half_size:
            freq_domain[:half_size] = fourier_coeffs[:half_size]
            freq_domain[fft_size - half_size + 1:] = np.conj(fourier_coeffs[:half_size-1][::-1])
        else:
            freq_domain[:len(fourier_coeffs)] = fourier_coeffs
            freq_domain[fft_size - len(fourier_coeffs) + 1:] = np.conj(fourier_coeffs[:len(fourier_coeffs)-1][::-1])
    
    # Perform convolution in frequency domain: g^ = f^ * f^ = |f^|²
    conv_freq = freq_domain * np.conj(freq_domain)
    
    # Inverse FFT to get spatial domain convolution
    g = np.fft.irfft(conv_freq, fft_size)
    
    # Normalize by the step width
    step_width = 0.5 / n_steps
    g = g * step_width
    
    # Compute required norms using trapezoidal rule (since we're not doing full trapezoidal integration)
    g_squared = g**2
    g_abs = np.abs(g)
    
    # For norms, we take the first half (due to symmetry in autoconvolution)
    g_half = g[:len(g)//2 + 1]
    g_squared_half = g_squared[:len(g_squared)//2 + 1]
    g_abs_half = g_abs[:len(g_abs)//2 + 1]
    
    # Approximate integration using trapezoidal rule (simplified)
    dx = 0.5 / fft_size  # approximate grid spacing
    
    # ||g||₂² (L2 norm squared)
    norm_2_squared = np.sum(g_squared_half) * dx
    
    # ||g||₁ (L1 norm)  
    norm_1 = np.sum(g_abs_half) * dx
    
    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs_half)
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms_fft(fourier_coeffs, n_steps):
    """
    Compute the three norms needed for C2 calculation using FFT-based approach.
    This is a more robust version that handles edge cases better.
    """
    try:
        # Convert to numpy array
        coeffs = np.array(fourier_coeffs)
        
        # We'll do a slightly different approach - build the actual function first
        # then compute convolution via FFT
        
        # Create step function in spatial domain from Fourier coefficients
        # For simplicity, we'll use the inverse FFT method
        step_width = 0.5 / n_steps
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Create a simple reconstruction of the step function
        # We'll treat the frequencies as a filter applied to a unit impulse
        # but this requires a bit of care - let's use the more direct approach
        
        # Let's do something simpler: just use the fact that we want to compute
        # g = f * f, where f is a piecewise linear function. Since we'll use
        # FFT anyway, let's make sure our data is properly formed.
        
        # Use the JIT version for computation
        norm_2_squared, norm_1, norm_inf = compute_fourier_convolution_norms_jit(fourier_coeffs, n_steps)
        return norm_2_squared, norm_1, norm_inf
        
    except Exception as e:
        # Fallback to basic convolution
        try:
            # If FFT fails, fall back to traditional convolution for reliability
            f = np.zeros(n_steps)
            # Assume first k coefficients are given, rest are zero
            k = min(len(fourier_coeffs), n_steps)
            f[:k] = fourier_coeffs[:k]
            
            # Pad with zeros to ensure we have enough samples for convolution
            f_padded = np.pad(f, (0, n_steps), 'constant', constant_values=0)
            
            # Perform autoconvolution
            g = np.convolve(f_padded, f_padded, mode='full')
            g = g[:len(g)//2 + 1]  # First half due to symmetry
            
            # Scale appropriately
            dx = 0.5 / n_steps
            g = g * dx
            
            # Compute norms
            g_squared = g**2
            g_abs = np.abs(g)
            
            # Approximate integration
            norm_2_squared = np.sum(g_squared) * dx
            norm_1 = np.sum(g_abs) * dx
            norm_inf = np.max(g_abs)
            
            return norm_2_squared, norm_1, norm_inf
        except:
            return 0.0, 0.0, 0.0

def calculate_c2_from_fourier(fourier_coeffs, n_steps):
    """Calculate C₂ from Fourier coefficients"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms_fft(fourier_coeffs, n_steps)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except:
        return 0.0

def initialize_population_fourier(pop_size, min_steps, max_steps):
    """Initialize population with diverse Fourier coefficients"""
    population = []
    for _ in range(pop_size):
        # Random number of steps
        n_steps = np.random.randint(min_steps, max_steps)
        
        # Generate Fourier coefficients
        # Use a mix of frequencies to encourage complex patterns
        # Start with a base spectrum and add some randomness
        base_coeffs = np.zeros(n_steps//2 + 1, dtype=complex)
        
        # Add some low frequencies (dominant components)
        for i in range(min(5, n_steps//2)):
            base_coeffs[i] = complex(np.random.uniform(0, 1), np.random.uniform(-0.5, 0.5))
        
        # Add higher frequencies with smaller magnitudes
        for i in range(5, min(15, n_steps//2)):
            mag = np.random.exponential(scale=0.1)
            phase = np.random.uniform(0, 2*np.pi)
            base_coeffs[i] = complex(mag * np.cos(phase), mag * np.sin(phase))
        
        # Add occasional larger peaks for diversity
        for _ in range(3):
            idx = np.random.randint(1, min(50, n_steps//2))  # Skip DC component
            mag = np.random.exponential(scale=0.5)
            phase = np.random.uniform(0, 2*np.pi)
            base_coeffs[idx] = base_coeffs[idx] + complex(mag * np.cos(phase), mag * np.sin(phase))
            
        # Flatten to real array for storage (real and imaginary parts)
        flattened_coeffs = np.empty(2 * len(base_coeffs) - 1, dtype=float)
        flattened_coeffs[0::2] = np.real(base_coeffs)  # Real parts
        flattened_coeffs[1::2] = np.imag(base_coeffs)  # Imaginary parts
        
        # Clip negative values (though they might be valid in complex domain)
        flattened_coeffs = np.maximum(flattened_coeffs, 0)
        population.append(flattened_coeffs.tolist())
        
    return population

def crossover_fourier(parent1, parent2):
    """Perform crossover between two Fourier coefficient sets"""
    if len(parent1) != len(parent2):
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]
        
    if np.random.random() < CROSSOVER_RATE:
        # Uniform crossover on the flattened representation
        child1, child2 = [], []
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    else:
        return parent1, parent2

def mutate_fourier(individual, mutation_rate, generation=None):
    """Mutate Fourier coefficients with adaptive scaling"""
    mutated = individual.copy()
    
    # Mutate real and imaginary parts separately for better control
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Apply Gaussian noise to each coefficient
            # Consider the magnitude when determining noise scale
            if i % 2 == 0:  # Real part
                if len(mutated) > i+1:
                    magnitude = abs(mutated[i] + 1j * mutated[i+1])  # Magnitude of complex number
                    noise_scale = 0.1 * magnitude + 0.01
                else:
                    noise_scale = 0.1
            else:  # Imaginary part
                if i-1 >= 0:
                    magnitude = abs(mutated[i-1] + 1j * mutated[i])  # Magnitude of complex number
                    noise_scale = 0.1 * magnitude + 0.01
                else:
                    noise_scale = 0.1
                    
            noise = np.random.normal(0, noise_scale)
            mutated[i] = max(0, mutated[i] + noise)
            
    return mutated

def evaluate_fitness_fourier(population):
    """Evaluate fitness of entire population"""
    results = []
    for ind in population:
        # Extract n_steps from individual size
        n_steps = max(MIN_STEPS, int(np.random.uniform(MIN_STEPS, MAX_STEPS)))
        # For now, we'll assume the population member represents the correct number of steps
        # But let's fix the number of coefficients based on n_steps
        half_size = n_steps // 2 + 1
        expected_coefficients = 2 * half_size - 1
        
        if len(ind) < expected_coefficients:
            # Pad with zeros
            ind = ind + [0.0] * (expected_coefficients - len(ind))
        elif len(ind) > expected_coefficients:
            # Truncate
            ind = ind[:expected_coefficients]
        
        # Compute C2 directly
        c2 = calculate_c2_from_fourier(ind, n_steps)
        results.append(c2)
    
    return results

def select_parents_fourier(population, fitness_scores):
    """Tournament selection for Fourier domain"""
    selected = []
    tournament_size = min(5, len(population) // 3)
    
    for _ in range(len(population)):
        # Tournament selection
        tournament_indices = np.random.choice(len(population), 
                                             min(tournament_size, len(population)), 
                                             replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index].copy())
    
    return selected

def elitism_fourier(population, fitness_scores, elite_count):
    """Keep best individuals"""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite = [population[i].copy() for i in sorted_indices[:elite_count]]
    return elite

def fourier_guided_evolution():
    """Main evolutionary algorithm operating in Fourier domain"""
    # Initialize population in Fourier domain
    population = initialize_population_fourier(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)
    
    best_individual = None
    best_fitness = -np.inf
    
    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = evaluate_fitness_fourier(population)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Print progress every 15 generations
        if generation % 15 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.4f}")
        
        # Elitism
        elite = elitism_fourier(population, fitness_scores, ELITISM_COUNT)
        
        # Selection
        parents = select_parents_fourier(population, fitness_scores)
        
        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child1, child2 = crossover_fourier(parents[p1], parents[p2])
            
            child1 = mutate_fourier(child1, MUTATION_RATE, generation)
            child2 = mutate_fourier(child2, MUTATION_RATE, generation)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]
    
    return best_individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Run Fourier-guided evolution
    result = fourier_guided_evolution()
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Validate the result
    n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
    try:
        c2 = calculate_c2_from_fourier(result, n_steps)
    except:
        c2 = 0.0
        
    print(f"Evaluated in {eval_time:.2f} seconds")
    print(f"Best C2 found: {c2:.6f}")
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
