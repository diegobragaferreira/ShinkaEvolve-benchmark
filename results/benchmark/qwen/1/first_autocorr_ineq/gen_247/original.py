# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve
import time

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return float('inf')

    # Compute autocorrelation using FFT for efficiency
    autocorr = fftconvolve(sequence, sequence[::-1], mode='full')
    # Take the second half which corresponds to the actual autocorrelation
    autocorr = autocorr[len(sequence)-1:]
    max_autocorr = np.max(autocorr)

    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    # C₁ = 2n * max(autocorr) / (sum(sequence))^2
    C1 = 2 * len(sequence) * max_autocorr / sum_sq

    return C1

def objective_function(sequence):
    """
    Objective function to minimize (negative of inverse of C₁).
    """
    C1 = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return float('inf')
    return -1.0 / C1

def generate_initial_population(popsize, n_min, n_max):
    """
    Generate initial population of sequences.
    """
    population = []
    for _ in range(popsize):
        n = np.random.randint(n_min, n_max)
        seq = np.random.rand(n)
        # Normalize to ensure sum > 0.01
        seq = seq * (0.01 / (np.sum(seq) + 1e-10))
        population.append(seq)
    return population

def evaluate_population(population):
    """
    Evaluate fitness of the entire population.
    """
    fitness = []
    for ind in population:
        fit = objective_function(ind)
        fitness.append(fit)
    return np.array(fitness)

def evolve_sequences():
    """
    Evolutionary optimization for finding optimal sequence.
    """
    # Parameters
    popsize = 20
    maxiter = 100
    n_min, n_max = 50, 1000
    seed = 42

    np.random.seed(seed)

    # Start with a good random sequence
    initial_n = np.random.randint(n_min, n_max)
    initial_seq = np.random.rand(initial_n)
    initial_seq = initial_seq * (0.01 / (np.sum(initial_seq) + 1e-10))

    # Define bounds for each element in the sequence
    bounds = [(0, 1000) for _ in range(len(initial_seq))]

    # Define custom callback to stop early if needed
    def callback(xk, convergence):
        pass

    # Run differential evolution optimization
    result = differential_evolution(
        objective_function,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        disp=False,
        callback=callback,
        strategy='best1bin'
    )

    if result.success:
        return result.x
    else:
        # If optimization fails, return the initial sequence
        return initial_seq

def search_for_best_sequence():
    """
    Main function to find the best sequence by evolutionary optimization.
    """
    start_time = time.time()

    # Run the evolutionary optimization
    best_sequence = evolve_sequences()

    # Ensure the sequence meets minimum requirements
    if np.sum(best_sequence) < 0.01:
        # Reinitialize if necessary
        n = np.random.randint(50, 1000)
        best_sequence = np.random.rand(n)
        best_sequence = best_sequence * (0.01 / (np.sum(best_sequence) + 1e-10))

    end_time = time.time()

    return list(best_sequence)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")