# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import random
import time
from scipy.fft import fft, ifft

def compute_autocorrelation_constant(sequence):
    """
    Compute C₁ for a given sequence using FFT for efficiency.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    """
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Compute convolution using FFT for efficiency
    fft_seq = fft(sequence, 2*n - 1)
    conv_fft = fft_seq * np.conj(fft_seq)
    conv = ifft(conv_fft).real[:2*n-1]
    max_conv = np.max(conv)

    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def evaluate_objective(sequence):
    """
    Evaluate the objective function: -1/C₁ (we minimize this to maximize 1/C₁)
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return float('inf')  # Invalid solution
    return -1.0 / c1  # Negative because we want to maximize 1/C₁

def generate_initial_sequence():
    """
    Generate a good initial random sequence.
    """
    n = random.randint(100, 1000)
    return [random.uniform(0.1, 100.0) for _ in range(n)]

def generate_population(size, min_size=100, max_size=1000):
    """Generate a population of sequences."""
    population = []
    for _ in range(size):
        n = random.randint(min_size, max_size)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)
    return population

def quadratic_optimization_step(current_seq):
    """
    Perform a quadratic optimization step to improve the sequence.
    """
    n = len(current_seq)
    # Define bounds: all elements must be in [0, 1000]
    bounds = [(0.0, 1000.0) for _ in range(n)]

    # Define constraints
    def sum_constraint(x):
        return np.sum(x) - 0.01  # Require sum >= 0.01

    constraints = [{'type': 'ineq', 'fun': sum_constraint}]

    # Objective function to minimize
    def objective(x):
        return evaluate_objective(x)

    # Use SLSQP method which handles both bounds and constraints well
    try:
        result = minimize(objective, current_seq, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 50, 'ftol': 1e-6})
        if result.success:
            return result.x.tolist()
    except:
        pass

    # If optimization fails, return the original sequence slightly perturbed
    perturbed = [max(0.0, x + random.gauss(0, 0.1)) for x in current_seq]
    if np.sum(perturbed) < 0.01:
        perturbed[0] = max(0.0, perturbed[0] + 0.01)
    return perturbed

def mutate_sequence(sequence, mutation_rate=0.1):
    """Mutate a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = max(0.0, mutated[i] + random.gauss(0, 0.1))
    return mutated

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence using a hybrid approach.
    """
    start_time = time.time()
    population_size = 20
    generations = 50
    keep_top = 5

    # Generate initial population
    population = generate_population(population_size)

    # Evaluate initial population
    fitness_scores = []
    for seq in population:
        fitness = evaluate_objective(seq)
        fitness_scores.append((seq, fitness))

    # Sort population by fitness (lower is better)
    fitness_scores.sort(key=lambda x: x[1])

    # Main evolution loop
    for gen in range(generations):
        if time.time() - start_time > 170:  # Leave 10 seconds for finalization
            break

        # Keep top performers
        top_performers = [seq for seq, _ in fitness_scores[:keep_top]]

        # Create new population
        new_population = top_performers[:]

        # Add mutated versions of top performers
        for i in range(population_size - len(top_performers)):
            parent = random.choice(top_performers)
            child = mutate_sequence(parent)
            new_population.append(child)

        # Apply local optimization to some individuals
        for i in range(0, len(new_population), 2):
            if random.random() < 0.5:
                new_population[i] = quadratic_optimization_step(new_population[i])

        # Evaluate new population
        fitness_scores = []
        for seq in new_population:
            fitness = evaluate_objective(seq)
            fitness_scores.append((seq, fitness))

        # Sort population by fitness
        fitness_scores.sort(key=lambda x: x[1])

    # Return the best sequence found
    best_sequence = fitness_scores[0][0]
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")