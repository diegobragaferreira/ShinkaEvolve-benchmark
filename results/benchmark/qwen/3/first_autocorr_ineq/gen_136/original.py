# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import random
import time
from scipy.fft import fft, ifft
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import copy

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

def evaluate_sequence_with_cache(sequence, cache):
    """
    Evaluate sequence with caching to avoid redundant computations.
    """
    key = tuple(sequence)
    if key in cache:
        return cache[key]

    result = evaluate_objective(sequence)
    cache[key] = result
    return result

def generate_initial_sequence():
    """
    Generate a good initial random sequence with more structure.
    """
    # Try to make sequences that have some structure in them
    n = random.randint(100, 1000)
    # Use a combination of distributions to create structure
    if random.random() < 0.3:
        # Power law distribution - heavy tail
        sequence = [random.expovariate(0.1) for _ in range(n)]
        # Normalize to prevent extreme values
        max_val = max(sequence)
        sequence = [x * 100.0 / max_val if max_val > 0 else 1.0 for x in sequence]
    elif random.random() < 0.6:
        # Uniform distribution with some peaks
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        # Add a few peaks
        for i in range(min(10, len(sequence)//20)):
            peak_pos = random.randint(0, len(sequence)-1)
            sequence[peak_pos] = random.uniform(100.0, 1000.0)
    else:
        # Mixed distribution
        sequence = []
        for i in range(n):
            if random.random() < 0.7:
                sequence.append(random.uniform(0.1, 10.0))
            else:
                sequence.append(random.uniform(50.0, 100.0))

    return sequence

def generate_population(size, min_size=100, max_size=1000):
    """Generate a population of sequences."""
    population = []
    for _ in range(size):
        n = random.randint(min_size, max_size)
        sequence = generate_initial_sequence()
        population.append(sequence)
    return population

def quadratic_optimization_step(current_seq, max_iter=100):
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

    # Try multiple optimization methods
    methods_to_try = ['SLSQP', 'L-BFGS-B']

    for method in methods_to_try:
        try:
            # Use smaller tolerance for faster convergence
            result = minimize(objective, current_seq, method=method, bounds=bounds,
                            constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
            if result.success:
                return result.x.tolist()
        except:
            continue

    # If all methods fail, return the original sequence slightly perturbed
    perturbed = [max(0.0, x + random.gauss(0, 0.05)) for x in current_seq]
    if np.sum(perturbed) < 0.01:
        perturbed[0] = max(0.0, perturbed[0] + 0.01)
    return perturbed

def mutate_sequence(sequence, mutation_rate=0.1):
    """Mutate a sequence by randomly changing some elements."""
    mutated = sequence.copy()
    # Adaptive mutation rate: lower for longer sequences
    if len(sequence) > 500:
        mutation_rate *= 0.5
    elif len(sequence) < 200:
        mutation_rate *= 1.5

    # Calculate standard deviation for mutation scaling
    std_dev = np.std(sequence) if len(sequence) > 0 else 1.0
    mutation_scale = max(0.1, std_dev * 0.1)  # Scale mutation by sequence variability

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate with Gaussian noise scaled by sequence variability
            mutated[i] = max(0.0, mutated[i] + random.gauss(0, mutation_scale))
    return mutated

def crossover_sequences(parent1, parent2):
    """Perform crossover between two sequences with adaptive mixing."""
    # Use a blend crossover that considers characteristics of parents
    n1, n2 = len(parent1), len(parent2)
    min_len = min(n1, n2)

    # Create offspring with blended elements
    offspring = []

    # Determine if we're doing crossover or just taking one parent
    if random.random() < 0.7:  # 70% chance of crossover
        # Blend elements with weight based on parent characteristics
        for i in range(max(n1, n2)):
            if i < min_len:
                # Blend based on similarity of elements
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
            elif i < n1:
                # Extending beyond shorter parent
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
    else:
        # Just take one parent with some variation
        parent = parent1 if random.random() < 0.5 else parent2
        offspring = parent.copy()

    # Ensure all elements are non-negative
    offspring = [max(0.0, x) for x in offspring]
    return offspring

def diversify_population(population, diversity_threshold=0.05):
    """
    Introduce diversity by adding random sequences when population becomes too homogeneous.
    """
    if len(population) < 2:
        return population

    # Check diversity by looking at standard deviation of average values
    avg_values = [np.mean(seq) for seq in population]
    diversity_metric = np.std(avg_values) / (np.mean(avg_values) + 1e-8)

    if diversity_metric < diversity_threshold:
        # Add some random sequences to maintain diversity
        for _ in range(len(population) // 4):  # Add 25% random sequences
            new_seq = generate_initial_sequence()
            population.append(new_seq)

    return population

def adaptive_local_search(current_seq, max_iter=200):
    """Enhanced local search with adaptive step sizes"""
    n = len(current_seq)
    bounds = [(0.0, 1000.0) for _ in range(n)]
    
    def sum_constraint(x):
        return np.sum(x) - 0.01

    constraints = [{'type': 'ineq', 'fun': sum_constraint}]
    
    def objective(x):
        return evaluate_objective(x)
    
    # Try multiple local optimization approaches
    methods = ['SLSQP', 'L-BFGS-B']
    best_result = None
    best_value = float('inf')
    
    for method in methods:
        try:
            # Start with small steps and gradually increase
            for step_size in [1e-3, 1e-2, 1e-1]:
                attempt = current_seq.copy()
                if random.random() < 0.5:  # Randomize a bit
                    attempt = mutate_sequence(attempt, step_size)
                
                result = minimize(objective, attempt, method=method, bounds=bounds, constraints=constraints, 
                                options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_result = result.x.tolist()
        except:
            continue
    
    if best_result is not None:
        return best_result
    else:
        return current_seq

def evaluate_population_parallel(population, max_workers=4):
    """Evaluate a population in parallel using multiple processes"""
    with mp.Pool(processes=max_workers) as pool:
        func = partial(evaluate_objective)
        results = pool.map(func, population)
    return results

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence using an enhanced evolutionary approach.
    """
    start_time = time.time()
    population_size = 40
    generations = 80
    keep_top = 10
    elite_preservation = 3
    diversity_check_interval = 10

    # Cache for storing previously computed evaluations
    evaluation_cache = {}
    
    # Track historical best for diversity preservation
    historical_best = []
    
    # Generate initial population
    population = generate_population(population_size)
    
    # Evaluate initial population in parallel
    fitness_scores = []
    for seq in population:
        fitness = evaluate_sequence_with_cache(seq, evaluation_cache)
        fitness_scores.append((seq, fitness))
    
    # Sort population by fitness (lower is better)
    fitness_scores.sort(key=lambda x: x[1])
    
    # Track best solution globally
    global_best = fitness_scores[0][0]
    global_best_fitness = fitness_scores[0][1]
    
    # Adaptive parameters
    mutation_rate = 0.1
    crossover_prob = 0.3
    local_search_prob = 0.7
    patience_limit = 15
    patience_counter = 0
    
    # Main evolution loop
    for gen in range(generations):
        if time.time() - start_time > 170:  # Leave 10 seconds for finalization
            break
            
        # Adjust adaptive parameters based on performance
        if patience_counter > patience_limit // 2:
            mutation_rate = min(0.3, mutation_rate * 1.1)
            crossover_prob = min(0.6, crossover_prob * 1.1)
            
        # Periodic diversity check
        if gen % diversity_check_interval == 0:
            population = diversify_population(population)

        # Keep top performers (elite)
        top_performers = [seq for seq, _ in fitness_scores[:keep_top]]
        
        # Create new population
        new_population = top_performers[:]
        
        # Preserve elites
        if elite_preservation > 0:
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[:elite_preservation]
            elites = [fitness_scores[i][0] for i in elite_indices]
            new_population.extend(elites)
        
        # Add mutated versions of top performers
        for i in range(population_size - len(new_population)):
            if random.random() < crossover_prob:  # Crossover
                p1, p2 = random.sample(top_performers, 2)
                child = crossover_sequences(p1, p2)
            else:  # Mutation
                parent = random.choice(top_performers)
                child = mutate_sequence(parent, mutation_rate)
            
            new_population.append(child)
        
        # Apply local optimization to some individuals
        for i in range(0, len(new_population), 2):
            if random.random() < local_search_prob:  # Local search
                new_population[i] = adaptive_local_search(new_population[i])
        
        # Evaluate new population in parallel
        fitness_scores = []
        for seq in new_population:
            fitness = evaluate_sequence_with_cache(seq, evaluation_cache)
            fitness_scores.append((seq, fitness))
        
        # Sort population by fitness
        fitness_scores.sort(key=lambda x: x[1])
        
        # Update global best
        if fitness_scores[0][1] < global_best_fitness:
            global_best = fitness_scores[0][0]
            global_best_fitness = fitness_scores[0][1]
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            
        # Diversity preservation: add historical best occasionally
        if len(historical_best) < 5 and random.random() < 0.1:
            historical_best.append(copy.deepcopy(global_best))
            # Add one historical best to population
            if len(historical_best) > 0:
                historical_sample = historical_best[random.randint(0, len(historical_best)-1)]
                # Mutate slightly to add variation
                mutated_historical = mutate_sequence(historical_sample, 0.05)
                # Replace worst performer
                worst_index = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[-1]
                fitness_scores[worst_index] = (mutated_historical, evaluate_objective(mutated_historical))
        
        # Occasionally restart with a new random individual if stuck
        if patience_counter > patience_limit:
            new_individual = [random.uniform(0.1, 100.0) for _ in range(random.randint(100, 1000))]
            fitness_scores[-1] = (new_individual, evaluate_objective(new_individual))
            patience_counter = 0  # Reset patience
    
    # Final optimization of the best sequence
    final_best = adaptive_local_search(global_best, max_iter=300)
    
    # Return the best sequence found
    return final_best

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")