# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import random
import time
import math
from collections import deque

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

def quantum_mutation(sequence, mutation_rate=0.1, amplitude_factor=0.5):
    """Quantum-inspired mutation using amplitude modulation."""
    mutated = sequence.copy()
    n = len(mutated)
    
    for i in range(n):
        if random.random() < mutation_rate:
            # Apply quantum amplitude modulation
            amplitude = amplitude_factor * mutated[i]
            phase_shift = random.uniform(-np.pi, np.pi)
            mutated[i] = max(0.0, mutated[i] + amplitude * np.sin(phase_shift))
    
    return mutated

def quantum_crossover(parent1, parent2, crossover_prob=0.7):
    """Quantum-inspired crossover using superposition blending."""
    n1, n2 = len(parent1), len(parent2)
    min_len = min(n1, n2)
    offspring = []
    
    # Superposition blend with quantum probability
    for i in range(max(n1, n2)):
        if i < min_len:
            # Quantum probability of choosing from each parent
            prob_parent1 = 0.5 + 0.5 * np.sin(i * 0.1)
            if random.random() < prob_parent1:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        elif i < n1:
            offspring.append(parent1[i])
        else:
            offspring.append(parent2[i])
    
    return offspring

def simulate_annealing_step(sequence, temperature, objective_func):
    """Simulate a step of simulated annealing to escape local minima."""
    n = len(sequence)
    new_sequence = sequence.copy()
    
    # Perturb a random element
    idx = random.randint(0, n - 1)
    delta = random.gauss(0, temperature * 0.1)
    new_sequence[idx] = max(0.0, new_sequence[idx] + delta)
    
    # Accept or reject based on Metropolis criterion
    old_energy = objective_func(sequence)
    new_energy = objective_func(new_sequence)
    
    if new_energy < old_energy or random.random() < np.exp(-(new_energy - old_energy) / max(1e-8, temperature)):
        return new_sequence
    else:
        return sequence

def adaptive_local_search(current_seq, iteration, max_iter=200):
    """Enhanced local search with adaptive annealing and gradient techniques."""
    n = len(current_seq)
    bounds = [(0.0, 1000.0) for _ in range(n)]
    
    def sum_constraint(x):
        return np.sum(x) - 0.01

    constraints = [{'type': 'ineq', 'fun': sum_constraint}]
    
    def objective(x):
        return evaluate_objective(x)
    
    # Adaptive learning rate based on iteration
    lr = 0.1 * (1 - (iteration / max_iter)) + 0.01
    
    # Hybrid approach: gradient descent + simulated annealing
    try:
        # Start with a small gradient step
        result = minimize(objective, current_seq, method='L-BFGS-B', bounds=bounds, 
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
        
        if result.success:
            optimized_seq = result.x.tolist()
        else:
            optimized_seq = current_seq
    except:
        optimized_seq = current_seq

    # Add simulated annealing step with decreasing temperature
    temp = 1.0 - (iteration / max_iter)
    if temp > 0:
        optimized_seq = simulate_annealing_step(optimized_seq, temp, objective)
    
    return optimized_seq

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence using quantum-inspired evolutionary approach.
    """
    start_time = time.time()
    population_size = 30
    generations = 70
    keep_top = 8
    elite_preservation = 2
    
    # Initialize population with quantum-inspired starting points
    population = []
    for _ in range(population_size):
        n = random.randint(100, 1000)
        # Generate sequence using quantum-like distribution
        sequence = [random.uniform(0.1, 100.0) * (1 + 0.1 * np.sin(i * 0.1)) for i in range(n)]
        population.append(sequence)
    
    # Evaluate initial population
    fitness_scores = []
    for seq in population:
        fitness = evaluate_objective(seq)
        fitness_scores.append((seq, fitness))

    # Sort population by fitness (lower is better)
    fitness_scores.sort(key=lambda x: x[1])

    # Track best solution globally
    global_best = fitness_scores[0][0]
    global_best_fitness = fitness_scores[0][1]

    # Main evolution loop with quantum-inspired operators
    for gen in range(generations):
        if time.time() - start_time > 170:  # Leave 10 seconds for finalization
            break

        # Keep top performers (elite)
        top_performers = [seq for seq, _ in fitness_scores[:keep_top]]
        
        # Create new population
        new_population = top_performers[:]

        # Preserve elites
        if elite_preservation > 0:
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i][1])[:elite_preservation]
            elites = [fitness_scores[i][0] for i in elite_indices]
            new_population.extend(elites)
        
        # Add quantum-inspired mutated versions of top performers
        for i in range(population_size - len(new_population)):
            if random.random() < 0.6:  # 60% chance of quantum mutation
                parent = random.choice(top_performers)
                child = quantum_mutation(parent, 0.1, 0.3)
            else:  # 40% chance of quantum crossover
                p1, p2 = random.sample(top_performers, 2)
                child = quantum_crossover(p1, p2)
            
            new_population.append(child)
        
        # Apply adaptive local search to some individuals
        for i in range(0, len(new_population), 2):
            if random.random() < 0.7:  # 70% chance of local search
                new_population[i] = adaptive_local_search(new_population[i], gen)
        
        # Evaluate new population
        fitness_scores = []
        for seq in new_population:
            fitness = evaluate_objective(seq)
            fitness_scores.append((seq, fitness))
        
        # Sort population by fitness
        fitness_scores.sort(key=lambda x: x[1])
        
        # Update global best
        if fitness_scores[0][1] < global_best_fitness:
            global_best = fitness_scores[0][0]
            global_best_fitness = fitness_scores[0][1]

    # Final optimization of the best sequence
    final_best = adaptive_local_search(global_best, generations, max_iter=300)
    
    # Return the best sequence found
    return final_best

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")