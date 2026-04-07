# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import random
import time
from collections import deque

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1, max_mutation=0.5):
    """Mutate a sequence by randomly modifying elements."""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            new_sequence[i] *= random.uniform(1 - max_mutation, 1 + max_mutation)
            new_sequence[i] = max(0, min(1000, new_sequence[i]))
    if all(x == 0 for x in new_sequence):
        new_sequence[0] = max(0.1, new_sequence[0])
    return new_sequence

def crossover_sequences(seq1, seq2):
    """Perform uniform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    padded_seq1 = seq1 + [0] * (max_len - len(seq1))
    padded_seq2 = seq2 + [0] * (max_len - len(seq2))
    new_seq = []
    for i in range(max_len):
        if random.random() < 0.5:
            new_seq.append(padded_seq1[i])
        else:
            new_seq.append(padded_seq2[i])
    return new_seq

def genetic_algorithm_search(max_time_seconds=180, pop_size=50):
    """Search using genetic algorithm approach."""
    start_time = time.time()
    population = [generate_random_sequence() for _ in range(pop_size)]
    best_individual = None
    best_fitness = 0.0
    generation = 0
    while time.time() - start_time < max_time_seconds:
        generation += 1
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence(individual)
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        selected = []
        for _ in range(pop_size):
            tournament_indices = random.sample(range(pop_size), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % pop_size]
            child1 = crossover_sequences(parent1, parent2)
            child2 = crossover_sequences(parent2, parent1)
            child1 = mutate_sequence(child1)
            child2 = mutate_sequence(child2)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]
        if generation % 10 == 0:
            for i in range(0, pop_size, 5):
                if i < pop_size:
                    population[i] = generate_random_sequence()
    return (best_individual, best_fitness)

def local_improvement_search(initial_sequence, max_iter=100):
    """Improve a sequence using local search around it."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    temp = 1.0
    cooling_rate = 0.95
    min_temp = 1e-4
    for iteration in range(max_iter):
        mutated = mutate_sequence(current_sequence, mutation_rate=0.3, max_mutation=0.2)
        mutated_fitness = evaluate_sequence(mutated)
        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness
        else:
            if random.random() < np.exp((mutated_fitness - current_fitness) / (temp + 1e-10)):
                current_sequence = mutated
                current_fitness = mutated_fitness
        if current_fitness > best_fitness:
            best_sequence = current_sequence.copy()
            best_fitness = current_fitness
        temp = max(temp * cooling_rate, min_temp)
    return best_sequence, best_fitness

def quadratic_programming_optimization(initial_sequence, max_iter=100):
    """Optimizes the sequence using Quadratic Programming approach."""
    def objective(x):
        seq = np.abs(x)
        if np.sum(seq) < 0.01:
            return float('inf')
        c1 = compute_c1_constant(seq)
        if c1 == float('inf'):
            return float('inf')
        return -1.0 / c1

    bounds = [(0, 1000) for _ in range(len(initial_sequence))]
    x0 = np.array(initial_sequence) + np.random.normal(0, 0.1, len(initial_sequence))
    x0 = np.maximum(x0, 0)
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        if res.success:
            optimized_seq = np.abs(res.x)
            return optimized_seq.tolist(), evaluate_sequence(optimized_seq)
    except:
        pass
    return initial_sequence, evaluate_sequence(initial_sequence)

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    start_time = time.time()
    best_sequence = None
    best_inv_c1 = 0.0
    best_history = deque(maxlen=5)
    
    for attempt in range(20):
        if time.time() - start_time > 170:
            break
        initial_sequence = generate_random_sequence()
        ga_seq, ga_fitness = genetic_algorithm_search(5, pop_size=30)
        if ga_fitness > best_inv_c1:
            best_inv_c1 = ga_fitness
            best_sequence = ga_seq
        local_seq, local_fitness = local_improvement_search(initial_sequence, 50)
        if local_fitness > best_inv_c1:
            best_inv_c1 = local_fitness
            best_sequence = local_seq
        qp_seq, qp_fitness = quadratic_programming_optimization(initial_sequence, 50)
        if qp_fitness > best_inv_c1:
            best_inv_c1 = qp_fitness
            best_sequence = qp_seq
        if best_sequence is not None and evaluate_sequence(best_sequence) > 0.6653:
            break  # Early termination if benchmark is beaten
            
    if best_sequence is None:
        best_sequence = generate_random_sequence()
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")