# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import random
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

def compute_autocorrelation_constant(sequence):
    """Compute C₁ for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    a = np.array(sequence)
    n = len(a)

    # Compute convolution using FFT for efficiency
    b = signal.convolve(a, a, mode='full')
    max_conv = np.max(b)

    # Compute C₁ = 2n * max(b) / (sum(a))^2
    sum_a = np.sum(a)

    # Avoid division by zero or very small values
    if sum_a < 1e-10:
        return float('inf')

    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluate a sequence for the optimization problem."""
    # Ensure we have a valid sequence with sufficient sum
    if len(sequence) == 0:
        return float('-inf')

    # Clip values to [0, 1000] as per constraints
    sequence = np.clip(sequence, 0, 1000)

    # Check sum constraint
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('-inf')

    # Compute C₁
    c1 = compute_autocorrelation_constant(sequence)

    # Return 1/C₁ (we want to maximize this)
    # If C₁ is very large, 1/C₁ approaches 0
    if c1 > 1e10:
        return float('-inf')

    return 1.0 / c1

def generate_multi_scale_sequence():
    """Generate a sequence with multi-scale characteristics to enhance exploration."""
    n = random.randint(100, 1000)
    
    # Choose initialization strategy
    strategy = random.choice(['geometric', 'spike', 'uniform'])
    
    if strategy == 'geometric':
        # Geometric decay sequence
        sequence = [0.9 ** i for i in range(n)]
    elif strategy == 'spike':
        # Spike sequence with one prominent peak
        sequence = [0.0] * n
        peak_pos = random.randint(0, n-1)
        sequence[peak_pos] = random.uniform(100, 1000)
    else:
        # Uniform random sequence
        sequence = np.random.uniform(0, 1000, n).tolist()
    
    return sequence

def run_evolutionary_optimization(sequence, max_generations=20):
    """Run evolutionary optimization on the sequence."""
    n = len(sequence)
    if n < 10:
        return sequence

    # Set up evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1000)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_sequence)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=20)
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolve
    for generation in range(max_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values
            del child2.fitness.values
            
        new_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, new_individuals))
        for ind, fit in zip(new_individuals, fitnesses):
            ind.fitness.values = fit
            
        population[:] = offspring
    
    # Return best individual
    best_ind = tools.selBest(population, 1)[0]
    return list(best_ind)

def refine_with_local_search(sequence, max_iter=30):
    """Refine the sequence using local search."""
    n = len(sequence)
    if n < 10:
        return sequence

    # Objective function for local refinement
    def objective(x):
        return -evaluate_sequence(list(x))
    
    bounds = [(0, 1000) for _ in range(n)]
    
    try:
        # Use differential evolution for local search
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=10,
            seed=42,
            disp=False,
            tol=1e-6
        )
        
        if result.success:
            refined_seq = list(result.x)
            refined_seq = [max(0, min(1000, val)) for val in refined_seq]
            return refined_seq
    except Exception:
        pass
    
    return sequence

def optimize_sequence():
    """Optimize the sequence using a hybrid evolutionary and local search approach."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    best_score = float('-inf')
    best_sequence = None

    # Try multiple random starts to avoid local optima
    num_starts = 10

    for start in range(num_starts):
        # Generate a multi-scale starting sequence
        sequence = generate_multi_scale_sequence()

        # First, run evolutionary optimization
        evolved_sequence = run_evolutionary_optimization(sequence)
        
        # Then, refine with local search
        refined_sequence = refine_with_local_search(evolved_sequence)
        
        # Evaluate the final sequence
        final_score = evaluate_sequence(refined_sequence)
        
        if final_score > best_score:
            best_score = final_score
            best_sequence = refined_sequence

    # Final check to ensure we found a good solution
    if best_sequence is not None:
        # Evaluate one more time to make sure
        final_score = evaluate_sequence(best_sequence)
        return best_sequence, final_score
    else:
        # If no good solution found, return the initial random sequence
        initial_seq = generate_multi_scale_sequence()
        initial_score = evaluate_sequence(initial_seq)
        return initial_seq, initial_score

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence."""
    start_time = time.time()

    # Run optimization
    best_sequence, best_score = optimize_sequence()

    end_time = time.time()
    eval_time = end_time - start_time

    # Calculate benchmark ratio
    benchmark_ratio = best_score / 0.6653  # 1.5031 is the threshold for C₁

    print(f"Best 1/C₁: {best_score:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Execution time: {eval_time:.4f} seconds")

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")