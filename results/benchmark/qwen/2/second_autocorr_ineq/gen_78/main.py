# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import List
from deap import base, creator, tools, algorithms
import time

def construct_function() -> List[float]:
    """Optimized step function construction using evolutionary algorithm with specialized operators."""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Problem parameters
    n_steps = 5000  # Fixed at 5000 to match AlphaEvolve
    max_evaluations = 10000  # Limit evaluations for time constraint
    
    # Initialize DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define individual initialization
    def init_individual():
        # Create step function with random heights between 0 and 2
        return [random.uniform(0, 2) for _ in range(n_steps)]
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define custom crossover operator for step functions
    def cx_steps(ind1, ind2):
        """Crossover that maintains step function properties"""
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(cxpoint1, size)
        
        # Swap segments between individuals
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        
        return ind1, ind2
    
    # Define custom mutation operator for step functions
    def mut_steps(individual, indpb=0.05):
        """Mutation that adjusts step heights with adaptive parameters"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Adaptive mutation: smaller changes near boundaries, larger in middle
                if individual[i] < 0.1 or individual[i] > 1.9:
                    # Boundary regions: small mutations
                    individual[i] += random.gauss(0, 0.05)
                else:
                    # Middle regions: larger mutations
                    individual[i] += random.gauss(0, 0.1)
                # Ensure non-negativity
                individual[i] = max(0, individual[i])
        return individual,
    
    toolbox.register("mate", cx_steps)
    toolbox.register("mutate", mut_steps)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Define evaluation function
    def evaluate(individual):
        # Ensure individual has correct size
        if len(individual) != n_steps:
            individual = individual[:n_steps] + [0] * (n_steps - len(individual))
        
        # Compute autoconvolution using discrete convolution
        f = np.array(individual)
        g = np.convolve(f, f, mode='full')
        g = g[len(g)//2:]  # Take positive part
        
        # Truncate if necessary to match original length
        if len(g) > len(f):
            g = g[:len(f)]
            
        # Compute norms exactly as specified
        norm_2_sq = np.sum(g**2) * (0.5 / len(f))  # Approximate integral
        norm_1 = np.sum(np.abs(g)) / (len(g) + 1)
        norm_inf = np.max(np.abs(g))
        
        # Check for numerical stability
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    
    toolbox.register("evaluate", evaluate)
    
    # Genetic algorithm parameters
    population_size = min(50, max(10, n_steps // 200))
    generations = min(50, max(5, max_evaluations // (population_size * 10)))
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution algorithm
    try:
        # Run evolution with early termination if needed
        hof = tools.HallOfFame(1)
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
            stats=stats, halloffame=hof, verbose=False
        )
        
        # Return best individual
        best_individual = hof[0]
        
        # Ensure it's properly clipped and formatted
        result = [max(0, val) for val in best_individual]
        
        # Add final small noise for robustness
        noise = np.random.normal(0, 0.005, n_steps)
        noisy_result = np.array(result) + noise
        noisy_result = np.maximum(noisy_result, 0)
        
        return noisy_result.tolist()
        
    except Exception as e:
        # Fallback to simple constructed function
        base_function = np.ones(n_steps) * 0.5
        return base_function.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")