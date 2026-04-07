# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms
import copy

# Core computation module with JIT compilation for maximum performance
@jit(nopython=True)
def compute_autoconvolution_jit(f_vals, step_width):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size)

    # Compute convolution directly
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True)
def compute_norms_jit(g_vals):
    """
    Compute norms efficiently with numba
    """
    n = len(g_vals)

    # Compute L1 norm (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])

    # Compute L2 norm squared
    l2_norm_sq = 0.0
    for i in range(n):
        l2_norm_sq += g_vals[i] * g_vals[i]

    # Compute infinity norm
    linf_norm = 0.0
    for i in range(n):
        val = abs(g_vals[i])
        if val > linf_norm:
            linf_norm = val

    return l1_norm, l2_norm_sq, linf_norm

def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using direct computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution using direct convolution sum
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

# Pattern-based initialization functions
def generate_patterned_initial_function(n_steps):
    """Generate an initial function based on mathematical insight about optimal convolution shapes"""
    # Create a function designed to produce uniform convolution profiles
    # This pattern balances peak and flat regions to encourage good C2 values

    f_values = []

    # Create a pattern that starts low, rises to a peak, then falls back down
    # but with enough variation to be interesting
    half = n_steps // 2
    quarter = n_steps // 4

    # Base pattern with multiple regions
    for i in range(n_steps):
        if i < quarter:
            # Rising edge
            f_values.append(i / quarter)
        elif i < half:
            # Peak region
            f_values.append(1.0)
        elif i < 3*quarter:
            # Falling edge
            f_values.append((3*quarter - i) / quarter)
        else:
            # Low tail
            f_values.append((n_steps - i) / quarter)

    # Apply some smoothing to reduce sharp transitions
    smoothed = []
    for i in range(n_steps):
        if i == 0 or i == n_steps - 1:
            smoothed.append(f_values[i])
        else:
            # Weighted average
            smoothed.append(0.2 * f_values[i-1] + 0.6 * f_values[i] + 0.2 * f_values[i+1])

    # Normalize to ensure reasonable magnitude
    total_area = sum(smoothed) * (0.5 / n_steps)
    if total_area > 0:
        smoothed = [x / total_area * 2.0 for x in smoothed]

    return smoothed

def generate_geometric_initial_function(n_steps):
    """Generate geometric initial function for diversity"""
    x = np.linspace(-0.25, 0.25, n_steps)
    return np.exp(-0.5 * (x / 0.1) ** 2).tolist()

def generate_uniform_initial_function(n_steps):
    """Generate uniform initial function for baseline"""
    return [1.0] * n_steps

# Optimized evaluation system
class FitnessEvaluator:
    def __init__(self):
        self.cache = {}
        self.eval_count = 0

    def evaluate(self, individual):
        """Evaluate fitness of individual with caching and error handling"""
        # Create cache key
        key = tuple(individual)
        if key in self.cache:
            return self.cache[key]

        try:
            # Ensure non-negative values
            individual = np.maximum(0.0, individual)
            c2 = compute_c2(individual.tolist())

            # Cache result
            self.cache[key] = -c2  # Negative because we want to maximize C2
            self.eval_count += 1
            return -c2

        except Exception as e:
            # Return penalty for invalid solutions
            return 1e10

# Evolutionary optimization system
class EvolutionaryOptimizer:
    def __init__(self, n_steps, pop_size=30, generations=30):
        self.n_steps = n_steps
        self.pop_size = pop_size
        self.generations = generations
        self.evaluator = FitnessEvaluator()

        # Setup DEAP toolbox
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluator.evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._adaptive_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _generate_individual(self):
        """Generate individual with mixed initialization strategies"""
        strategy = random.choice(['pattern', 'geometric', 'uniform'])

        if strategy == 'pattern':
            return generate_patterned_initial_function(self.n_steps)
        elif strategy == 'geometric':
            return generate_geometric_initial_function(self.n_steps)
        else:
            return generate_uniform_initial_function(self.n_steps)

    def _adaptive_mutation(self, individual, gen_num=0):
        """Mutate an individual with adaptive rate"""
        # Decrease mutation rate over generations - start high, decrease to low
        adaptive_mut_rate = 0.1 * (1.0 - 0.05 * gen_num / 100)
        adaptive_mut_rate = max(adaptive_mut_rate, 0.01)

        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < adaptive_mut_rate:
                # Apply small Gaussian perturbation
                change = np.random.normal(0, 0.1 * mutated[i])
                mutated[i] = max(0, mutated[i] + change)
        return mutated

    def optimize(self):
        """Perform evolutionary optimization with early stopping"""
        # Create initial population
        pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)

        # Track best solution
        best_fitness = max(fitnesses)
        best_individual = pop[np.argmax(fitnesses)].copy()

        # Evolution loop with early stopping
        for gen in range(self.generations):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < 0.1:  # Mutation probability
                    self.toolbox.mutate(mutant, gen_num=gen)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

            # Replace the old population with the new one
            pop[:] = offspring

            # Update best solution
            current_best_fitness = max([ind.fitness.values[0] for ind in pop])
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = pop[np.argmax([ind.fitness.values[0] for ind in pop])].copy()

            # Early stopping: if no improvement in last few generations
            if gen > 5 and current_best_fitness <= best_fitness:
                # Check if we're plateauing
                recent_improvements = [current_best_fitness] * 3
                if all(f <= best_fitness for f in recent_improvements):
                    break

        return np.array(best_individual)

# Advanced local refinement with time management
def advanced_local_refinement(initial_individual, max_time, start_time):
    """Refine solution with adaptive local search"""
    refined_individual = initial_individual.copy()
    old_c2 = compute_c2(refined_individual.tolist())

    # Time-based refinement loop
    iteration = 0
    while time.time() - start_time < max_time and iteration < 15:
        improved = False

        # Sample a subset of indices for efficiency
        sample_size = min(10, len(refined_individual) // 3)
        sample_indices = np.random.choice(len(refined_individual), sample_size, replace=False)

        for i in sample_indices:
            if time.time() - start_time > max_time:
                break

            # Try small perturbations
            original_value = refined_individual[i]
            step_sizes = [0.01, 0.05, 0.1]

            for step in step_sizes:
                # Try increasing and decreasing
                for direction in [1, -1]:
                    if time.time() - start_time > max_time:
                        break
                    test_individual = refined_individual.copy()
                    new_val = original_value + direction * step
                    test_individual[i] = max(0, new_val)

                    new_c2 = compute_c2(test_individual.tolist())
                    if new_c2 > old_c2:
                        refined_individual = test_individual
                        old_c2 = new_c2
                        improved = True

        if not improved:
            break
        iteration += 1

    return refined_individual

# Main optimization workflow
def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimized hybrid approach."""
    # Set parameters
    n_steps = 200
    max_time = 85  # seconds
    start_time = time.time()

    # Track evaluation count
    eval_count = 0

    # Phase 1: Multi-resolution adaptive initialization
    best_solution = None
    best_c2 = -np.inf

    # Try different initialization patterns
    init_strategies = [
        ('pattern', generate_patterned_initial_function),
        ('geometric', generate_geometric_initial_function),
        ('uniform', generate_uniform_initial_function)
    ]

    for strategy_name, strategy_func in init_strategies:
        try:
            # Generate initial function
            f_values = strategy_func(n_steps)

            # Normalize for better numerical behavior
            total = sum(f_values)
            if total > 0:
                f_values = [x / total * 10 for x in f_values]

            # Evaluate
            current_c2 = compute_c2(f_values)
            eval_count += 1

            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = f_values.copy()

        except Exception as e:
            continue

    # Phase 2: Evolutionary optimization
    try:
        optimizer = EvolutionaryOptimizer(n_steps, pop_size=20, generations=20)
        evolved_individual = optimizer.optimize()

        # Evaluate evolved solution
        evolved_c2 = compute_c2(evolved_individual.tolist())
        eval_count += optimizer.evaluator.eval_count

        if evolved_c2 > best_c2:
            best_c2 = evolved_c2
            best_solution = evolved_individual.tolist()

    except Exception as e:
        pass

    # Phase 3: Advanced local refinement
    if best_solution is not None:
        refined_solution = advanced_local_refinement(
            np.array(best_solution), max_time, start_time
        )
        best_solution = refined_solution.tolist()

    # Ensure final solution is properly formatted
    final_solution = np.maximum(0.0, np.array(best_solution))

    # Normalize for better numerical behavior
    if np.sum(final_solution) > 0:
        final_solution = final_solution / np.sum(final_solution) * 10

    return final_solution.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")