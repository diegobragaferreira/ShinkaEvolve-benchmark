# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit
import multiprocessing as mp
from functools import partial

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: list[float]) -> tuple[float, float, float]:
    """
    Fast computation of the three norms needed for C2 calculation using piecewise linear integration.
    """
    # Convert to numpy array for easier manipulation
    f = np.array(f_values)
    n_steps = len(f)
    
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n_steps

    # Compute autoconvolution using discrete convolution
    g = np.convolve(f, f, mode='full')
    # Trim g to the correct size (this accounts for the convolution)
    g = g[len(f)-1:2*len(f)-1]

    # Compute L2 norm squared using piecewise linear integration
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        # Trapezoidal-like integration for quadratic function
        # Using formula for integral of ax^2 + bx + c over [x0,x1]
        # But here we approximate with piecewise linear segments
        # So we use: (dx/3)(y0^2 + y0*y1 + y1^2)
        y0, y1 = g[i], g[i+1]
        norm_2_squared += (dx/3) * (y0**2 + y0*y1 + y1**2)

    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g))

    # Infinity norm
    norm_inf = np.max(np.abs(g))

    # Handle numerical edge cases
    if norm_1 <= 1e-15:
        norm_1 = 1e-15
    if norm_inf <= 1e-15:
        norm_inf = 1e-15

    return norm_2_squared, norm_1, norm_inf

class FunctionConstructor:
    """Handles the construction of various function types for optimization."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    def _generate_basis_functions(self, n_steps: int, n_components: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Generate basis functions with controlled characteristics."""
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Generate a mix of different basis types
        basis_functions = []
        basis_weights = []
        
        # Central peak with Gaussian shape
        gaussian_peak = np.exp(-15 * x**2)
        basis_functions.append(gaussian_peak)
        basis_weights.append(1.0)
        
        # Multiple frequency components
        for i in range(n_components):
            freq = 2 + i * 0.5
            if i % 2 == 0:
                basis_functions.append(np.sin(2 * np.pi * freq * x))
                basis_weights.append(0.5)
            else:
                basis_functions.append(np.cos(2 * np.pi * freq * x))
                basis_weights.append(0.5)
        
        # Add some polynomial components
        for i in range(1, 4):
            poly = x**i
            basis_functions.append(poly)
            basis_weights.append(0.3)
        
        return np.array(basis_functions).T, np.array(basis_weights)
    
    def create_adaptive_function(self, n_steps: int = 1000) -> np.ndarray:
        """Create an adaptive function with strategic distribution of components."""
        # Generate basis functions
        basis, weights = self._generate_basis_functions(n_steps)
        
        # Create function as weighted combination of basis functions
        # Use random weights with some structure
        component_weights = np.random.rand(len(weights)) * weights
        
        # Apply non-negative constraint and normalize
        f = np.dot(basis, component_weights)
        f = np.maximum(f, 0)
        
        # Normalize to reasonable scale
        if np.sum(f) > 0:
            f = f / np.sum(f) * 100
            
        return f
    
    def create_spectral_function(self, n_steps: int = 1000) -> np.ndarray:
        """Create function based on spectral properties for better C2 optimization."""
        # Use a combination of Gaussian and sinc-like functions
        x = np.linspace(-0.25, 0.25, n_steps)
        
        # Central region with Gaussian concentration
        f = np.exp(-20 * x**2)
        
        # Add some periodic structure to avoid degenerate solutions
        f += 0.5 * np.sin(10 * np.pi * x) * np.exp(-x**2/0.1)
        
        # Add controlled noise
        noise = np.random.normal(0, 0.05, n_steps)
        f += noise
        
        # Non-negative constraint
        f = np.maximum(f, 0)
        
        # Normalize
        if np.sum(f) > 0:
            f = f / np.sum(f) * 100
            
        return f

class EvolutionaryOptimizer:
    """Handles evolutionary optimization with improved parallel processing."""
    
    def __init__(self, max_evaluations: int = 500, max_time: int = 85):
        self.max_evaluations = max_evaluations
        self.max_time = max_time
        self.setup_deap()
    
    def setup_deap(self):
        """Set up DEAP evolutionary framework."""
        random.seed(42)
        np.random.seed(42)
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -10, 10)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, n=20)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_individual(self, individual: list[float]) -> tuple[float,]:
        """Evaluate a single individual with enhanced error handling."""
        try:
            # Convert individual to function using spectral composition
            n_steps = 1000
            x = np.linspace(-0.25, 0.25, n_steps)
            f = np.zeros(n_steps)
            
            # Reconstruct function from individual
            max_freq = 20
            frequencies = np.linspace(1, max_freq, len(individual)//2)
            
            # Alternate between sine and cosine
            for i in range(0, len(individual), 2):
                amp = abs(individual[i])
                freq = frequencies[i//2] if i//2 < len(frequencies) else frequencies[-1]
                if i//2 % 2 == 0:
                    f += amp * np.sin(2 * np.pi * freq * x)
                else:
                    f += amp * np.cos(2 * np.pi * freq * x)
            
            # Ensure non-negative values and normalize
            f = np.maximum(f, 0)
            if np.sum(f) > 0:
                f = f / np.sum(f) * 100
            
            # Compute norms
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(f.tolist())
            
            if norm_1 <= 1e-15 or norm_inf <= 1e-15:
                return (0,)
            
            c2 = norm_2_sq / (norm_1 * norm_inf)
            return (c2,)
        except Exception:
            return (0,)
    
    def optimize(self, function_constructor: FunctionConstructor) -> np.ndarray:
        """Perform evolutionary optimization with time constraints."""
        start_time = time.time()
        
        # Initialize population with diverse starting functions
        pop_size = 50
        pop = []
        
        # Create diverse initial population
        for _ in range(pop_size):
            # Mix of different construction approaches
            if random.random() < 0.5:
                func = function_constructor.create_adaptive_function()
            else:
                func = function_constructor.create_spectral_function()
            
            # Convert to individual format for DEAP
            individual = func.tolist()[:20]  # Truncate to fit expected length
            individual.extend([random.uniform(-10, 10) for _ in range(20 - len(individual))])
            pop.append(individual)
        
        # Run evolution with early termination
        generation = 0
        max_generations = 20
        best_fitness = float('-inf')
        
        try:
            for gen in range(max_generations):
                if time.time() - start_time > self.max_time:
                    break
                    
                # Evaluate fitness for all individuals
                fitnesses = [self.evaluate_individual(ind) for ind in pop]
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit
                
                # Select, crossover, and mutate
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover and mutation
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < 0.2:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate new individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = [self.evaluate_individual(ind) for ind in invalid_ind]
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace old population
                pop[:] = offspring
                
                # Track best fitness
                current_best = max([ind.fitness.values[0] for ind in pop if ind.fitness.values])
                if current_best > best_fitness:
                    best_fitness = current_best
                    
        except Exception as e:
            pass  # Continue with fallback
        
        # Return best solution found
        try:
            best_ind = tools.selBest(pop, 1)[0] if pop else []
            if best_ind:
                # Convert back to function representation
                n_steps = 1000
                x = np.linspace(-0.25, 0.25, n_steps)
                f = np.zeros(n_steps)
                
                max_freq = 20
                frequencies = np.linspace(1, max_freq, len(best_ind)//2)
                
                for i in range(0, len(best_ind), 2):
                    amp = abs(best_ind[i])
                    freq = frequencies[i//2] if i//2 < len(frequencies) else frequencies[-1]
                    if i//2 % 2 == 0:
                        f += amp * np.sin(2 * np.pi * freq * x)
                    else:
                        f += amp * np.cos(2 * np.pi * freq * x)
                
                f = np.maximum(f, 0)
                if np.sum(f) > 0:
                    f = f / np.sum(f) * 100
                return f
        except Exception:
            pass
        
        # Fallback to constructor-generated function
        return function_constructor.create_spectral_function()

def construct_function() -> list[float]:
    """Main function to construct optimized step function."""
    
    # Initialize components
    function_constructor = FunctionConstructor(seed=42)
    optimizer = EvolutionaryOptimizer(max_time=85)
    
    # Try multiple approaches to maximize C2
    best_c2 = -1
    best_function = None
    start_time = time.time()
    
    try:
        # Attempt evolutionary optimization
        evolved_func = optimizer.optimize(function_constructor)
        if evolved_func is not None:
            # Evaluate the evolved function
            norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(evolved_func.tolist())
            if norm_1 > 1e-15 and norm_inf > 1e-15:
                c2 = norm_2_sq / (norm_1 * norm_inf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_function = evolved_func.tolist()
    except Exception:
        pass
    
    # Additional fallback approach
    if best_function is None:
        # Try direct function construction
        try:
            # Try a few different construction approaches
            for i in range(5):
                if time.time() - start_time > 80:
                    break
                    
                func = function_constructor.create_spectral_function()
                norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_fast(func.tolist())
                if norm_1 > 1e-15 and norm_inf > 1e-15:
                    c2 = norm_2_sq / (norm_1 * norm_inf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_function = func.tolist()
        except Exception:
            pass
    
    # Final fallback
    if best_function is None:
        n_steps = 1000
        best_function = [1.0] * n_steps
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")