# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, njit
import random
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.8
CROSSOVER_PROB = 0.7
NUM_STARTS = 5
MAX_EVALUATIONS = 10000

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the three norms of the autoconvolution g = f*f
    Returns ||g||₂², ||g||₁, ||g||∞
    """
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Compute autoconvolution g = f*f
    # Using discrete convolution with proper indexing
    g = np.zeros(2*n - 1)
    
    # For each element in output convolved array
    for i in range(2*n - 1):
        # Sum over valid indices
        start_idx = max(0, i - n + 1)
        end_idx = min(i, n - 1) + 1
        sum_val = 0.0
        for j in range(start_idx, end_idx):
            k = i - j
            if 0 <= k < n:
                sum_val += f_vals[j] * f_vals[k]
        g[i] = sum_val
    
    # Compute norms
    g_squared = g * g
    norm_l2_sq = np.sum(g_squared)
    
    norm_l1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))
    
    return norm_l2_sq, norm_l1, norm_inf

@njit
def compute_c2_score(f_vals):
    """
    Compute the C2 score: ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_l2_sq, norm_l1, norm_inf = compute_autoconvolution_norms(f_vals)
    
    # Avoid division by zero
    if norm_l1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0
    
    return norm_l2_sq / (norm_l1 * norm_inf)

def sophisticated_initialization(n):
    """Create a good initial step function with diverse strategies"""
    # Strategy 1: Alternating high-low pattern with some randomness
    if random.random() < 0.4:
        pattern = []
        for i in range(n):
            if i % 2 == 0:
                pattern.append(1.0 + random.random() * 0.5)
            else:
                pattern.append(0.1 + random.random() * 0.2)
        
    # Strategy 2: Gaussian-weighted peaks
    elif random.random() < 0.4:
        pattern = []
        center = n // 2
        std = n / 6.0
        for i in range(n):
            x = (i - center) / std
            val = np.exp(-0.5 * x * x) * (0.8 + 0.2 * random.random())
            pattern.append(max(0.0, val))
    
    # Strategy 3: Uniform distribution
    else:
        pattern = [0.5 + random.random() * 0.5 for _ in range(n)]
    
    return pattern

def evolutionary_optimization():
    """Main evolutionary optimization function"""
    best_c2 = 0.0
    best_f = None
    
    # Try multiple random starts
    for start_num in range(NUM_STARTS):
        # Initialize population with diverse strategies
        population = []
        pop_size = POPULATION_SIZE
        
        for i in range(pop_size):
            n = random.randint(100, 1000)
            individual = sophisticated_initialization(n)
            # Ensure non-negative values
            individual = [max(0.0, x) for x in individual]
            population.append(individual)
        
        # Differential evolution with custom bounds
        bounds = [(0.0, 2.0) for _ in range(len(population[0]))]
        
        def objective(x):
            # Convert to list of floats
            f_vals = [float(xi) for xi in x]
            try:
                c2 = compute_c2_score(f_vals)
                return -c2  # Negative because we maximize
            except:
                return 1e10  # Penalty for invalid solutions
        
        try:
            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=GENERATIONS,
                popsize=pop_size,
                mutation=MUTATION_RATE,
                recombination=CROSSOVER_PROB,
                seed=start_num,
                disp=False
            )
            
            if result.success:
                final_solution = [max(0.0, float(x)) for x in result.x]
                final_c2 = compute_c2_score(final_solution)
                
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_f = final_solution
                    
        except Exception as e:
            continue
    
    # If we found a solution, apply local refinement with L-BFGS-B
    if best_f is not None and best_c2 > 0:
        try:
            # Create bounds for the optimization
            bounds = [(0.0, 2.0) for _ in range(len(best_f))]
            
            # Local optimization
            def local_objective(x):
                # Ensure non-negativity
                x = [max(0.0, xi) for xi in x]
                return -compute_c2_score(x)
            
            local_result = minimize(
                local_objective,
                best_f,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            refined_solution = [max(0.0, float(x)) for x in local_result.x]
            refined_c2 = compute_c2_score(refined_solution)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_f = refined_solution
                
        except Exception as e:
            pass
    
    return best_f if best_f is not None else sophisticated_initialization(500)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the optimization
    start_time = time.time()
    try:
        f_values = evolutionary_optimization()
        elapsed_time = time.time() - start_time
        # Add safety check to ensure valid output
        if f_values is None or len(f_values) == 0:
            # Fallback to simple uniform distribution
            f_values = [0.5] * 200
    except:
        # Final fallback
        f_values = [0.5] * 200
    
    # Ensure we return a reasonable-sized list
    if len(f_values) < 50:
        f_values = f_values + [0.5] * (50 - len(f_values))
    elif len(f_values) > 10000:
        f_values = f_values[:10000]
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
