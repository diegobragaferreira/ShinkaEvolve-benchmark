# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
import random
import time
from scipy.optimize import differential_evolution
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution efficiently using numba"""
    n = len(f_vals)
    # Create output array for autoconvolution
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually with numba optimization
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@numba.jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with numba"""
    n = len(g_vals)
    
    # L2 norm squared (using trapezoidal-like scheme)
    l2_sq = 0.0
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])
    
    # L-infinity norm
    linf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val
    
    return l2_sq, l1, linf

def evaluate_individual(individual):
    """Evaluate fitness of an individual (step function)"""
    try:
        # Convert to numpy array and ensure non-negative
        f_vals = np.array(individual, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)
        
        # Skip if all zeros
        if np.sum(f_vals) == 0:
            return (0.0,)
            
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return (0.0,)
        
        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return (c2,)
    except:
        return (0.0,)

def harmonic_peak_construction(n_steps=None):
    """
    Construct function using harmonic peak pattern that optimizes autoconvolution.
    This approach builds on mathematical insights about how symmetric patterns
    in the original function translate to desirable properties in autoconvolution.
    """
    if n_steps is None:
        n_steps = random.randint(300, 1000)
    
    # Create a base function with carefully positioned harmonic peaks
    f_vals = np.zeros(n_steps)
    
    # Use a pattern based on sine harmonics which often produce good autoconvolutions
    # We place peaks in a way that creates constructive interference in convolution
    n_peaks = max(3, min(15, n_steps // 50))
    
    for i in range(n_peaks):
        # Position peaks with geometric spacing to avoid too regular patterns
        # but still maintain enough structure for good autoconvolution
        if i == 0:
            center = random.uniform(0.1 * n_steps, 0.3 * n_steps)
        elif i == n_peaks - 1:
            center = random.uniform(0.7 * n_steps, 0.9 * n_steps)
        else:
            # Distribute middle peaks with some randomness
            prev_center = (n_peaks - 1) * (0.5 * n_steps) / (n_peaks - 1) if n_peaks > 1 else 0.5 * n_steps
            center = random.uniform(prev_center + n_steps / (n_peaks * 2), 
                                  prev_center + n_steps / (n_peaks * 1.5))
        
        # Adjust center to stay within bounds
        center = max(0, min(n_steps - 1, center))
        
        # Width and height determined by harmonic relationships
        width = max(5, min(50, n_steps // (2 * (i + 1) + 1)))
        height = random.uniform(0.8, 2.5) * (1.0 / (i + 1))  # Harmonic decay
        
        # Create Gaussian-like peaks with varying shapes
        x = np.arange(n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / width) ** 2)
        f_vals += gaussian
    
    # Apply smoothing with adaptive window size
    if n_steps > 50:
        window_size = min(51, n_steps - 1)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size > 1:
            f_vals = signal.savgol_filter(f_vals, window_size, 3)
    
    # Ensure non-negativity and normalize
    f_vals = np.maximum(f_vals, 0)
    
    # Apply final constraints to prevent extreme values
    if np.max(f_vals) > 0:
        # Cap extreme values to prevent autoconvolution spikes
        threshold = np.percentile(f_vals, 95)
        f_vals = np.minimum(f_vals, threshold * 3.0)
        f_vals = f_vals / np.max(f_vals) * 2.0 if np.max(f_vals) > 0 else f_vals
    
    return f_vals.tolist()

def adaptive_harmonic_evolution(n_generations=30):
    """
    Evolutionary approach focused on harmonic peak structures
    """
    # Create initial population with harmonic constructions
    population = []
    pop_size = 25
    
    for _ in range(pop_size):
        individual = harmonic_peak_construction()
        population.append(individual)
    
    best_individual = None
    best_fitness = 0
    
    # Evolution loop
    for generation in range(n_generations):
        # Evaluate population
        fitnesses = list(map(evaluate_individual, population))
        
        # Track best
        for i, (ind, fit) in enumerate(zip(population, fitnesses)):
            if fit[0] > best_fitness:
                best_fitness = fit[0]
                best_individual = ind.copy()
        
        # Selection and reproduction
        # Tournament selection
        selected = []
        for _ in range(pop_size):
            tournament = random.sample(list(zip(population, fitnesses)), 3)
            winner = max(tournament, key=lambda x: x[1][0])
            selected.append(winner[0].copy())
        
        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            
            # Crossover - blend harmonic characteristics
            child1 = blend_harmonic_functions(parent1, parent2)
            child2 = blend_harmonic_functions(parent2, parent1)
            
            # Mutation with harmonic awareness
            child1 = mutate_harmonic_function(child1)
            child2 = mutate_harmonic_function(child2)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        population = offspring[:pop_size]
    
    return best_individual if best_individual is not None else []

def blend_harmonic_functions(func1, func2):
    """Blend two harmonic functions by averaging their elements"""
    blended = []
    min_len = min(len(func1), len(func2))
    
    for i in range(min_len):
        # Blend with preference toward higher values
        blended_val = (func1[i] + func2[i]) / 2.0
        blended.append(blended_val)
    
    # Append remaining elements if lengths differ
    if len(func1) > min_len:
        blended.extend(func1[min_len:])
    elif len(func2) > min_len:
        blended.extend(func2[min_len:])
    
    return blended

def mutate_harmonic_function(func):
    """Apply mutation that maintains harmonic properties"""
    mutated = func.copy()
    
    # Apply mutations with harmonic-aware approach
    for i in range(len(mutated)):
        if random.random() < 0.15:  # 15% chance to mutate
            # Choose mutation type based on current value
            if mutated[i] > 0.1:
                # Small additive mutation
                delta = random.gauss(0, 0.1 * mutated[i])
                mutated[i] = max(0, mutated[i] + delta)
            else:
                # Multiplicative mutation for small values
                factor = random.uniform(0.8, 1.2)
                mutated[i] = max(0, mutated[i] * factor)
    
    return mutated

def local_improvement_search(individual, max_iter=20):
    """
    Local search using differential evolution to fine-tune harmonic structure
    """
    if not individual or len(individual) < 10:
        return individual
    
    try:
        # Convert to array for optimization
        x0 = np.array(individual)
        
        # Define bounds
        bounds = [(0, 5) for _ in range(len(x0))]
        
        # Objective function
        def obj_func(x):
            x = np.maximum(x, 0)
            score = evaluate_individual(x.tolist())[0]
            return -score if score > 0 else 1e10  # Minimize negative of score
        
        # Run differential evolution for local refinement
        result = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=50,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result.success:
            refined = np.maximum(result.x, 0).tolist()
            if evaluate_individual(refined)[0] > evaluate_individual(individual)[0]:
                return refined
                
    except:
        pass
    
    return individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Phase 1: Harmony-based initialization
    best_result = []
    best_c2 = 0
    
    # Try several harmonic constructions
    for attempt in range(5):
        try:
            # Create harmonic peak function
            harmonic_func = harmonic_peak_construction()
            f_vals = np.array(harmonic_func, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                
                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = harmonic_func.copy()
        except:
            continue
    
    # Phase 2: Evolutionary optimization of harmonic structure
    if time.time() - start_time < 65:
        try:
            evolved_result = adaptive_harmonic_evolution(25)
            if evolved_result:
                f_vals = np.array(evolved_result, dtype=np.float64)
                f_vals = np.maximum(f_vals, 0.0)
                if np.sum(f_vals) > 0:
                    g_vals = compute_autoconvolution_numba(f_vals)
                    l2_sq, l1, linf = compute_norms_numba(g_vals)
                    
                    if l1 > 1e-15 and linf > 1e-15:
                        c2 = l2_sq / (l1 * linf)
                        if c2 > best_c2:
                            best_c2 = c2
                            best_result = evolved_result
        except Exception as e:
            pass
    
    # Phase 3: Local improvement
    if best_result and time.time() - start_time < 75:
        try:
            improved_result = local_improvement_search(best_result)
            f_vals = np.array(improved_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = improved_result
        except:
            pass
    
    # Phase 4: Fallback if nothing is good enough
    if len(best_result) == 0 or best_c2 < 0.85:
        # Use a more robust harmonic construction
        n_steps = random.randint(400, 800)
        best_result = harmonic_peak_construction(n_steps)
    
    # Final validation check
    if best_result:
        try:
            f_vals = np.array(best_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                if l1 > 1e-15 and linf > 1e-15:
                    final_c2 = l2_sq / (l1 * linf)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
        except:
            pass
    
    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")