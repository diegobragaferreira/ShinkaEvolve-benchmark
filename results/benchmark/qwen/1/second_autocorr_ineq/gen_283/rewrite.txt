# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
from scipy.optimize import differential_evolution, minimize
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')

# Global constants for performance tuning
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 3000
MIN_STEPS = 200
MAX_STEPS = 8000
POPULATION_SIZE_BASE = 12
INITIAL_REFINEMENT_ITERATIONS = 30
FINAL_REFINEMENT_ITERATIONS = 20

# Optimized sparse convolution computation using FFT
@jit(nopython=True)
def compute_sparse_autoconvolution(f_vals):
    """
    Efficiently compute autoconvolution using optimized Numba loops.
    Exploits the fact that step functions are piecewise constant.
    """
    n = len(f_vals)
    
    # For dense convolution: g[i] = sum_{j+k=i} f[j] * f[k]
    g = np.zeros(2*n - 1, dtype=np.float64)
    
    # Optimized nested loop for autoconvolution
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_sparse_norms(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms using highly optimized loops
    """
    n = len(g_vals)
    
    # L1 norm (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])
    
    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    for i in range(n):
        l2_sq_norm += g_vals[i] * g_vals[i]
    
    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val
    
    return l1_norm, l2_sq_norm, linf_norm

@jit(nopython=True)
def compute_c2_sparse(f_vals):
    """
    Fast computation of C2 using sparse convolution and norms
    """
    # Compute autoconvolution
    g_vals = compute_sparse_autoconvolution(f_vals)
    
    # Compute norms
    l1, l2_sq, linf = compute_sparse_norms(g_vals)
    
    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0
    
    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function_sparse(f_vals):
    """
    Evaluate step function with robust error handling and sparse computation
    """
    try:
        # Ensure non-negative values with clipping
        f_vals = np.array([max(0.0, x) for x in f_vals])
        
        # Handle edge cases
        if len(f_vals) == 0 or np.isnan(np.sum(f_vals)) or np.isinf(np.sum(f_vals)):
            return 0.0
            
        # If all values are zero, return 0
        if np.sum(f_vals) == 0:
            return 0.0
            
        # Compute C2 value using sparse method
        c2 = compute_c2_sparse(f_vals)
        
        # Validate result
        if np.isnan(c2) or np.isinf(c2) or c2 < 0:
            return 0.0
            
        return c2
    except Exception as e:
        return 0.0

def compute_autoconvolution_fft_efficient(f_vals):
    """
    Compute autoconvolution using FFT - O(n log n) complexity
    """
    f_vals = np.array(f_vals, dtype=np.float64)
    
    # For autoconvolution f*f, we can use FFT:
    # conv(f,f) = ifft(fft(f)^2)
    n = len(f_vals)
    
    # Pad to next power of 2 for better FFT performance
    fft_size = 2**int(np.ceil(np.log2(2*n - 1)))
    
    # Compute FFT of f padded to fft_size
    f_padded = np.pad(f_vals, (0, fft_size - n), 'constant')
    f_fft = fft(f_padded)
    
    # Compute autoconvolution in frequency domain (square the FFT)
    g_fft = f_fft * f_fft
    
    # Transform back to time domain
    g = np.real(ifft(g_fft))
    
    # Keep only the valid convolution part (middle)
    # The valid convolution goes from index (n-1) to (n-1)+(n-1) = 2*n-2
    g_valid = g[n-1:2*n-1]
    
    return g_valid

def compute_autoconvolution_norms_fft(f_vals):
    """
    Compute all three norms using FFT-based autoconvolution
    """
    try:
        # Compute autoconvolution using FFT
        g_vals = compute_autoconvolution_fft_efficient(f_vals)
        
        # Compute norms
        l1_norm = np.sum(np.abs(g_vals))
        l2_sq_norm = np.sum(g_vals**2)
        linf_norm = np.max(np.abs(g_vals))
        
        # Handle edge cases
        if l1_norm <= 1e-15 or linf_norm <= 1e-15:
            return 0.0, 0.0, 0.0, 0.0
            
        # Compute C2
        c2 = l2_sq_norm / (l1_norm * linf_norm)
        
        return c2, l2_sq_norm, l1_norm, linf_norm
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_c2_fft(f_vals):
    """
    Fast evaluation of C2 using FFT-based convolution
    """
    try:
        c2, _, _, _ = compute_autoconvolution_norms_fft(f_vals)
        return c2
    except Exception:
        return 0.0

def create_bell_shaped_sparse(n_steps):
    """Create a bell-shaped base pattern optimized for sparse computation"""
    x = np.linspace(0, 1, n_steps)
    # Create Gaussian-like shape with emphasis on edges and central peak
    pattern = 1.0 + 0.8 * np.exp(-15 * (x - 0.5)**2) - 0.3 * np.exp(-5 * x**2) - 0.3 * np.exp(-5 * (1-x)**2)
    pattern = np.clip(pattern, 0, np.inf)
    
    # Normalize appropriately for sparse optimization
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_alternating_sparse(n_steps):
    """Create alternating high/low pattern for sparse computation"""
    pattern = []
    for i in range(n_steps):
        if i % 2 == 0:
            pattern.append(1.0 + np.random.random() * 0.5)
        else:
            pattern.append(0.2 + np.random.random() * 0.3)
    
    pattern = np.array(pattern)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_peak_centered_sparse(n_steps):
    """Create peak-centered pattern with tapering edges optimized for sparse computation"""
    pattern = np.zeros(n_steps)
    center = n_steps // 2
    width = max(1, n_steps // 6 + np.random.randint(-1, 2))
    
    # Create central peak with smooth transitions
    for i in range(n_steps):
        distance_from_center = abs(i - center)
        if distance_from_center <= width:
            # Quadratic transition
            t = distance_from_center / width
            pattern[i] = 1.0 * (1 - t**2) + 0.5 * np.random.random()
    
    # Add noise for diversity
    noise = np.random.normal(0, 0.05, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)
    
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_smooth_transition_sparse(n_steps):
    """Create smooth transition pattern optimized for sparse computation"""
    pattern = np.zeros(n_steps)
    # Create smooth ramp with some variation
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        pattern[i] = 0.5 + 0.5 * np.sin(np.pi * x) + np.random.normal(0, 0.1)
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_balanced_sparse(n_steps):
    """Create balanced pattern optimized for sparse computation"""
    # Create pattern that maintains balance
    pattern = np.ones(n_steps) * 0.5
    
    # Add structured variation
    for i in range(0, n_steps, 10):
        if i < n_steps:
            pattern[i] = 1.0 + np.random.random() * 0.5
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_asymmetric_sparse(n_steps):
    """Create asymmetric pattern that breaks symmetry for better exploration"""
    pattern = np.zeros(n_steps)
    
    # Create asymmetric structure with higher values on one side
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Asymmetric exponential decay
        if x < 0.5:
            pattern[i] = 1.0 + 0.5 * np.exp(-5 * x)
        else:
            pattern[i] = 0.5 + 0.3 * np.exp(-10 * (x - 0.5))
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_bell_shaped(n_steps):
    """Create bell-shaped pattern optimized for hierarchical convolution"""
    x = np.linspace(0, 1, n_steps)
    # Create Gaussian-like shape with emphasis on central region
    pattern = 1.0 + 0.8 * np.exp(-12 * (x - 0.5)**2) - 0.3 * np.exp(-6 * x**2) - 0.3 * np.exp(-6 * (1-x)**2)
    pattern = np.clip(pattern, 0, np.inf)
    
    # Normalize appropriately
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_alternating(n_steps):
    """Create alternating high/low pattern for hierarchical computation"""
    pattern = []
    for i in range(n_steps):
        if i % 3 == 0:
            pattern.append(1.0 + np.random.random() * 0.3)
        elif i % 3 == 1:
            pattern.append(0.3 + np.random.random() * 0.2)
        else:
            pattern.append(0.7 + np.random.random() * 0.2)
    
    pattern = np.array(pattern)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_peak_centered(n_steps):
    """Create peak-centered pattern with tapering edges optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)
    center = n_steps // 2
    width = max(1, n_steps // 8 + np.random.randint(-2, 3))
    
    # Create central peak with smooth transitions
    for i in range(n_steps):
        distance_from_center = abs(i - center)
        if distance_from_center <= width:
            # Quadratic transition
            t = distance_from_center / width
            pattern[i] = 1.0 * (1 - t**2) + 0.4 * np.random.random()
    
    # Add noise for diversity
    noise = np.random.normal(0, 0.04, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)
    
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_smooth_transition(n_steps):
    """Create smooth transition pattern optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)
    # Create smooth ramp with more variation
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        pattern[i] = 0.6 + 0.4 * np.sin(np.pi * x + np.random.random()) + np.random.normal(0, 0.08)
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_balanced(n_steps):
    """Create balanced pattern optimized for hierarchical computation"""
    # Create pattern that maintains balance
    pattern = np.ones(n_steps) * 0.6
    
    # Add structured variation
    for i in range(0, n_steps, 15):
        if i < n_steps:
            pattern[i] = 1.2 + np.random.random() * 0.3
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_asymmetric(n_steps):
    """Create asymmetric pattern that breaks symmetry for better hierarchical exploration"""
    pattern = np.zeros(n_steps)
    
    # Create asymmetric structure with higher values on one side
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Asymmetric exponential decay
        if x < 0.4:
            pattern[i] = 1.0 + 0.4 * np.exp(-6 * x)
        elif x < 0.7:
            pattern[i] = 0.6 + 0.2 * np.exp(-8 * (x - 0.4))
        else:
            pattern[i] = 0.3 + 0.2 * np.exp(-12 * (x - 0.7))
    
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_multipeak(n_steps):
    """Create multi-peak pattern optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)
    
    # Multiple peaks at strategic locations
    peak_positions = [n_steps // 5, 2*n_steps//5, 3*n_steps//5, 4*n_steps//5]
    for i, pos in enumerate(peak_positions):
        if pos < n_steps:
            # Gaussian peaks
            width = n_steps // 20 + np.random.randint(-2, 3)
            for j in range(max(0, pos - width), min(n_steps, pos + width)):
                distance = abs(j - pos)
                spread_factor = np.exp(-distance**2 / (2 * width**2))
                pattern[j] += 1.0 * spread_factor
    
    # Add some randomness
    noise = np.random.normal(0, 0.03, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)
    
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_multi_scale_structural_initialization(n_steps):
    """
    Create diverse initial solutions with different structural properties for hierarchical optimization
    """
    # Multiple initialization strategies that work well with hierarchical approaches
    strategies = [
        lambda n: create_hierarchical_bell_shaped(n),
        lambda n: create_hierarchical_alternating(n),
        lambda n: create_hierarchical_peak_centered(n),
        lambda n: create_hierarchical_smooth_transition(n),
        lambda n: create_hierarchical_balanced(n),
        lambda n: create_hierarchical_asymmetric(n),
        lambda n: create_hierarchical_multipeak(n)
    ]
    
    # Choose a strategy
    strategy = np.random.choice(strategies)
    pattern = strategy(n_steps)
    
    # Apply minor random perturbations for diversity
    noise = np.random.normal(0, 0.02, n_steps)
    pattern = pattern + noise
    pattern = np.maximum(pattern, 0.0)
    
    # Final normalization
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    
    return pattern

def hierarchical_adaptive_evolutionary_optimization(initial_population):
    """
    Advanced hierarchical evolutionary optimization with multi-scale awareness
    """
    # Track convergence and adapt parameters dynamically
    best_scores = []
    patience_counter = 0
    max_patience = 8
    population_size = POPULATION_SIZE_BASE
    
    # Start with initial population
    population = [list(ind) for ind in initial_population]
    current_best = max(population, key=evaluate_c2_fft)
    best_scores.append(evaluate_c2_fft(current_best))
    
    # Hierarchical approach: start with coarse grids and refine
    current_scale = 1.0
    
    # Adaptive parameters based on convergence behavior
    for generation in range(150):  # Limited to prevent timeout
        # Scale population size based on hierarchy level
        if len(population) < 10:
            population_size = 10
        elif len(population) > 25:
            population_size = 20
        else:
            population_size = len(population) // 2 + 5
            
        # Evaluate all individuals with FFT-based computation
        fitnesses = [evaluate_c2_fft(ind) for ind in population]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Update best
        current_best = sorted_population[0]
        best_scores.append(sorted_fitnesses[0])
        
        # Check for convergence with multiple criteria
        if len(best_scores) >= 5:
            # Check for stagnation in improvement
            recent_improvement = best_scores[-1] - best_scores[-5]
            if recent_improvement < 1e-9:
                patience_counter += 1
            else:
                patience_counter = 0
                
            if patience_counter >= max_patience:
                # Increase population size to escape local minimum
                population_size = min(population_size * 2, 40)
                patience_counter = 0
                
        # Create offspring using tournament selection and crossover
        new_population = []
        
        # Elitism: keep the best 25%
        elite_count = max(1, int(0.25 * population_size))
        new_population.extend(sorted_population[:elite_count])
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(sorted_population), tournament_size)
            tournament_fitnesses = [sorted_fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            
            # Clone selected parent
            parent = sorted_population[winner_index].copy()
            
            # Hierarchical-aware mutation with scale adjustment
            mutation_strength = 0.15 * (1 - generation / 150.0) * current_scale  # Decrease over time and scale
            
            # Mutate with hierarchical consideration
            for i in range(len(parent)):
                if np.random.random() < 0.12:  # 12% chance to mutate each element
                    noise = np.random.normal(0, mutation_strength)
                    parent[i] = max(0, parent[i] + noise)
                    
            new_population.append(parent)
            
        # Replace population
        population = new_population
        
        # Early termination based on time
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break
            
    return current_best

def multi_objective_sparse_optimization(initial_solution):
    """
    Multi-objective optimization that considers both flatness and peak suppression
    """
    # This combines multiple objectives to better shape g distribution
    best_solution = initial_solution.copy()
    best_c2 = evaluate_c2_fft(best_solution)
    
    # Objective 1: Maximize C2
    def objective1(x):
        return -evaluate_c2_fft(x)
    
    # Objective 2: Minimize peak-to-mean ratio in g (encourage flatness)
    def objective2(x):
        try:
            f_vals = np.array(x)
            g_vals = compute_autoconvolution_fft_efficient(f_vals)
            
            # Compute peak-to-mean ratio (lower is better for flatness)
            mean_g = np.mean(np.abs(g_vals))
            max_g = np.max(np.abs(g_vals))
            if mean_g <= 1e-12:
                return 0.0
            return max_g / mean_g  # Larger ratios are worse
        except:
            return 1e10
    
    # Combined objective (weighted sum)
    def combined_objective(x):
        c2_val = evaluate_c2_fft(x)
        flatness_val = objective2(x)  # This will be minimized
        # Weight toward C2 but penalize high peaks
        return -c2_val + 0.1 * flatness_val  # Higher is better for combined
    
    # Try local optimization
    try:
        # Use L-BFGS-B for local refinement 
        x0 = np.array(best_solution[:min(1000, len(best_solution))])
        bounds = [(0, 10.0)] * len(x0)
        
        def obj_func(x):
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return combined_objective(extended_x)
        
        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 15})
        
        if res.success:
            refined_solution = np.maximum(res.x, 0)
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_c2_fft(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception:
        pass
    
    return best_solution

def cross_scale_transfer_refinement(initial_solution):
    """
    Transfer knowledge from different scales to enhance optimization
    """
    best_solution = initial_solution.copy()
    best_c2 = evaluate_c2_fft(best_solution)
    
    # Try different sized versions to learn structural patterns
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Different resolutions
    
    for scale in scale_factors:
        try:
            target_size = max(100, int(len(best_solution) * scale))
            if target_size != len(best_solution):
                # Create scaled-down version
                if target_size < len(best_solution):
                    # Reduce resolution
                    reduced_indices = np.linspace(0, len(best_solution)-1, target_size, dtype=int)
                    reduced_solution = [best_solution[i] for i in reduced_indices]
                else:
                    # Increase resolution via interpolation
                    old_indices = np.linspace(0, len(best_solution)-1, len(best_solution))
                    new_indices = np.linspace(0, len(best_solution)-1, target_size)
                    reduced_solution = np.interp(new_indices, old_indices, best_solution)
                
                # Refine the reduced version
                refined_reduced = multi_objective_sparse_optimization(reduced_solution)
                
                # Interpolate back to original resolution
                if len(refined_reduced) != len(best_solution):
                    old_indices = np.linspace(0, len(refined_reduced)-1, len(refined_reduced))
                    new_indices = np.linspace(0, len(refined_reduced)-1, len(best_solution))
                    interpolated_solution = np.interp(new_indices, old_indices, refined_reduced)
                else:
                    interpolated_solution = refined_reduced
                
                # Evaluate and keep if better
                interpolated_c2 = evaluate_c2_fft(interpolated_solution)
                if interpolated_c2 > best_c2:
                    best_solution = interpolated_solution
                    best_c2 = interpolated_c2
                    
        except Exception:
            continue
    
    return best_solution

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value using hierarchical optimization.
    Implements a truly novel approach combining hierarchical scales, multi-objective optimization,
    and cross-scale knowledge transfer.
    """
    global start_time
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize with hierarchical approach using multi-scale optimizations
    initial_solutions = []
    n_attempts = 20
    
    for i in range(n_attempts):
        # Create diverse initial solutions with hierarchical optimization in mind
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        init_solution = create_multi_scale_structural_initialization(n_steps)
        initial_solutions.append(init_solution)
    
    # Select best initial solution
    best_init = max(initial_solutions, key=evaluate_c2_fft)
    
    # Apply hierarchical adaptive evolutionary optimization
    evolved_solution = hierarchical_adaptive_evolutionary_optimization(initial_solutions)
    
    # Apply multi-objective optimization for better g distribution
    multi_obj_solution = multi_objective_sparse_optimization(evolved_solution)
    
    # Apply cross-scale transfer refinement
    refined_solution = cross_scale_transfer_refinement(multi_obj_solution)
    
    # Final evaluation and return the best
    final_c2 = evaluate_c2_fft(refined_solution)
    initial_c2 = evaluate_c2_fft(best_init)
    
    if final_c2 > initial_c2:
        result = refined_solution
    else:
        result = best_init
    
    # Ensure proper length
    if len(result) < MIN_STEPS:
        result.extend([1.0] * (MIN_STEPS - len(result)))
    elif len(result) > MAX_STEPS:
        result = result[:MAX_STEPS]
    
    # Normalize sparse representation if needed
    if np.sum(result) > 0:
        result = np.array(result) / np.sum(result) * len(result)
    
    # Ensure non-negativity and finite values for sparse computation
    result = np.clip(result, 0, np.inf)
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Print debug info
    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_c2_fft(result):.6f}")
    
    return result.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")