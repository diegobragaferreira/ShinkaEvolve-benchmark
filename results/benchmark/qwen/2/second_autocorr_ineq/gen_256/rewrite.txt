# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List, Tuple
import time
import warnings
from scipy.ndimage import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Custom gradient computation for step function optimization
def compute_gradient(f_values: List[float], eps: float = 1e-6) -> List[float]:
    """
    Compute numerical gradient of step function to guide optimization
    """
    f = np.array(f_values, dtype=np.float64)
    grad = np.zeros_like(f)
    
    # Forward difference for interior points
    for i in range(1, len(f)-1):
        grad[i] = (f[i+1] - f[i-1]) / (2.0 * eps)
    
    # Forward and backward differences for boundaries
    grad[0] = (f[1] - f[0]) / eps
    grad[-1] = (f[-1] - f[-2]) / eps
    
    return grad.tolist()

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞) where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Convert to numpy array
    f = np.array(f_values)

    # Compute autoconvolution g = f * f
    g = signal.convolve(f, f, mode='full')

    # Extract central portion (valid autoconvolution)
    half_len = len(f) - 1
    g = g[half_len:]  # Take right half

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)

    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given function"""
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if norm_1 <= 1e-12 or norm_inf <= 1e-12:
        return 0.0

    c2 = norm_2_sq / (norm_1 * norm_inf)
    return c2

def create_gradient_enhanced_step_function(n_steps: int) -> List[float]:
    """
    Create a step function enhanced with gradient information for better optimization guidance
    """
    # Generate base structure with strategic peak placement
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Create a multi-scale structure with clear gradient characteristics
    # Use a combination of different frequency components
    base_shape = np.zeros(n_steps)
    
    # Add multiple frequency components to create rich gradient structure
    frequencies = [10, 20, 30, 40]
    amplitudes = [0.5, 0.3, 0.15, 0.05]
    
    for freq, amp in zip(frequencies, amplitudes):
        base_shape += amp * np.sin(freq * np.pi * x)
    
    # Add Gaussian-like peaks for better local structure
    n_peaks = min(8, max(3, n_steps // 100))
    peak_positions = np.linspace(-0.2, 0.2, n_peaks)
    peak_widths = np.random.uniform(0.01, 0.03, n_peaks)
    peak_heights = np.random.uniform(0.3, 1.0, n_peaks)
    
    for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
        base_shape += height * np.exp(-0.5 * ((x - pos) / width) ** 2)
    
    # Ensure non-negativity and normalize
    base_shape = np.maximum(base_shape, 0)
    
    if np.max(base_shape) > 0:
        base_shape = base_shape / np.max(base_shape) * 1.5
    
    # Apply sophisticated smoothing to maintain smooth gradients
    if n_steps > 50:
        base_shape = gaussian_filter1d(base_shape, sigma=1.5, mode='constant', cval=0.0)
    
    # Add some random noise to break symmetries but preserve structure
    noise_level = 0.05 * np.std(base_shape)
    noise = np.random.normal(0, noise_level, n_steps)
    final_shape = np.maximum(base_shape + noise, 0)
    
    return final_shape.tolist()

def compute_gradient_guided_norms(f_values: List[float]) -> Tuple[float, float, float, List[float]]:
    """
    Compute norms with gradient guidance for optimization
    Returns (||g||₂², ||g||₁, ||g||∞, gradient_of_f)
    """
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Compute gradient of the original function
    gradient = compute_gradient(f_values)
    
    return norm_2_sq, norm_1, norm_inf, gradient

def gradient_guided_local_search(initial_f: List[float], max_iter: int = 50) -> List[float]:
    """
    Perform gradient-guided local search focusing on promising regions
    """
    f_current = np.array(initial_f, dtype=np.float64)
    best_c2 = compute_c2(f_current.tolist())
    best_f = f_current.copy()
    
    # Identify promising regions based on gradient magnitude
    gradient = compute_gradient(f_current.tolist())
    gradient_magnitude = np.abs(gradient)
    
    # Focus on regions with high gradient change (potential for optimization)
    threshold = np.percentile(gradient_magnitude, 70)
    interesting_regions = np.where(gradient_magnitude > threshold)[0]
    
    # Perform adaptive updates based on gradient information
    for iteration in range(max_iter):
        f_new = f_current.copy()
        
        # For high-gradient regions, make more aggressive adjustments
        if len(interesting_regions) > 0:
            # Select random regions to update
            update_indices = random.sample(list(interesting_regions), 
                                         min(10, len(interesting_regions)))
            
            for idx in update_indices:
                # Use gradient direction to inform adjustments
                grad_val = gradient[idx]
                if abs(grad_val) > 1e-6:  # Significant gradient
                    # Adjust more aggressively in gradient direction
                    adjustment = np.random.normal(0, 0.1 * abs(grad_val))
                    f_new[idx] = max(0, f_new[idx] + adjustment)
                else:  # Low gradient, make moderate adjustments
                    adjustment = np.random.normal(0, 0.05)
                    f_new[idx] = max(0, f_new[idx] + adjustment)
        else:
            # Default random perturbation if no interesting regions
            for i in range(len(f_new)):
                adjustment = np.random.normal(0, 0.03)
                f_new[i] = max(0, f_new[i] + adjustment)
        
        # Evaluate new function
        new_c2 = compute_c2(f_new.tolist())
        
        # Accept improvement
        if new_c2 > best_c2:
            best_c2 = new_c2
            best_f = f_new.copy()
        
        f_current = f_new
    
    return best_f.tolist()

def adaptive_evolution_with_gradient_guidance(n_steps: int) -> List[float]:
    """
    Main evolutionary approach guided by gradient information
    """
    # Initialize population with gradient-enhanced individuals
    population_size = 30
    population = []
    
    for i in range(population_size):
        individual = create_gradient_enhanced_step_function(n_steps)
        population.append(individual)
    
    # Evaluate initial population
    fitness_scores = []
    for ind in population:
        score = compute_c2(ind)
        fitness_scores.append(score)
    
    # Selection based on fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite_count = max(5, population_size // 4)
    elite_indices = sorted_indices[:elite_count]
    
    # Create offspring through gradient-aware crossover and mutation
    offspring = []
    
    # Elitism: preserve best individuals
    for idx in elite_indices:
        offspring.append(population[idx])
    
    # Gradient-guided breeding
    while len(offspring) < population_size:
        # Select parents
        parent1_idx = random.choice(elite_indices)
        parent2_idx = random.choice(elite_indices)
        
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        
        # Create child with gradient-inspired crossover
        child = []
        for i in range(n_steps):
            # Blend parents with gradient-aware selection
            if random.random() < 0.5:
                child_val = parent1[i]
            else:
                child_val = parent2[i]
            
            # Apply gradient-informed mutation
            if random.random() < 0.1:  # Mutation rate
                # Use gradient information to determine mutation strength
                grad1 = compute_gradient(parent1)[i] if i < len(compute_gradient(parent1)) else 0
                grad2 = compute_gradient(parent2)[i] if i < len(compute_gradient(parent2)) else 0
                avg_grad = (abs(grad1) + abs(grad2)) / 2
                
                # Mutate more in high-gradient regions
                mutation_strength = 0.05 + 0.1 * avg_grad
                mutation = np.random.normal(0, mutation_strength)
                child_val = max(0, child_val + mutation)
            
            child.append(child_val)
        
        offspring.append(child)
    
    # Local refinement with gradient guidance
    refined_population = []
    for child in offspring:
        refined_child = gradient_guided_local_search(child, max_iter=30)
        refined_population.append(refined_child)
    
    # Evaluate refined population
    refined_scores = []
    for ind in refined_population:
        score = compute_c2(ind)
        refined_scores.append(score)
    
    # Return best individual
    best_idx = np.argmax(refined_scores)
    return refined_population[best_idx]

def construct_function() -> List[float]:
    """
    Main function to construct step-function with high C2 value using gradient-guided evolutionary approach.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    max_time_seconds = 85
    
    # Determine number of steps
    n_steps = min(10000, max(100, 1000 + int(np.random.randint(0, 300) * 5)))
    
    # Try multiple strategies to maximize chance of success
    best_c2 = 0.0
    best_function = []
    
    # Strategy 1: Gradient-guided evolutionary approach
    try:
        if time.time() - start_time < max_time_seconds - 5:
            func1 = adaptive_evolution_with_gradient_guidance(n_steps)
            c2_score = compute_c2(func1)
            if c2_score > best_c2:
                best_c2 = c2_score
                best_function = func1
    except Exception:
        pass
    
    # Strategy 2: Hybrid approach combining different initialization methods
    if time.time() - start_time < max_time_seconds - 5:
        try:
            # Create multiple diverse initializations
            diverse_functions = []
            
            # Different initialization strategies
            for i in range(5):
                if i == 0:
                    # Gaussian-based
                    func = create_gradient_enhanced_step_function(n_steps)
                elif i == 1:
                    # Polynomial-based approach
                    x = np.linspace(-0.25, 0.25, n_steps)
                    func = (0.5 * np.exp(-x**2 / 0.02) + 
                           0.3 * np.exp(-((x - 0.1)**2) / 0.01) + 
                           0.2 * np.exp(-((x + 0.1)**2) / 0.01)).tolist()
                else:
                    # Sinusoidal approach
                    x = np.linspace(-0.25, 0.25, n_steps)
                    func = (0.5 * np.sin(20 * np.pi * x) + 
                           0.3 * np.sin(40 * np.pi * x) + 
                           0.2 * np.sin(60 * np.pi * x) + 
                           0.5).tolist()
                
                diverse_functions.append(func)
            
            # Evaluate and select best
            candidate_scores = []
            for func in diverse_functions:
                score = compute_c2(func)
                candidate_scores.append(score)
            
            best_diverse_idx = np.argmax(candidate_scores)
            func2 = diverse_functions[best_diverse_idx]
            
            if candidate_scores[best_diverse_idx] > best_c2:
                best_c2 = candidate_scores[best_diverse_idx]
                best_function = func2
                
        except Exception:
            pass
    
    # Strategy 3: Final refinement if we have a function
    if best_function and time.time() - start_time < max_time_seconds - 3:
        try:
            # Apply gradient-guided local search
            refined_func = gradient_guided_local_search(best_function, max_iter=50)
            refined_c2 = compute_c2(refined_func)
            
            if refined_c2 > best_c2:
                best_function = refined_func
        except Exception:
            pass
    
    # Fallback if nothing worked
    if not best_function:
        # Create simple but robust function
        best_function = [1.0] * n_steps
        
    # Ensure correct length
    if len(best_function) != n_steps:
        if len(best_function) < n_steps:
            best_function.extend([0.0] * (n_steps - len(best_function)))
        else:
            best_function = best_function[:n_steps]
    
    return best_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")