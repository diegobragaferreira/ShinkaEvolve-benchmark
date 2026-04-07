# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time
from numba import njit
import random
import copy

# Global constants
MAX_TIME_SECONDS = 85
INITIAL_SEARCH_SPACES = [100, 300, 500, 700, 1000]
POP_SIZES = [15, 20, 25]
MAX_ITERATIONS = [30, 50, 70]

@njit
def compute_autoconvolution_numba(f):
    """Compute autoconvolution g = f * f using numba JIT"""
    n = len(f)
    if n == 0:
        return np.array([])
        
    # Autoconvolution using discrete convolution
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]

    # Trim to center portion to reflect domain [-1/4, 1/4] properly
    # The convolution spans [-1/2, 1/2], so we take center portion of length n
    offset = (2*n - 1 - n) // 2
    g_trimmed = g[offset:offset+n]
    return g_trimmed

@njit
def compute_c2_numba(f):
    """Compute C2 value for given step function f using numba JIT"""
    if len(f) < 2:
        return 0.0

    # Compute autoconvolution
    g = compute_autoconvolution_numba(f)

    if len(g) == 0:
        return 0.0

    # Compute norms using trapezoidal integration for L2^2
    # For piecewise linear integration on equally spaced points:
    # Each segment of length h contributes h/3*(y1^2 + y1*y2 + y2^2)
    norm_l2_sq = 0.0
    norm_l1 = 0.0
    norm_inf = 0.0

    # Compute L1 norm
    for i in range(len(g)):
        abs_g = abs(g[i])
        norm_l1 += abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g

    # Compute L2^2 norm with trapezoidal integration
    if len(g) > 1:
        # First and last points get weight 1, middle points get weight 2
        # But for the specific trapezoidal weighted quadratic integration:
        # Each segment contributes (h/3)(y1^2 + y1*y2 + y2^2)
        step_width = 0.5 / len(f)  # Width of each step in [-1/4, 1/4]
        g_abs = np.abs(g)
        widths = np.full(len(g_abs)-1, step_width)
        y1 = g_abs[:-1]
        y2 = g_abs[1:]
        norm_l2_sq = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)
    else:
        norm_l2_sq = g[0] * g[0]

    # Avoid division by zero
    if norm_l1 < 1e-15 or norm_inf < 1e-15:
        return 0.0

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

def generate_pattern_initialization(n):
    """Generate sophisticated initial patterns for better optimization starting points"""
    # Pattern 1: Multi-Gaussian with varying centers and widths
    x = np.linspace(-1, 1, n)
    pattern1 = np.zeros(n)
    for _ in range(3):  # Three peaks
        center = np.random.uniform(-0.5, 0.5)
        width = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.8, 1.5)
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    
    # Pattern 2: Alternating high/low regions with smooth transitions
    pattern2 = np.zeros(n)
    segment_size = max(1, n // 8)
    for i in range(0, n, segment_size):
        end_idx = min(i + segment_size, n)
        if (i // segment_size) % 2 == 0:
            pattern2[i:end_idx] = 1.0 + np.random.random(end_idx - i) * 0.5
        else:
            pattern2[i:end_idx] = 0.2 + np.random.random(end_idx - i) * 0.3
    
    # Pattern 3: Sinusoidal modulation with random frequency
    pattern3 = 0.8 + 0.3 * np.sin(2 * np.pi * np.linspace(0, 5, n) * np.random.uniform(0.5, 2.0))
    
    # Pattern 4: Heavy-tailed distribution
    pattern4 = np.zeros(n)
    for i in range(n):
        if np.random.random() < 0.7:
            pattern4[i] = 0.2 + 0.3 * np.random.random()
        else:
            pattern4[i] = 1.0 + 2.0 * np.random.random()
    
    # Pattern 5: Single dominant peak with exponential decay
    center = n // 2
    pattern5 = np.zeros(n)
    for i in range(n):
        distance = abs(i - center) / (n // 2)
        pattern5[i] = max(0.0, 1.0 * np.exp(-2 * distance**2))
        # Add some random noise
        pattern5[i] += 0.1 * np.random.random()
    
    # Normalize all patterns
    patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]
    normalized_patterns = []
    
    for p in patterns:
        p = np.maximum(p, 0)  # Ensure non-negative
        if np.sum(p) > 0:
            p = p / np.sum(p) * 5  # Scale to reasonable magnitude
        normalized_patterns.append(p.tolist())
    
    # Select the best pattern based on initial C2 computation
    best_pattern = normalized_patterns[0]
    best_score = -1.0
    
    for p in normalized_patterns:
        try:
            score = compute_c2_numba(np.array(p))
            if score > best_score:
                best_score = score
                best_pattern = p
        except:
            continue
            
    return best_pattern

def adaptive_optimization_search():
    """Main adaptive optimization routine with multi-scale approach"""
    best_c2 = -np.inf
    best_f = None
    start_time = time.time()
    
    # Multi-start with different configurations
    configurations = []
    for n in INITIAL_SEARCH_SPACES:
        for pop_size in POP_SIZES:
            for max_iter in MAX_ITERATIONS:
                configurations.append((n, pop_size, max_iter))
    
    # Shuffle configurations for varied exploration
    random.shuffle(configurations)
    
    for i, (n, pop_size, max_iter) in enumerate(configurations):
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break
            
        try:
            # Generate sophisticated initial pattern
            initial_f = generate_pattern_initialization(n)
            
            # Differential evolution optimization
            bounds = [(0, 10) for _ in range(n)]
            
            def objective(x):
                f_vals = np.maximum(x, 0)
                try:
                    c2 = compute_c2_numba(f_vals)
                    return -c2  # Negative because we maximize
                except:
                    return 1e10  # Large penalty for invalid solutions
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                seed=i,
                strategy='best1bin',
                disp=False
            )
            
            if result.success:
                final_solution = np.maximum(result.x, 0)
                final_c2 = -objective(final_solution)
                
                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_f = final_solution.tolist()
                    
        except Exception as e:
            continue
    
    # If we found a solution, apply further refinement
    if best_f is not None and best_c2 > 0:
        # Local refinement with L-BFGS-B
        try:
            bounds = [(0, 10) for _ in range(len(best_f))]
            
            def local_objective(x):
                x = np.maximum(x, 0)
                try:
                    c2 = compute_c2_numba(x)
                    return -c2
                except:
                    return 1e10
            
            refined_result = minimize(
                local_objective,
                best_f,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30}
            )
            
            if refined_result.success:
                final_refined = np.maximum(refined_result.x, 0)
                refined_c2 = -local_objective(final_refined)
                
                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_f = final_refined.tolist()
                    
        except Exception as e:
            pass
            
        # Additional simulated annealing style refinement
        try:
            current_sol = copy.deepcopy(best_f)
            current_c2 = best_c2
            
            # Simulated annealing-like cooling schedule
            temp = 1.0
            max_steps = 50
            for step in range(max_steps):
                if temp < 1e-6:
                    break
                    
                # Create neighbor solution
                trial = current_sol.copy()
                idx = random.randint(0, len(trial) - 1)
                # Small perturbation
                perturbation = random.uniform(0.7, 1.3)
                trial[idx] = max(0, trial[idx] * perturbation)
                
                # Normalize
                trial_sum = sum(trial)
                if trial_sum > 0:
                    trial = [t/trial_sum * len(trial) for t in trial]
                
                trial_c2 = compute_c2_numba(np.array(trial))
                
                # Accept with probability based on temperature
                if trial_c2 > current_c2 or random.random() < np.exp((trial_c2 - current_c2) / temp):
                    current_sol = trial
                    current_c2 = trial_c2
                    
                temp *= 0.95  # Cooling
            
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f = current_sol
                
        except Exception as e:
            pass
    
    return best_f if best_f is not None else generate_pattern_initialization(500)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        f_values = adaptive_optimization_search()
        # Final safety check
        if f_values is None or len(f_values) == 0:
            f_values = [1.0] * 200
        else:
            # Ensure non-negative
            f_values = [max(0.0, x) for x in f_values]
    except Exception as e:
        # Final fallback
        f_values = [1.0] * 200
    
    # Ensure reasonable size
    if len(f_values) < 50:
        f_values = f_values + [1.0] * (50 - len(f_values))
    elif len(f_values) > 10000:
        f_values = f_values[:10000]
        
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")