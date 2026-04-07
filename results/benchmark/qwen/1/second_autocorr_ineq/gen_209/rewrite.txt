# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
from numba import jit, njit
import random
import time
import warnings
from scipy.stats import qmc
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit

# Global constants
MAX_TIME_SECONDS = 85
DEFAULT_N_STEPS = 1000
MAX_EVALUATIONS = 10000

@njit
def compute_autoconvolution_manual(f_vals):
    """Manual computation of autoconvolution for better performance with Numba"""
    n = len(f_vals)
    if n == 0:
        return np.array([])

    # Allocate result array for autoconvolution
    g = np.zeros(2 * n - 1)

    # Manual convolution computation
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the three norms of the autoconvolution g = f*f
    Returns ||g||₂², ||g||₁, ||g||∞
    """
    # Ensure input is numpy array
    f = np.array(f_vals, dtype=np.float64)
    f = np.maximum(f, 0)  # Clip negative values to 0

    if len(f) == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution manually for better control and performance
    g_full = compute_autoconvolution_manual(f)

    # Trim to match proper interval [-1/4, 1/4]
    half_len = len(f)
    g_center = len(g_full) // 2
    g_trimmed = g_full[g_center - half_len : g_center + half_len]

    # Compute norms using trapezoidal integration for ||g||₂²
    if len(g_trimmed) < 2:
        norm_l2_sq = 0.0
    else:
        # Trapezoidal integration formula for piecewise linear segments
        # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
        step_width = 0.5 / len(f)
        g_abs = np.abs(g_trimmed)
        widths = np.full(len(g_abs)-1, step_width)
        y1 = g_abs[:-1]
        y2 = g_abs[1:]
        norm_l2_sq = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)

    # ||g||_1 = sum of absolute values normalized
    norm_l1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-15

    # ||g||_∞ = max absolute value
    norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-15

    return norm_l2_sq, norm_l1, norm_inf

@njit
def compute_c2_score_numba(f_vals):
    """
    Compute the C2 score: ||g||₂² / (||g||₁ · ||g||∞)
    Using numba-optimized version for performance
    """
    norm_l2_sq, norm_l1, norm_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_l1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_l2_sq / (norm_l1 * norm_inf)

@jax_jit
def compute_autoconvolution_jax(f_vals):
    """JAX version of autoconvolution computation for gradient support"""
    # Convert to JAX array and ensure non-negativity
    f = jnp.array(f_vals)
    f = jnp.maximum(f, 0)

    if len(f_vals) == 0:
        return jnp.array([])

    # Using JAX's convolution operation
    g_full = jnp.convolve(f, f, mode='full')

    # Trim to center portion
    half_len = len(f_vals)
    center_start = len(g_full) // 2 - half_len + 1
    center_end = len(g_full) // 2 + half_len - 1
    g_trimmed = g_full[center_start:center_end]

    return g_trimmed

@jax_jit
def compute_c2_jax(f_vals):
    """JAX version of C2 computation for automatic differentiation support"""
    # Ensure non-negative values
    f = jnp.clip(jnp.array(f_vals), 0, None)

    # Compute autoconvolution
    g = compute_autoconvolution_jax(f)

    # Compute norms using JAX operations
    g_abs = jnp.abs(g)

    # L2 squared norm using summation (simplified for JAX compatibility)
    norm_l2_sq = jnp.sum(g_abs * g_abs)

    # L1 norm
    norm_l1 = jnp.sum(g_abs) / (len(g) + 1)

    # L-infinity norm
    norm_inf = jnp.max(g_abs)

    # Avoid division by zero
    safe_l1 = jnp.where(norm_l1 <= 1e-15, 1e-15, norm_l1)
    safe_inf = jnp.where(norm_inf <= 1e-15, 1e-15, norm_inf)

    return norm_l2_sq / (safe_l1 * safe_inf)

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 score using JAX automatic differentiation with fallback"""
    try:
        # Ensure inputs are properly clipped and converted to JAX arrays
        f_array = jnp.array(np.clip(f_vals, 0, None))

        # Compute gradient using JAX automatic differentiation
        grad_func = grad(compute_c2_jax)
        gradients = grad_func(f_array)

        # Convert back to numpy array
        return np.array(gradients)
    except Exception as e:
        # If JAX gradient computation fails, fall back to finite differences
        epsilon = 1e-6
        gradients = np.zeros_like(f_vals)
        base_c2 = compute_c2_jax(f_array)
        
        for i in range(len(f_vals)):
            # Create perturbed arrays
            f_vals_plus = f_array.at[i].set(f_array[i] + epsilon)
            f_vals_minus = f_array.at[i].set(f_array[i] - epsilon)

            # Compute C2 values
            c2_plus = compute_c2_jax(f_vals_plus)
            c2_minus = compute_c2_jax(f_vals_minus)

            # Finite difference approximation
            gradients[i] = (c2_plus - c2_minus) / (2 * epsilon)
        
        return gradients

def generate_diverse_initialization(n):
    """Generate 10 diverse mathematical patterns for better exploration"""
    patterns = []
    
    # Pattern 1: Multi-peak Gaussian with varying scales (enhanced version)
    x = np.linspace(-1, 1, n)
    pattern1 = np.zeros(n)
    for i in range(5):  # Five peaks instead of three
        center = -0.8 + i * 0.4
        width = 0.1 + random.random() * 0.2
        height = 0.5 + random.random() * 0.8
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    patterns.append(pattern1.tolist())
    
    # Pattern 2: Alternating pattern with high/low values (enhanced)
    pattern2 = []
    for i in range(n):
        # More varied pattern: 10 different states instead of 3
        state = i % 10
        base_val = 1.8 if state < 2 else 0.2 if state < 4 else 1.0 if state < 6 else 0.4 if state < 8 else 1.2
        pattern2.append(base_val + random.random() * 0.3)
    patterns.append(pattern2)

    # Pattern 3: Sinusoidal pattern with modulation (enhanced)
    pattern3 = []
    for i in range(n):
        x_pos = i / (n - 1) if n > 1 else 0.5
        base = 1.0 + 0.4 * np.sin(10 * np.pi * x_pos)
        mod = 0.3 * np.cos(15 * np.pi * x_pos) + 0.2 * np.sin(25 * np.pi * x_pos)
        pattern3.append(max(0.0, base + mod))
    patterns.append(pattern3)

    # Pattern 4: Single dominant peak with decay (enhanced)
    pattern4 = []
    center = n // 2
    for i in range(n):
        distance = abs(i - center) / (n // 2)
        # More complex decay function
        val = max(0.0, 1.5 * np.exp(-3 * distance**2) + 0.5 * np.exp(-10 * distance**4))
        pattern4.append(val + 0.1 * random.random())
    patterns.append(pattern4)

    # Pattern 5: Random with heavy-tailed distribution (enhanced)
    pattern5 = []
    for i in range(n):
        # Heavy-tailed distribution with more extreme values
        r = random.random()
        if r < 0.6:
            pattern5.append(0.1 + 0.3 * random.random())
        elif r < 0.9:
            pattern5.append(0.8 + 1.2 * random.random())
        else:
            pattern5.append(2.0 + 3.0 * random.random())
    patterns.append(pattern5)
    
    # Pattern 6: Fractal-like self-similar structure
    pattern6 = []
    for i in range(n):
        x_pos = i / (n - 1) if n > 1 else 0.5
        # Nested sine waves creating fractal-like behavior
        val = 0.5 + 0.3 * np.sin(2 * np.pi * x_pos) + \
              0.2 * np.sin(6 * np.pi * x_pos) + \
              0.1 * np.sin(18 * np.pi * x_pos)
        pattern6.append(max(0.0, val))
    patterns.append(pattern6)

    # Pattern 7: Multi-scale sinc pattern
    pattern7 = []
    x = np.linspace(-1, 1, n)
    # Combines several sinc functions with different scales
    sinc1 = np.sinc(x * 5) * 0.6 + 0.4
    sinc2 = np.sinc(x * 3) * 0.4 + 0.3
    sinc3 = np.sinc(x * 7) * 0.5 + 0.2
    pattern7 = np.clip((sinc1 + sinc2 + sinc3) / 3.0, 0, None).tolist()
    patterns.append(pattern7)

    # Pattern 8: Optimized symmetric waveform
    pattern8 = []
    x = np.linspace(-1, 1, n)
    # Create a bell-shaped symmetric pattern
    pattern8 = np.clip(0.8 * np.exp(-2 * (x**2)) + 0.2, 0, None).tolist()
    patterns.append(pattern8)

    # Pattern 9: Wavelet-inspired structure
    pattern9 = []
    x = np.linspace(-1, 1, n)
    # Generate a pattern with wavelet-like characteristics
    wavelet_base = np.exp(-4 * x**2) * np.cos(8 * np.pi * x)
    pattern9 = np.clip(wavelet_base + 0.3, 0, None).tolist()
    patterns.append(pattern9)

    # Pattern 10: Multi-peak with specific spacing
    pattern10 = []
    for i in range(n):
        x_pos = i / (n - 1) if n > 1 else 0.5
        # Create peaks at specific locations for enhanced convolution properties
        peak_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        val = 0.0
        for pos in peak_positions:
            val += 0.8 * np.exp(-((x_pos - pos)**2) / 0.01) + 0.2 * np.exp(-((x_pos - pos)**2) / 0.05)
        pattern10.append(max(0.0, val))
    patterns.append(pattern10)

    # Select the best performing pattern among these
    best_pattern = patterns[0]
    best_score = -1.0

    for p in patterns:
        try:
            score = compute_c2_score_numba(p)
            if score > best_score:
                best_score = score
                best_pattern = p
        except:
            continue

    return best_pattern

def adaptive_gradient_descent(initial_params, max_iterations=500, use_momentum=True):
    """Enhanced adaptive gradient descent with momentum and adaptive learning rates"""
    current_params = np.array(initial_params, dtype=float)
    current_c2 = compute_c2_score_numba(current_params)
    
    # Initialize momentum terms
    velocity = np.zeros_like(current_params)
    momentum_factor = 0.9
    initial_learning_rate = 0.1
    
    # Adaptive learning rate
    learning_rate = initial_learning_rate
    patience = 0
    best_c2 = current_c2
    best_params = current_params.copy()
    
    improvement_history = []
    
    for iteration in range(max_iterations):
        # Compute gradient using JAX for better accuracy
        try:
            grad = compute_gradient_jax(current_params)
        except:
            # Fall back to numerical gradient if needed
            grad = np.zeros_like(current_params)
            epsilon = 1e-5
            base_c2 = compute_c2_score_numba(current_params)
            for i in range(len(current_params)):
                f_vals_plus = current_params.copy()
                f_vals_plus[i] = max(0, current_params[i] + epsilon)
                c2_plus = compute_c2_score_numba(f_vals_plus)
                grad[i] = (c2_plus - base_c2) / epsilon

        # Apply momentum if requested
        if use_momentum:
            velocity = momentum_factor * velocity - learning_rate * grad
            new_params = current_params + velocity
        else:
            new_params = current_params - learning_rate * grad
        
        # Apply non-negativity constraints
        new_params = np.maximum(new_params, 0)

        # Evaluate new solution
        new_c2 = compute_c2_score_numba(new_params)

        improvement = new_c2 - current_c2
        improvement_history.append(improvement)
        
        # Early termination if no significant improvement
        if len(improvement_history) >= 10:
            recent_improvements = improvement_history[-10:]
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement < 1e-8:
                break

        if new_c2 > current_c2:
            current_params = new_params
            current_c2 = new_c2
            patience = 0
            
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_params = current_params.copy()
        else:
            patience += 1
            if patience > 10:
                learning_rate *= 0.5
                patience = 0
                if learning_rate < 1e-6:
                    break

    return best_params, best_c2

def improved_multi_stage_optimization(initial_params):
    """Improved multi-stage optimization approach with enhanced refinement"""
    current_solution = np.array(initial_params)
    best_c2 = compute_c2_score(current_solution)
    best_solution = current_solution.copy()

    # Pre-optimization stage 1: Coarse grid search
    try:
        # Generate a coarse grid around the initial solution
        grid_size = min(20, len(current_solution))
        coarse_params = []
        for i in range(grid_size):
            # Perturb each parameter slightly
            perturbed = current_solution.copy()
            idx = i % len(perturbed)
            perturbed[idx] = max(0, perturbed[idx] + (random.random() - 0.5) * 0.5)
            coarse_params.append(perturbed)
        
        # Evaluate all candidates
        for candidate in coarse_params:
            c2 = compute_c2_score(candidate)
            if c2 > best_c2:
                best_c2 = c2
                best_solution = candidate.copy()
    except:
        pass

    # Stage 1: Coarse optimization with larger steps
    stage1_params = current_solution.copy()
    stage1_params, stage1_c2 = adaptive_gradient_descent(stage1_params, max_iterations=100, use_momentum=False)

    if stage1_c2 > best_c2:
        best_c2 = stage1_c2
        best_solution = stage1_params.copy()

    # Stage 2: Fine-grained optimization with momentum
    stage2_params, stage2_c2 = adaptive_gradient_descent(stage1_params, max_iterations=200, use_momentum=True)

    if stage2_c2 > best_c2:
        best_c2 = stage2_c2
        best_solution = stage2_params.copy()

    # Stage 3: Local refinement with L-BFGS
    try:
        def objective(x):
            return -compute_c2_score(x)

        bounds = [(0, None) for _ in range(len(best_solution))]
        result = minimize(
            objective,
            best_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )

        if result.success:
            refined_solution = np.maximum(result.x, 0)
            refined_c2 = compute_c2_score(refined_solution)

            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_solution
    except:
        pass

    # Post-processing refinement: stochastic perturbation
    try:
        # Apply small random perturbations to escape local minima
        perturbed_solution = best_solution.copy()
        for i in range(len(perturbed_solution)):
            if random.random() < 0.3:  # 30% chance to perturb
                perturbation = (random.random() - 0.5) * 0.1 * best_solution[i]
                perturbed_solution[i] = max(0, perturbed_solution[i] + perturbation)
        
        perturbed_c2 = compute_c2_score(perturbed_solution)
        if perturbed_c2 > best_c2:
            best_c2 = perturbed_c2
            best_solution = perturbed_solution
    except:
        pass

    return best_solution, best_c2

def adaptive_optimization_strategy():
    """Main adaptive optimization function with enhanced strategies"""
    best_c2 = -np.inf
    best_params = None

    # Strategy 1: Multiple random starts with different initialization sizes
    start_configs = [
        (300, 42),
        (500, 123),
        (700, 234),
        (900, 345),
        (1100, 456),
        (1300, 567)
    ]

    # Add some random configurations
    for _ in range(5):  # Increase from 3 to 5 for better exploration
        n = random.randint(400, 1200)
        seed = random.randint(1000, 9999)
        start_configs.append((n, seed))

    for n, seed in start_configs:
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break

        try:
            random.seed(seed)
            np.random.seed(seed)

            # Generate initial solution with enhanced diversity
            initial_params = generate_diverse_initialization(n)

            # Optimize using improved multi-stage approach
            optimized_params, optimized_c2 = improved_multi_stage_optimization(initial_params)

            if optimized_c2 > best_c2:
                best_c2 = optimized_c2
                best_params = optimized_params.copy()

        except Exception as e:
            continue

    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    global start_time
    start_time = time.time()

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    try:
        # Run adaptive optimization
        f_values, best_c2 = adaptive_optimization_strategy()

        # Fallback in case of failure
        if f_values is None or len(f_values) == 0:
            # Use simple initialization
            n = 500
            f_values = generate_diverse_initialization(n)

        # Ensure non-negative values
        f_values = np.maximum(f_values, 0).tolist()

        # Ensure reasonable size
        if len(f_values) < 50:
            f_values = f_values + [0.5] * (50 - len(f_values))
        elif len(f_values) > 10000:
            f_values = f_values[:10000]

        return f_values

    except Exception as e:
        # Final fallback
        return [0.5] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")