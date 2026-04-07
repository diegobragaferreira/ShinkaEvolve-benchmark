# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from numba import jit
import random
import time
from scipy.special import erf
import warnings
from deap import base, creator, tools, algorithms
import jax
import jax.numpy as jnp

# === Core Computation Module ===
@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values):
    """Optimized computation of autoconvolution norms using numba"""
    n = len(f_values)

    # Initialize autoconvolution array
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Keep only center portion
    half_len = n - 1
    g_center = g[half_len:-half_len]

    # Compute norms
    norm_2_squared = np.sum(g_center**2)
    norm_1 = np.sum(np.abs(g_center))
    norm_inf = np.max(np.abs(g_center))

    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation with optimized FFT-based convolution"""
    try:
        if not f_values:
            return 0.0, 0.0, 0.0, 0.0

        # Convert to numpy array for easier manipulation
        f = np.array(f_values, dtype=np.float64)

        # Ensure non-negative values
        f = np.maximum(f, 0.0)

        # Use FFT-based convolution for better performance
        # For autoconvolution f*f, we can use FFT: conv(f,f) = ifft(fft(f)^2)
        n = len(f)

        # Pad to next power of 2 for better FFT performance
        fft_size = 2**int(np.ceil(np.log2(2*n - 1)))

        # Compute FFT of f padded to fft_size
        f_padded = np.pad(f, (0, fft_size - n), 'constant')
        f_fft = np.fft.fft(f_padded)

        # Compute autoconvolution in frequency domain
        g_fft = f_fft * f_fft

        # Transform back to time domain
        g = np.real(np.fft.ifft(g_fft))

        # Keep only the valid convolution part (middle)
        # The valid convolution goes from index (n-1) to (n-1)+(n-1) = 2*n-2
        g_valid = g[n-1:2*n-1]

        # Compute norms
        norm_2_squared = np.sum(g_valid**2)
        norm_1 = np.sum(np.abs(g_valid))
        norm_inf = np.max(np.abs(g_valid))

        # Avoid division by zero
        if norm_1 == 0 or norm_inf == 0:
            return 0.0, 0.0, 0.0, 0.0

        # C2 = ||g||₂² / (||g||₁ · ||g||∞)
        c2 = norm_2_squared / (norm_1 * norm_inf)

        return c2, norm_2_squared, norm_1, norm_inf
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_c2(individual):
    """Evaluate fitness of individual (step function) for maximizing C2"""
    try:
        # Ensure non-negative values
        individual = np.maximum(individual, 0.0)

        # Compute C2 value
        c2, _, _, _ = compute_autoconvolution_norms(individual)

        # Return negative because we want to maximize
        return -c2
    except Exception:
        # Return very poor fitness if error occurs
        return 1e10

# JAX version for automatic differentiation
@jax.jit
def compute_autoconvolution_jax(f_values):
    """JAX version of autoconvolution for gradient computation"""
    f = jnp.array(f_values, dtype=jnp.float32)
    n = len(f)

    # Use JAX's convolution
    f_padded = jnp.pad(f, (0, 2*n - 1 - n), 'constant')
    f_fft = jnp.fft.fft(f_padded)
    g_fft = f_fft * f_fft
    g = jnp.real(jnp.fft.ifft(g_fft))

    # Keep valid convolution part
    g_valid = g[n-1:2*n-1]

    return g_valid

@jax.jit
def compute_c2_jax(f_values):
    """JAX version to compute C2 for gradient computation"""
    try:
        g_vals = compute_autoconvolution_jax(f_values)

        # Compute norms in JAX
        l1 = jnp.sum(jnp.abs(g_vals))
        l2_sq = jnp.sum(g_vals**2)
        linf = jnp.max(jnp.abs(g_vals))

        # Avoid division by zero
        l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
        linf_safe = jnp.where(linf <= 1e-15, 1e-15, linf)

        return l2_sq / (l1_safe * linf_safe)
    except Exception:
        return 0.0

# Gradient function using JAX with improved error handling
@jax.jit
def compute_c2_jax_safe(f_values):
    """Safe JAX version to compute C2 with better error handling"""
    try:
        g_vals = compute_autoconvolution_jax(f_values)

        # Compute norms in JAX
        l1 = jnp.sum(jnp.abs(g_vals))
        l2_sq = jnp.sum(g_vals**2)
        linf = jnp.max(jnp.abs(g_vals))

        # Avoid division by zero with safe denominators
        l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
        linf_safe = jnp.where(linf <= 1e-15, 1e-15, linf)

        return l2_sq / (l1_safe * linf_safe)
    except Exception:
        return 0.0

compute_c2_grad = jax.grad(compute_c2_jax_safe)

# === Initialization Module ===
def smooth_step_function(x, center, width, height):
    """Generate a smooth approximation to a step function using error functions"""
    # Convert to smooth sigmoid-like shape
    return height * (erf((x - center + width/2) / (width/3)) -
                     erf((x - center - width/2) / (width/3))) / 2

def multi_scale_initialization(n_steps):
    """Create multi-scale initialization for better exploration"""
    # Create coarse grid first
    coarse_points = np.linspace(-0.25, 0.25, min(21, n_steps//2 + 1))
    coarse_values = np.random.rand(len(coarse_points)) * 0.5 + 0.5

    # Interpolate to fine grid
    fine_grid = np.linspace(-0.25, 0.25, n_steps)

    # Use piecewise linear interpolation
    coarse_fine = np.interp(fine_grid, coarse_points, coarse_values)

    # Add some noise for diversity
    noise = np.random.normal(0, 0.02, n_steps)
    adjusted = coarse_fine + noise

    # Ensure non-negative
    adjusted = np.maximum(adjusted, 0.0)

    # Normalize to reasonable scale
    if np.sum(adjusted) > 0:
        adjusted = adjusted * n_steps / np.sum(adjusted)

    return adjusted.tolist()

def kernel_smoothed_initialization(n_steps):
    """Initialize with kernel-smoothed random pattern"""
    # Generate base random pattern
    base_pattern = np.random.rand(n_steps) * 0.8 + 0.2

    # Apply gaussian smoothing
    kernel_size = max(1, n_steps // 50)
    kernel = np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * (kernel_size/3)**2))
    kernel = kernel / np.sum(kernel)

    smoothed = np.convolve(base_pattern, kernel, mode='same')

    # Apply soft thresholding to encourage sparsity
    threshold = np.mean(smoothed) * 0.3
    smoothed = np.maximum(smoothed - threshold, 0.0)

    # Normalize
    if np.sum(smoothed) > 0:
        smoothed = smoothed * n_steps / np.sum(smoothed)

    return smoothed.tolist()

# === Evolutionary Optimization Module ===
def setup_evolutionary_algorithm(n_steps, pop_size=50, cxpb=0.5, mutpb=0.2, ngen=40):
    """Setup evolutionary algorithm for optimizing step functions"""
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene representation (normalized values between 0 and 1)
    def create_individual():
        # Create individual with random values and normalize
        individual = [random.uniform(0, 1) for _ in range(n_steps)]
        # Normalize to sum to 1 to maintain reasonable magnitude
        total = sum(individual)
        if total > 0:
            individual = [x/total for x in individual]
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register("evaluate", lambda ind: evaluate_c2(list(ind)))

    # Register genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def run_evolutionary_optimization(n_steps, max_time_seconds=60):
    """Run evolutionary optimization to find high C2 solutions"""
    # Set up evolutionary algorithm
    toolbox = setup_evolutionary_algorithm(n_steps, pop_size=50, cxpb=0.5, mutpb=0.2, ngen=40)

    # Create initial population
    population = toolbox.population(n=50)

    # Run evolutionary algorithm
    start_time = time.time()
    best_individual = None
    best_fitness = float('-inf')

    try:
        # Run until time limit or convergence
        for gen in range(40):
            if time.time() - start_time > max_time_seconds:
                break

            # Evaluate fitness for entire population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)

            # Select the next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace population
            population[:] = offspring

            # Track best individual
            for ind in population:
                if ind.fitness.values[0] > best_fitness:
                    best_fitness = ind.fitness.values[0]
                    best_individual = list(ind)

    except Exception as e:
        print(f"Evolutionary optimization error: {e}")

    return best_individual if best_individual is not None else None

# === Enhanced Optimization Module ===
def multi_objective_optimization(initial_solution, max_iter=1000):
    """Enhanced optimization using multi-objective approach with multiple refinement phases"""
    x = np.array(initial_solution, dtype=float)
    n = len(x)

    # Multiple refinement phases
    phases = [
        {'lr': 0.1, 'momentum': 0.9, 'decay_rate': 0.99, 'max_iter': 300},
        {'lr': 0.05, 'momentum': 0.95, 'decay_rate': 0.995, 'max_iter': 200},
        {'lr': 0.01, 'momentum': 0.98, 'decay_rate': 0.999, 'max_iter': 200}
    ]

    # Track best solution
    best_x = x.copy()
    best_c2 = evaluate_c2(x)

    # Precompute gradient function for better performance
    @jax.jit
    def get_grad(x_array):
        return np.array(compute_c2_grad(x_array))

    for phase_config in phases:
        lr = phase_config['lr']
        momentum = phase_config['momentum']
        decay_rate = phase_config['decay_rate']
        max_iter_phase = phase_config['max_iter']

        velocity = np.zeros_like(x)
        patience_counter = 0
        current_lr = lr

        for iteration in range(max_iter_phase):
            # Compute gradient using JAX automatic differentiation instead of finite differences
            try:
                grad = get_grad(x)
            except Exception:
                # Fallback to finite differences if JAX fails
                eps = 1e-6
                grad = np.zeros_like(x)
                for i in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    grad_i = (evaluate_c2(x_plus) - evaluate_c2(x_minus)) / (2 * eps)
                    grad[i] = grad_i

            # Update with momentum
            velocity = momentum * velocity - current_lr * grad
            x = x + velocity

            # Project onto feasible region
            x = np.maximum(x, 0.0)

            # Check improvement
            current_score = evaluate_c2(x)

            if current_score < best_c2:
                best_c2 = current_score
                best_x = x.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Adaptive learning rate
            if patience_counter > 10:
                current_lr *= decay_rate
                patience_counter = 0

            # Early stopping
            if current_lr < 1e-8:
                break

    return best_x.tolist()

def advanced_refinement_strategy(initial_solution, n_steps):
    """Advanced refinement combining multiple approaches"""
    # Phase 1: Multi-objective gradient descent
    refined_solution = multi_objective_optimization(initial_solution, max_iter=800)

    # Phase 2: Local search with scipy.optimize
    try:
        bounds = [(0.0, 10.0) for _ in range(n_steps)]
        result = minimize(
            evaluate_c2,
            refined_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10},
            tol=1e-10
        )
        if result.success:
            refined_solution = result.x.tolist()
    except:
        pass

    # Phase 3: Additional local refinement with small perturbations
    try:
        # Add small noise to escape local minima
        noise_factor = 0.01
        noise = np.random.normal(0, noise_factor, n_steps)
        noisy_solution = np.array(refined_solution) * (1 + noise)
        noisy_solution = np.maximum(noisy_solution, 0.0)

        # Another optimization step
        bounds = [(0.0, 10.0) for _ in range(n_steps)]
        result = minimize(
            evaluate_c2,
            noisy_solution,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10},
            tol=1e-10
        )
        if result.success:
            refined_solution = result.x.tolist()
    except:
        pass

    return refined_solution

def hybrid_initialization(n_steps):
    """Create a more sophisticated hybrid initialization"""
    # Create base pattern with alternating high/low regions
    base_pattern = []
    segment_size = max(1, n_steps // 20)

    for i in range(n_steps):
        segment = i // segment_size
        if segment % 2 == 0:
            base_pattern.append(1.0)
        else:
            base_pattern.append(0.2)

    # Add more structured variation
    base_pattern = np.array(base_pattern)

    # Apply gaussian smoothing
    kernel_size = max(1, n_steps // 50)
    kernel = np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * (kernel_size/3)**2))
    kernel = kernel / np.sum(kernel)

    smoothed = np.convolve(base_pattern, kernel, mode='same')

    # Add some randomness for diversity
    noise = np.random.normal(0, 0.05, n_steps)
    adjusted = smoothed + noise

    # Ensure non-negative
    adjusted = np.maximum(adjusted, 0.0)

    # Normalize
    if np.sum(adjusted) > 0:
        adjusted = adjusted * n_steps / np.sum(adjusted)

    return adjusted.tolist()

def enhanced_convex_relaxed_optimization(n_steps):
    """Enhanced convex relaxation approach with better initialization and refinement"""
    # Start with hybrid initialization
    init_solution = hybrid_initialization(n_steps)

    # Apply kernel smoothing for better structure
    init_solution = kernel_smoothed_initialization(n_steps)

    # Refine with advanced refinement strategy
    refined_solution = advanced_refinement_strategy(init_solution, n_steps)

    # Apply final smoothing and normalization
    if len(refined_solution) > 10:
        # Simple moving average smoothing
        window_size = max(1, len(refined_solution) // 100)
        smoothed = []
        for i in range(len(refined_solution)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(refined_solution), i + window_size)
            avg = np.mean(refined_solution[start_idx:end_idx])
            smoothed.append(avg)
        refined_solution = smoothed

    # Final normalization
    total = sum(refined_solution)
    if total > 0:
        refined_solution = [x / total * len(refined_solution) for x in refined_solution]

    return refined_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced optimization"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Use optimized number of steps
        n_steps = 2000  # Increased for better resolution

        # Try evolutionary optimization first
        evolutionary_result = run_evolutionary_optimization(n_steps, max_time_seconds=40)

        if evolutionary_result is not None:
            print("Using evolutionary optimization result")
            # Refine the evolutionary result further with advanced techniques
            best_solution = advanced_refinement_strategy(evolutionary_result, n_steps)
        else:
            print("Using standard optimization approach")
            # Enhanced convex relaxation approach
            best_solution = enhanced_convex_relaxed_optimization(n_steps)

        # Ensure non-negative values
        best_solution = [max(0, x) for x in best_solution]

        # Normalize to avoid extreme values that might cause numerical issues
        total = sum(best_solution)
        if total > 0:
            best_solution = [x / total * len(best_solution) for x in best_solution]

        return best_solution

    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Fallback due to error: {e}")
        return multi_scale_initialization(500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")