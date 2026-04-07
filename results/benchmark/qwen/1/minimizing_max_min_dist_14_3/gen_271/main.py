# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral method"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def sobol_like_distribution(n):
    """Generate points using a Sobol-like distribution for better space-filling"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))

    # Generate Fibonacci points but with Sobol-like spacing adjustments
    for i in range(n):
        # Standard Fibonacci approach
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        # Apply small perturbations to improve distribution (Sobol-like)
        # Use quasi-random sequence for perturbations
        pert_x = 0.02 * np.sin(i * 1.618) * np.cos(i * 0.785)
        pert_y = 0.02 * np.cos(i * 1.618) * np.sin(i * 0.785)
        pert_z = 0.02 * np.sin(i * 0.785) * np.cos(i * 1.618)

        x += pert_x
        y += pert_y
        z += pert_z

        # Normalize to unit sphere
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            x /= norm
            y /= norm
            z /= norm

        points.append([x, y, z])

    return np.array(points)

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance between all point pairs"""
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist <= 0:
        return 0
    return min_dist / max_dist

def optimize_points_global(initial_points, max_iter=500):
    """Global optimization using differential evolution"""
    def objective(x):
        points = x.reshape(-1, 3)
        # We want to maximize the ratio, so we minimize its negative
        ratio = min_max_dist_ratio(points)
        if ratio <= 0:
            return 1e10
        return -ratio

    n_vars = len(initial_points) * 3
    bounds = [(-1, 1) for _ in range(n_vars)]
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=max_iter,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            # Ensure points are on unit sphere
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return optimized_points / norms
        else:
            return initial_points
            
    except Exception:
        return initial_points

def optimize_points_local(initial_points, max_iter=500):
    """Local optimization using L-BFGS-B"""
    def objective(x):
        points = x.reshape(-1, 3)
        # We want to maximize the ratio, so we minimize its negative
        ratio = min_max_dist_ratio(points)
        if ratio <= 0:
            return 1e10
        return -ratio

    def constraint_sphere(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    # Flatten initial points for optimization
    x0 = initial_points.flatten()
    
    # Define bounds (slightly relaxed for better exploration)
    bounds = [(-1.1, 1.1) for _ in range(len(x0))]

    cons = [{'type': 'ineq', 'fun': constraint_sphere}]

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            # Ensure points are on unit sphere
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return optimized_points / norms
        else:
            return initial_points
            
    except Exception:
        return initial_points

def generate_diverse_configs(n):
    """Generate multiple diverse starting configurations"""
    configs = []
    
    # Configuration 1: Standard Fibonacci
    configs.append(fibonacci_sphere(n))
    
    # Configuration 2: Fibonacci with small perturbations
    fib_points = fibonacci_sphere(n)
    np.random.seed(100)
    perturbed = fib_points + np.random.normal(0, 0.02, fib_points.shape)
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    configs.append(perturbed / norms)
    
    # Configuration 3: Random points on sphere
    np.random.seed(200)
    random_points = np.random.randn(n, 3)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    configs.append(random_points / norms)
    
    # Configuration 4: Alternative Fibonacci variant
    np.random.seed(300)
    alt_fib = np.zeros((n, 3))
    for i in range(n):
        theta = np.arccos(1 - 2*(i/(n-1)))
        phi = i * 4 * np.pi / (1 + np.sqrt(5))  # Different golden ratio variant
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        alt_fib[i] = [x, y, z]
    configs.append(alt_fib)
    
    # Configuration 5: Perturbed random
    np.random.seed(400)
    perturbed_random = np.random.randn(n, 3)
    perturbed_random = perturbed_random / np.linalg.norm(perturbed_random, axis=1, keepdims=True)
    configs.append(perturbed_random)
    
    # Configuration 6: Sobol-like distribution
    configs.append(sobol_like_distribution(n))
    
    # Configuration 7: Perturbed Sobol-like
    sobol_points = sobol_like_distribution(n)
    np.random.seed(500)
    perturbed_sobol = sobol_points + np.random.normal(0, 0.01, sobol_points.shape)
    norms = np.linalg.norm(perturbed_sobol, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    configs.append(perturbed_sobol / norms)
    
    # Configuration 8: Another Fibonacci variant with different parameters
    np.random.seed(600)
    fib_alt2 = np.zeros((n, 3))
    for i in range(n):
        # Different spacing formula
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = (i * 1.618033988749895) % (2 * np.pi)  # Golden ratio multiplication
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        # Add small perturbations
        x += 0.01 * np.sin(i)
        y += 0.01 * np.cos(i)
        z += 0.01 * np.sin(i * 2)
        
        # Normalize
        norm = np.sqrt(x*x + y*y + z*z)
        if norm > 0:
            x /= norm
            y /= norm
            z /= norm
            
        fib_alt2[i] = [x, y, z]
    configs.append(fib_alt2)
    
    return configs

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Generate diverse starting configurations
    configs = generate_diverse_configs(14)
    
    best_ratio = 0
    best_points = None
    
    # Try each configuration with multi-stage optimization
    for i, config in enumerate(configs):
        try:
            # Phase 1: Global optimization
            global_optimized = optimize_points_global(config, max_iter=300)
            
            # Phase 2: Local refinement
            local_optimized = optimize_points_local(global_optimized, max_iter=300)
            
            # Evaluate the result
            ratio = min_max_dist_ratio(local_optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = local_optimized.copy()
                
        except Exception as e:
            continue
    
    # If no improvement found, fallback to the best configuration
    if best_points is None:
        # Return the configuration with highest initial ratio
        best_initial_ratio = 0
        for config in configs:
            ratio = min_max_dist_ratio(config)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_points = config.copy()
    
    # Final safeguard: ensure points are on unit sphere
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        best_points = best_points / norms
    
    # Fallback to Fibonacci if nothing worked
    if best_points is None:
        return fibonacci_sphere(14)
    
    return best_points

# EVOLVE-BLOCK-END