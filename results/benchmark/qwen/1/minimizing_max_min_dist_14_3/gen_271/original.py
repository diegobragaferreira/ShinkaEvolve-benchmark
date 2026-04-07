# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
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

def compute_voronoi_objective(points):
    """Compute objective based on Voronoi cell uniformity and minimum distance"""
    try:
        sv = SphericalVoronoi(points)
        areas = sv.voronoi_cell_areas()
        if len(areas) > 0:
            mean_area = np.mean(areas)
            if mean_area > 0:
                # Coefficient of variation of areas (lower is better)
                cv = np.std(areas) / mean_area
                # Also consider the minimum distance
                distances = cdist(points, points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    # Combine uniformity with distance ratio
                    distance_ratio = min_dist / max_dist
                    # Weighted combination favoring uniformity but preserving distance quality
                    return -cv - 0.1 * (1 - distance_ratio)
                else:
                    return 1e10
            else:
                return 1e10
        else:
            return 1e10
    except:
        return 1e10

def spherical_lloyd_step(points):
    """Apply one step of Lloyd's algorithm on sphere"""
    try:
        sv = SphericalVoronoi(points)
        centroids = sv._voronoi_cell_centroids()
        # Normalize centroids to sphere
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return centroids / norms
    except:
        return points

def optimize_voronoi_based(initial_points, max_iterations=100):
    """Optimize using Voronoi-based iterative refinement"""
    points = initial_points.copy()
    
    # Initial Voronoi-based improvement
    for _ in range(10):
        new_points = spherical_lloyd_step(points)
        # Check if improvement is significant
        diff = np.mean(np.linalg.norm(new_points - points, axis=1))
        if diff < 1e-6:
            break
        points = new_points
    
    # Gradient-based refinement using Voronoi-aware objective
    def objective(x):
        points = x.reshape(-1, 3)
        return compute_voronoi_objective(points)
    
    def constraint_sphere(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms
    
    # Use L-BFGS-B for local optimization
    x0 = points.flatten()
    cons = [{'type': 'ineq', 'fun': constraint_sphere}]
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            constraints=cons,
            options={'maxiter': 50, 'ftol': 1e-12},
            tol=1e-12
        )
        if result.success:
            points = result.x.reshape(-1, 3)
            # Normalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
    except:
        pass
    
    return points

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
    
    return configs

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance between all point pairs"""
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist <= 0:
        return 0
    return min_dist / max_dist

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
    
    # Try each configuration with Voronoi-based optimization
    for i, config in enumerate(configs):
        try:
            # Apply Voronoi-based optimization
            optimized = optimize_voronoi_based(config, max_iterations=100)
            
            # Evaluate the result
            ratio = min_max_dist_ratio(optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized.copy()
                
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