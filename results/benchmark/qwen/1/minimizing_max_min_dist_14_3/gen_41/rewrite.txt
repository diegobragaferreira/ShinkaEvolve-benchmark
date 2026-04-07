# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        return -np.min(distances)

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_max_distance(x):
        # Ensure maximum distance doesn't exceed some reasonable bound
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        return 2 - max_dist  # Should be >= 0 (allowing up to diameter 2)
        
    def normalize_points(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def generate_initial_spherical_config(n):
        """Generate a good initial configuration using Fibonacci-like approach on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)
    
    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def generate_random_points(n):
        """Generate random points on unit sphere"""
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def generate_perturbed_fibonacci_points(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = generate_fibonacci_points(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    # Multiple starting configurations
    configs = []
    
    # Configuration 1: Fibonacci spiral on sphere
    configs.append(generate_initial_spherical_config(14))
    
    # Configuration 2: Random points on sphere
    np.random.seed(42)
    random_points = np.random.randn(14, 3)
    random_points = normalize_points(random_points)
    configs.append(random_points)
    
    # Configuration 3: Perturbed Fibonacci
    fib_points = generate_initial_spherical_config(14)
    np.random.seed(100)
    perturbed_fib = fib_points + np.random.normal(0, 0.02, fib_points.shape)
    perturbed_fib = normalize_points(perturbed_fib)
    configs.append(perturbed_fib)
    
    # Configuration 4: Another random seed
    np.random.seed(200)
    random_points2 = np.random.randn(14, 3)
    random_points2 = normalize_points(random_points2)
    configs.append(random_points2)
    
    # Configuration 5: Fibonacci points
    configs.append(generate_fibonacci_points(14))
    
    # Configuration 6: Perturbed Fibonacci with larger perturbation
    np.random.seed(300)
    perturbed_fib_large = generate_fibonacci_points(14) + np.random.normal(0, 0.05, (14, 3))
    perturbed_fib_large = normalize_points(perturbed_fib_large)
    configs.append(perturbed_fib_large)

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0
    
    # Optimization with multiple restarts
    for i, initial_config in enumerate(configs):
        try:
            # Use differential evolution for global search first
            n_points = 14
            n_vars = n_points * 3  # 14 points * 3 coordinates each
            
            # Bounds for each coordinate: [-1, 1] to allow for sphere constraint
            bounds = [(-1, 1) for _ in range(n_vars)]
            
            # Run differential evolution with the current initial config
            result = differential_evolution(
                objective,
                bounds,
                seed=42 + i,
                maxiter=500,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            # Extract optimized points and normalize
            optimized_points = result.x.reshape(-1, 3)
            optimized_points = normalize_points(optimized_points)
            
            # Evaluate this solution
            distances = cdist(optimized_points, optimized_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = optimized_points.copy()
                    
        except Exception as e:
            continue
    
    # If no good solution found, try local optimization from best configurations
    if best_final_points is None:
        # Try optimizing from the best initial configurations using local method
        for i, initial_config in enumerate(configs[:3]):  # Try first 3 configs
            try:
                # Local optimization around initial point
                x0 = initial_config.flatten()
                cons = [
                    {'type': 'ineq', 'fun': constraint_sphere},
                    {'type': 'ineq', 'fun': constraint_max_distance}
                ]
                
                result = minimize(objective, x0, method='SLSQP', constraints=cons,
                                options={'ftol': 1e-8, 'maxiter': 500})
                
                optimized_points = result.x.reshape(-1, 3)
                optimized_points = normalize_points(optimized_points)
                
                # Evaluate this solution
                distances = cdist(optimized_points, optimized_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_final_points = optimized_points.copy()
                        
            except Exception as e:
                continue
    
    # If still no solution, return the last attempt or fallback to Fibonacci
    if best_final_points is None:
        return generate_initial_spherical_config(14)
        
    return best_final_points

# EVOLVE-BLOCK-END