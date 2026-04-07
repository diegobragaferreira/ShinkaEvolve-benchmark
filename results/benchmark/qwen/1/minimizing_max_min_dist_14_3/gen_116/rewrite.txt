# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance between all point pairs"""
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist <= 0:
        return 0
    return min_dist / max_dist

def compute_voronoi_uniformity(points):
    """Compute a measure of how uniform the spherical Voronoi cells are"""
    try:
        sv = SphericalVoronoi(points)
        areas = sv.voronoi_cell_areas()
        # Return coefficient of variation of cell areas (lower is more uniform)
        if len(areas) > 0:
            mean_area = np.mean(areas)
            if mean_area > 0:
                cv = np.std(areas) / mean_area
                return cv
        return 1.0
    except:
        return 1.0

def optimize_with_multi_start(initial_points, num_starts=8, max_iter=500):
    """Run optimization from multiple starting points to find better solution"""
    best_ratio = 0
    best_points = initial_points.copy()

    # Generate multiple diverse starting configurations
    configs = []
    
    # Original Fibonacci
    configs.append(initial_points.copy())
    
    # Perturbed Fibonacci with different strengths
    for pert_strength in [0.02, 0.03, 0.05]:
        np.random.seed(42)
        perturbed = initial_points + np.random.normal(0, pert_strength, initial_points.shape)
        configs.append(perturbed.copy())
    
    # Random configurations
    for seed_val in [100, 200, 300, 400]:
        np.random.seed(seed_val)
        random_points = np.random.randn(14, 3)
        random_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)
        configs.append(random_points.copy())

    for i, start_config in enumerate(configs):
        try:
            # Normalize to unit sphere
            norms = np.linalg.norm(start_config, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized_config = start_config / norms
            
            # Apply Differential Evolution for global search
            n_vars = 14 * 3
            bounds = [(-1, 1) for _ in range(n_vars)]
            
            def objective_DE(x):
                points = x.reshape(-1, 3)
                distances = cdist(points, points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist <= 0:
                    return 1e10
                return -min_dist / max_dist
            
            de_result = differential_evolution(
                objective_DE,
                bounds,
                seed=42 + i,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                # Normalize DE result
                norms = np.linalg.norm(de_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                de_points = de_points / norms
                
                # Fine-tune with local optimization
                def objective_local(x):
                    points = x.reshape(-1, 3)
                    distances = cdist(points, points)
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist <= 0:
                        return 1e10
                    return -min_dist / max_dist
                
                def constraint_sphere(x):
                    points = x.reshape(-1, 3)
                    norms = np.linalg.norm(points, axis=1)
                    return 1 - norms
                
                try:
                    cons = [{'type': 'ineq', 'fun': constraint_sphere}]
                    result = minimize(
                        objective_local,
                        de_points.flatten(),
                        method='L-BFGS-B',
                        constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-10},
                        tol=1e-10
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        norms = np.linalg.norm(final_points, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        final_points = final_points / norms
                        
                        # Evaluate final ratio
                        ratio = min_max_dist_ratio(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                except:
                    pass
                    
        except Exception as e:
            continue
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Initialize points using Fibonacci spiral on sphere
    np.random.seed(42)
    points = fibonacci_sphere(14)
    
    # Improve with multi-start optimization
    optimized_points = optimize_with_multi_start(points, num_starts=8, max_iter=500)
    
    return optimized_points

# EVOLVE-BLOCK-END