# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
from numba import jit
import warnings

@jit(nopython=True)
def fast_pairwise_distance_matrix(points):
    """Fast computation of pairwise distance matrix using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def voronoi_energy_objective(x):
        """Minimize Voronoi cell area variance to promote uniform distribution"""
        points = x.reshape(-1, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points_normalized = points / norms[:, np.newaxis]
        
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points_normalized)
            cell_areas = sv.calculate_areas()
            # Minimize variance of cell areas (maximize uniformity)
            return np.var(cell_areas)
        except:
            # Fallback to distance-based measure if Voronoi fails
            distances = fast_pairwise_distance_matrix(points_normalized)
            np.fill_diagonal(distances, np.inf)
            # Maximize minimum distance as fallback
            min_dist = np.min(distances)
            return -min_dist
            
    def distance_ratio_objective(x):
        """Direct objective: maximize min/max distance ratio"""
        points = x.reshape(-1, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points_normalized = points / norms[:, np.newaxis]
        
        # Use fast distance computation
        distances = fast_pairwise_distance_matrix(points_normalized)
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max < 1e-12:
            return 1e12  # Penalize invalid configurations
            
        # Return negative ratio (we minimize to maximize ratio)
        return -d_min / d_max
    
    def constraint_sphere(x):
        """Constraint to keep points on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def fibonacci_sphere_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Golden angle spiral
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.arccos(y)  # angle from z-axis
            phi = (i * 2 * np.pi) / golden_ratio  # azimuthal angle
            
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_initial_population():
        """Create diverse initial configurations"""
        configs = []
        
        # Method 1: Fibonacci sphere distribution
        fib_points = fibonacci_sphere_points(14)
        configs.append(fib_points)
        
        # Method 2: Random points normalized
        np.random.seed(42)
        rand_points = np.random.randn(14, 3)
        configs.append(rand_points)
        
        # Method 3: Slightly perturbed Fibonacci
        perturbed_fib = fib_points + np.random.normal(0, 0.05, fib_points.shape)
        configs.append(perturbed_fib)
        
        # Method 4: Perturbed random
        perturbed_rand = rand_points + np.random.normal(0, 0.1, rand_points.shape)
        configs.append(perturbed_rand)
        
        return configs
    
    # Generate diverse initial population
    initial_configs = generate_initial_population()
    
    best_points = None
    best_ratio = -np.inf
    
    # Try different initial configurations
    for i, initial_config in enumerate(initial_configs):
        # Normalize initial points to unit sphere
        norms = np.linalg.norm(initial_config, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_points = initial_config / norms[:, np.newaxis]
        
        # Initial optimization using Voronoi energy to get good spread
        x0 = normalized_points.flatten()
        cons = [{'type': 'eq', 'fun': constraint_sphere}]
        
        try:
            # First phase: energy-based optimization for good distribution
            result_energy = minimize(
                voronoi_energy_objective,
                x0,
                method='L-BFGS-B',
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result_energy.success:
                # Second phase: refine with distance ratio optimization
                refined_x = result_energy.x
                result_distance = minimize(
                    distance_ratio_objective,
                    refined_x,
                    method='L-BFGS-B',
                    constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result_distance.success:
                    final_points = result_distance.x.reshape(-1, 3)
                    
                    # Calculate final ratio
                    norms_final = np.linalg.norm(final_points, axis=1)
                    norms_final = np.where(norms_final == 0, 1.0, norms_final)
                    final_normalized = final_points / norms_final[:, np.newaxis]
                    
                    distances = fast_pairwise_distance_matrix(final_normalized)
                    np.fill_diagonal(distances, np.inf)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_normalized.copy()
                            
        except Exception as e:
            continue
    
    # Fall back to Fibonacci if nothing worked
    if best_points is None:
        fib_points = fibonacci_sphere_points(14)
        norms = np.linalg.norm(fib_points, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        best_points = fib_points / norms[:, np.newaxis]
    
    return best_points

# EVOLVE-BLOCK-END