# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
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
    
    def voronoi_uniformity_score(points):
        """Calculate score based on Voronoi cell area uniformity"""
        try:
            # Project points to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized_points = points / norms
            
            sv = SphericalVoronoi(normalized_points)
            areas = sv.voronoi_cell_areas()
            
            # Return coefficient of variation of areas (lower is better for uniformity)
            if np.std(areas) > 1e-10:
                return np.std(areas) / np.mean(areas)
            else:
                return 0.0
        except:
            return 1.0
    
    def distance_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie exactly on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def spherical_projection(points):
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-12)
    
    def spherical_voronoi_evolution_step(points, learning_rate=0.05):
        """
        Single evolution step using Voronoi-based point adjustment
        """
        # Create Voronoi diagram
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            
            # Compute centroid of Voronoi cells
            centroids = []
            for i in range(len(points)):
                # Get vertices of Voronoi cell around point i
                cell_vertices = sv.vertices[sv.regions[i]]
                if len(cell_vertices) > 0:
                    # Average the vertices to get cell centroid
                    centroid = np.mean(cell_vertices, axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(points[i])
            
            centroids = np.array(centroids)
            
            # Adjust points towards better Voronoi cell centroids
            adjusted_points = []
            for i, (point, centroid) in enumerate(zip(points, centroids)):
                # Move point towards corresponding cell centroid
                direction = centroid - point
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-12:
                    # Normalize and scale by learning rate
                    step_direction = direction / norm_dir
                    new_point = point + learning_rate * step_direction
                    
                    # Project back to sphere
                    new_point = spherical_projection(new_point.reshape(1, 3)).flatten()
                else:
                    new_point = point
                    
                adjusted_points.append(new_point)
            
            return np.array(adjusted_points)
            
        except:
            # Fallback: simple gradient descent on distance ratio
            return points
    
    def local_improve_step(points, max_iter=50):
        """
        Local optimization using gradient-based approach on distance ratio
        """
        def objective(x):
            points_reshaped = x.reshape(-1, 3)
            # Ensure points are on unit sphere
            norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
            normalized_points = points_reshaped / np.maximum(norms, 1e-12)
            return -distance_ratio(normalized_points)
        
        def constraint_sphere(x):
            points_reshaped = x.reshape(-1, 3)
            norms = np.linalg.norm(points_reshaped, axis=1)
            return norms - 1.0
        
        constraints = []
        for i in range(14):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_sphere(x)[i]})
        
        bounds = [(-1.2, 1.2)] * len(points.flatten())
        
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass
        
        return points
    
    def adaptive_initialization(n_points):
        """Generate high-quality initial points"""
        # Start with Fibonacci points on sphere
        fib_points = fibonacci_sphere(n_points)
        
        # Normalize to unit sphere
        fib_points = normalize_to_unit_sphere(fib_points)
        
        # Add some randomness to break symmetries
        noise_magnitude = 0.08
        noise = np.random.normal(0, noise_magnitude, fib_points.shape)
        initial_points = fib_points + noise
        
        # Re-normalize to sphere
        initial_points = normalize_to_unit_sphere(initial_points)
        
        return initial_points
    
    def spherical_voronoi_evolution(initial_points, max_iterations=1000):
        """
        Core evolution algorithm using spherical Voronoi properties
        """
        current_points = initial_points.copy()
        
        best_points = current_points.copy()
        best_ratio = distance_ratio(current_points)
        best_uniformity = voronoi_uniformity_score(current_points)
        
        # Evolve with decreasing learning rate
        for iteration in range(max_iterations):
            # Every 20 iterations, do local improvement
            if iteration % 20 == 0:
                current_points = local_improve_step(current_points)
            
            # Apply Voronoi evolution step
            learning_rate = max(0.01, 0.1 * (1 - iteration / max_iterations))
            current_points = spherical_voronoi_evolution_step(current_points, learning_rate)
            
            # Keep points on sphere
            current_points = normalize_to_unit_sphere(current_points)
            
            # Check for improvement
            current_ratio = distance_ratio(current_points)
            current_uniformity = voronoi_uniformity_score(current_points)
            
            # Prefer solutions with better ratio, but also consider uniformity
            current_score = current_ratio - 0.2 * current_uniformity
            
            if current_score > best_ratio - 0.2 * best_uniformity:
                best_ratio = current_ratio
                best_uniformity = current_uniformity
                best_points = current_points.copy()
            
            # Early stopping if improvement is minimal
            if iteration > 100 and iteration % 100 == 0:
                last_improvement = iteration - 100
                if best_ratio < 0.4898:  # Benchmark threshold
                    break
        
        return best_points
    
    # Multi-start with different initializations
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple configurations
    initial_configs = []
    
    # Config 1: Fibonacci with noise
    np.random.seed(42)
    fib_points = adaptive_initialization(14)
    initial_configs.append(("fibonacci", fib_points))
    
    # Config 2: Random but normalized
    np.random.seed(123)
    random_points = np.random.randn(14, 3)
    random_points = normalize_to_unit_sphere(random_points)
    initial_configs.append(("random", random_points))
    
    # Config 3: Perturbed Fibonacci
    np.random.seed(456)
    perturbed_fib = adaptive_initialization(14)
    noise = np.random.normal(0, 0.05, perturbed_fib.shape)
    perturbed_fib += noise
    perturbed_fib = normalize_to_unit_sphere(perturbed_fib)
    initial_configs.append(("perturbed_fib", perturbed_fib))
    
    # Run evolution on each configuration
    for config_name, initial_points in initial_configs:
        try:
            evolved_points = spherical_voronoi_evolution(initial_points, max_iterations=500)
            
            ratio = distance_ratio(evolved_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = evolved_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to best initialization if evolution fails
    if best_points is None:
        np.random.seed(42)
        initial_points = adaptive_initialization(14)
        best_points = initial_points
    
    return best_points

# EVOLVE-BLOCK-END