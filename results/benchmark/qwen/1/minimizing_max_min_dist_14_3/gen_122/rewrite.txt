# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time

def sobol_points_sphere(n_points):
    """Generate points on sphere using 3D Sobol sequence for superior space-filling properties."""
    try:
        from sobol_seq import i4_sobol_generate
        sobol_points = i4_sobol_generate(3, n_points)
        
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            u = sobol_points[i, 0]
            v = sobol_points[i, 1]
            
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            points[i] = [x, y, z]
        return points
    except ImportError:
        # Fallback to fibonacci if sobol not available
        return fibonacci_spiral_sphere(n_points)

def fibonacci_spiral_sphere(n_points):
    """Generate points on a sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist < 1e-12:
        return 0.0
    return min_dist / max_dist

def initialize_14_points():
    """Initialize 14 points using multiple strategies for better coverage."""
    # Strategy 1: Sobol sequence initialization
    sobol_points = sobol_points_sphere(14)
    
    # Strategy 2: Fibonacci spiral with perturbation
    fib_points = fibonacci_spiral_sphere(14)
    
    # Strategy 3: Icosahedron-based initialization  
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    ico_points = np.vstack([vertices, [[0, 0, 1], [0, 0, -1]]])
    
    # Return multiple initial configurations
    return [sobol_points, fib_points, ico_points]

def calculate_voronoi_forces(points, alpha=0.1, beta=0.05):
    """Calculate forces based on Voronoi cell properties for point movement."""
    n = len(points)
    
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points)
        sv.sort_vertices_of_regions()
        
        # Get cell centers and areas
        cell_centers = []
        cell_areas = []
        for i, region in enumerate(sv.regions):
            area = sv.calculate_area_of_region(region)
            cell_areas.append(area)
            
            # Calculate centroid of cell
            if len(region) > 0:
                cell_points = points[region]
                center = np.mean(cell_points, axis=0)
                center = center / np.linalg.norm(center)
                cell_centers.append(center)
            else:
                cell_centers.append(points[i])
        
        # Calculate neighbor relationships based on distance
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Find 6 nearest neighbors for each point
        neighbors = []
        for i in range(n):
            nearest = np.argsort(distances[i])[:6]
            neighbors.append(nearest.tolist())
        
        # Calculate forces
        new_points = points.copy()
        for i in range(n):
            # Attraction force towards neighbors (weighted by inverse distance)
            neighbor_points = points[neighbors[i]]
            if len(neighbor_points) > 0:
                dist_to_neighbors = cdist(points[i:i+1], neighbor_points)[0]
                weights = 1.0 / np.maximum(dist_to_neighbors, 1e-8)
                weights = weights / np.sum(weights)
                
                avg_neighbor = np.average(neighbor_points, axis=0, weights=weights)
                avg_neighbor = avg_neighbor / np.linalg.norm(avg_neighbor)
                
                # Apply attraction force
                new_points[i] = (1 - alpha) * points[i] + alpha * avg_neighbor
                
                # Repulsion from close neighbors
                close_indices = np.where(dist_to_neighbors < 0.25)[0]
                if len(close_indices) > 0:
                    close_points = neighbor_points[close_indices]
                    for close_point in close_points:
                        diff = points[i] - close_point
                        diff_norm = np.linalg.norm(diff)
                        if diff_norm > 1e-6:
                            repulsion = beta * (diff / diff_norm) * (0.25 - diff_norm)
                            new_points[i] += repulsion
            
            # Project back onto sphere
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-12:
                new_points[i] = new_points[i] / norm
                
        return new_points
        
    except Exception:
        # Fallback to simple smoothing
        new_points = points.copy()
        for i in range(n):
            if i < n - 1:
                avg = (points[i] + points[i+1]) / 2
                avg = avg / np.linalg.norm(avg)
                new_points[i] = 0.95 * points[i] + 0.05 * avg
            else:
                avg = (points[i] + points[0]) / 2
                avg = avg / np.linalg.norm(avg)
                new_points[i] = 0.95 * points[i] + 0.05 * avg
        return new_points

def adaptive_optimization_stage(points, max_iter=1000, stage=1):
    """Perform adaptive optimization stage with varying parameters."""
    n = len(points)
    
    def objective(x_flat):
        points_reshaped = x_flat.reshape(n, 3)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / np.maximum(norms, 1e-12)
        return -min_max_dist_ratio(normalized_points)

    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(n, 3)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0

    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * 3)]
    
    # Adaptive parameters based on stage
    if stage == 1:  # Coarse optimization
        method = 'SLSQP'
        options = {'maxiter': max_iter//3, 'ftol': 1e-6, 'gtol': 1e-6}
        tol = 1e-6
    elif stage == 2:  # Medium optimization
        method = 'trust-constr'
        options = {'maxiter': max_iter//3, 'xtol': 1e-9, 'gtol': 1e-9}
        tol = 1e-9
    else:  # Fine optimization
        method = 'L-BFGS-B' 
        options = {'maxiter': max_iter//3, 'ftol': 1e-12, 'gtol': 1e-12}
        tol = 1e-12
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=tol
        )
        
        if result.success:
            optimized_points = result.x.reshape(n, 3)
            # Ensure points are on unit sphere
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            normalized_points = optimized_points / np.maximum(norms, 1e-12)
            return normalized_points
    except Exception:
        pass
        
    return points

def voronoi_refinement(points, iterations=50):
    """Refine points using Voronoi-based evolution."""
    current_points = points.copy()
    
    for i in range(iterations):
        # Adaptive learning rate
        alpha = 0.05 * (1 - i / iterations)
        beta = 0.01 * (1 - i / iterations)
        
        # Apply Voronoi forces
        evolved_points = calculate_voronoi_forces(current_points, alpha, beta)
        
        # Add small random perturbations to avoid getting stuck
        if i % 5 == 0:
            noise_magnitude = 0.005 * (1 - i / iterations)
            noise = np.random.normal(0, noise_magnitude, evolved_points.shape)
            evolved_points += noise
            
            # Project back onto sphere
            norms = np.linalg.norm(evolved_points, axis=1, keepdims=True)
            evolved_points = evolved_points / np.maximum(norms, 1e-12)
            
        current_points = evolved_points
        
    return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Initialize multiple configurations
    initial_configs = initialize_14_points()
    
    best_ratio = -np.inf
    best_points = None
    
    # Try each initial configuration with different optimization strategies
    for i, initial_points in enumerate(initial_configs):
        # Add slight random noise to break symmetry
        np.random.seed(42 + i * 10)
        noisy_points = initial_points + np.random.normal(0, 0.02, (n, d))
        
        # Ensure all points are on unit sphere
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        normalized_points = noisy_points / np.maximum(norms, 1e-12)
        
        # Apply Voronoi refinement first
        refined_points = voronoi_refinement(normalized_points, iterations=30)
        
        # Multi-stage optimization
        optimized_points = adaptive_optimization_stage(refined_points, max_iter=500, stage=1)
        optimized_points = adaptive_optimization_stage(optimized_points, max_iter=500, stage=2)
        optimized_points = adaptive_optimization_stage(optimized_points, max_iter=500, stage=3)
        
        # Final Voronoi refinement
        optimized_points = voronoi_refinement(optimized_points, iterations=20)
        
        # Evaluate solution
        ratio = min_max_dist_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # If no good solution found, return a random configuration
    if best_points is None:
        np.random.seed(42)
        random_points = np.random.randn(n, d)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        best_points = random_points / np.maximum(norms, 1e-12)
    
    return best_points

# EVOLVE-BLOCK-END