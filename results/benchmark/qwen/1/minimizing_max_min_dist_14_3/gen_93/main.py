# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist, cdist
import time
from sklearn.cluster import KMeans

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

def fibonacci_spiral_sphere(n_points):
    """Generate points on a sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def initialize_14_points():
    """Initialize 14 points using a combination of geometric constructions."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Add two more points for 14 total - use poles
    points = np.vstack([vertices, [[0, 0, 1], [0, 0, -1]]])
    
    # Perturb slightly to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.03, (points.shape[0], 3))
    
    # Re-normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / np.maximum(norms[:, np.newaxis], 1e-12)
    
    return points

def calculate_voronoi_properties(points):
    """Calculate Voronoi cell areas and geometric properties."""
    # Create spherical Voronoi diagram
    sv = SphericalVoronoi(points)
    sv.sort_vertices_of_regions()
    
    # Calculate areas of Voronoi cells
    cell_areas = []
    for region in sv.regions:
        area = sv.calculate_area_of_region(region)
        cell_areas.append(area)
    
    return np.array(cell_areas)

def sphere_voronoi_evolution_step(points, alpha=0.1, beta=0.05):
    """Perform one evolution step based on Voronoi properties."""
    n = len(points)
    
    # Calculate current Voronoi properties
    try:
        sv = SphericalVoronoi(points)
        sv.sort_vertices_of_regions()
        
        # Get cell centers and areas
        cell_centers = []
        cell_areas = []
        for i, region in enumerate(sv.regions):
            area = sv.calculate_area_of_region(region)
            cell_areas.append(area)
            
            # Calculate centroid of cell
            cell_points = points[region] if len(region) > 0 else points
            center = np.mean(cell_points, axis=0)
            center = center / np.linalg.norm(center)
            cell_centers.append(center)
        
        cell_areas = np.array(cell_areas)
        
        # Calculate neighbor relationships
        neighbors = {}
        for i in range(n):
            neighbors[i] = []
        
        # For simplicity, use a distance-based neighbor definition
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Each point connects to its 4 nearest neighbors
        for i in range(n):
            nearest = np.argsort(distances[i])[:4]
            neighbors[i] = nearest.tolist()
        
        # Update points based on Voronoi geometry
        new_points = points.copy()
        
        # Apply force-based adjustment
        for i in range(n):
            # Move towards average of neighbors, weighted by inverse distance
            neighbor_points = points[neighbors[i]]
            if len(neighbor_points) > 0:
                # Calculate weights based on distances to neighbors
                dist_to_neighbors = cdist(points[i:i+1], neighbor_points)[0]
                # Avoid division by zero
                weights = 1.0 / np.maximum(dist_to_neighbors, 1e-8)
                weights = weights / np.sum(weights)
                
                # Weighted average of neighbors
                avg_neighbor = np.average(neighbor_points, axis=0, weights=weights)
                avg_neighbor = avg_neighbor / np.linalg.norm(avg_neighbor)
                
                # Apply attraction to neighbors (but keep away from very close points)
                # This creates a balance between clustering and spreading
                new_points[i] = (1 - alpha) * points[i] + alpha * avg_neighbor
                
                # Also apply repulsion from close neighbors
                close_indices = np.where(dist_to_neighbors < 0.3)[0]
                if len(close_indices) > 0:
                    close_points = neighbor_points[close_indices]
                    for close_point in close_points:
                        diff = points[i] - close_point
                        diff_norm = np.linalg.norm(diff)
                        if diff_norm > 1e-6:
                            repulsion = beta * (diff / diff_norm) * (0.3 - diff_norm)
                            new_points[i] += repulsion
                
                # Project back onto sphere
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] = new_points[i] / norm
                    
        return new_points
        
    except Exception:
        # Fallback to simple smoothing if Voronoi fails
        new_points = points.copy()
        for i in range(n):
            # Simple averaging with neighbors
            if i < n - 1:
                avg = (points[i] + points[i+1]) / 2
                avg = avg / np.linalg.norm(avg)
                new_points[i] = 0.9 * points[i] + 0.1 * avg
            else:
                avg = (points[i] + points[0]) / 2
                avg = avg / np.linalg.norm(avg)
                new_points[i] = 0.9 * points[i] + 0.1 * avg
        return new_points

def sphere_voronoi_optimization(points, max_iterations=1000):
    """Optimize points using Voronoi-based evolution."""
    current_points = points.copy()
    
    # Track best solution
    best_ratio = min_max_dist_ratio(current_points)
    best_points = current_points.copy()
    
    # Evolution parameters
    decay_factor = 0.995
    learning_rate = 0.1
    
    for iteration in range(max_iterations):
        # Adaptive learning rate
        current_lr = learning_rate * (decay_factor ** iteration)
        
        # Perform evolution step
        evolved_points = sphere_voronoi_evolution_step(
            current_points, 
            alpha=current_lr * 0.5, 
            beta=current_lr * 0.1
        )
        
        # Evaluate current solution
        ratio = min_max_dist_ratio(evolved_points)
        
        # Accept better solutions or occasionally accept worse ones for escape
        if ratio > best_ratio or (iteration % 100 == 0 and np.random.random() < 0.1):
            current_points = evolved_points.copy()
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = evolved_points.copy()
        else:
            # Occasionally add some noise to escape local minima
            if np.random.random() < 0.05:
                noise_magnitude = 0.01 * current_lr
                noise = np.random.normal(0, noise_magnitude, current_points.shape)
                current_points = (current_points + noise)
                norms = np.linalg.norm(current_points, axis=1)
                current_points = current_points / np.maximum(norms[:, np.newaxis], 1e-12)
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize with a good starting configuration
    initial_points = initialize_14_points()
    
    # Apply Voronoi-based optimization multiple times with different random seeds
    best_ratio = -np.inf
    best_points = None
    
    # Try several random restarts with different seeds
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        
        # Start with fresh random perturbation of the initial configuration
        perturbed_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
        
        # Normalize to sphere
        norms = np.linalg.norm(perturbed_points, axis=1)
        normalized_points = perturbed_points / np.maximum(norms[:, np.newaxis], 1e-12)
        
        # Run Voronoi optimization
        optimized_points = sphere_voronoi_optimization(normalized_points, max_iterations=500)
        
        # Evaluate solution
        ratio = min_max_dist_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # If we failed to find a good solution, return a decent fallback
    if best_points is None:
        np.random.seed(42)
        random_points = np.random.randn(14, 3)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        best_points = random_points / np.maximum(norms, 1e-12)
    
    return best_points

# EVOLVE-BLOCK-END