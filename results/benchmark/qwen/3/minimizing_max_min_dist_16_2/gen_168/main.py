# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import time

def compute_voronoi_uniformity(points):
    """Compute uniformity of Voronoi cells."""
    try:
        vor = Voronoi(points)
        areas = []
        for region in vor.regions:
            if not any(v == -1 for v in region) and len(region) >= 3:
                polygon = [vor.vertices[i] for i in region]
                if len(polygon) >= 3:
                    # Calculate area using shoelace formula
                    area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                       polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                       for i in range(len(polygon))))
                    areas.append(area)
        
        if not areas:
            return 0.0
            
        mean_area = np.mean(areas)
        if mean_area == 0:
            return 0.0
            
        std_area = np.std(areas)
        return 1.0 / (1.0 + std_area / mean_area) if mean_area > 0 else 0.0
        
    except:
        return 0.0

def compute_distance_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0.0
    
    # Compute pairwise distances
    dist_matrix = distance.cdist(points, points)
    # Set diagonal to large value to exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)
    
    d_min = np.min(dist_matrix)
    d_max = np.max(dist_matrix)
    
    if d_max <= 0:
        return 0.0
    
    return d_min / d_max

def create_hexagonal_lattice(n_points=16):
    """Create initial hexagonal lattice configuration."""
    # Create points in a hexagonal pattern
    rows = int(np.ceil(np.sqrt(n_points)))
    cols = int(np.ceil(n_points / rows))
    
    points = []
    sqrt3 = np.sqrt(3)
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n_points:
                break
            # Hexagonal offset for alternating rows
            x = j + (i % 2) * 0.5
            y = i * sqrt3 / 2
            points.append([x, y])
            
    points = np.array(points[:n_points])
    
    # Normalize to [0,1] bounds
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
    # Scale to fit nicely in unit square
    points = points * 0.9 + 0.05
    
    # Add subtle perturbations to break symmetry
    np.random.seed(42)
    noise = np.random.normal(0, 0.005, points.shape)
    points += noise
    points = np.clip(points, 0, 1)
    
    return points

def improve_voronoi_uniformity(points, max_iter=200):
    """Iteratively improve Voronoi cell uniformity."""
    current_points = points.copy()
    
    for _ in range(max_iter):
        # Compute current uniformity
        current_uniformity = compute_voronoi_uniformity(current_points)
        
        # Try small perturbations to improve uniformity
        best_points = current_points.copy()
        best_uniformity = current_uniformity
        
        # Try perturbing each point individually
        for i in range(len(current_points)):
            test_points = current_points.copy()
            # Small random perturbation
            delta = np.random.normal(0, 0.001, 2)
            test_points[i] = current_points[i] + delta
            test_points[i] = np.clip(test_points[i], 0, 1)
            
            uniformity = compute_voronoi_uniformity(test_points)
            if uniformity > best_uniformity:
                best_uniformity = uniformity
                best_points = test_points.copy()
        
        # Update if improvement found
        if best_uniformity > current_uniformity:
            current_points = best_points
        else:
            # Reduce perturbation size if no improvement
            break
            
    return current_points

def optimize_with_voronoi_guidance(initial_points, max_iter=1000):
    """Main optimization using Voronoi-based guidance."""
    current_points = initial_points.copy()
    
    # Primary optimization using gradient-based approach
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Combine distance ratio with Voronoi uniformity
        ratio = compute_distance_ratio(points)
        uniformity = compute_voronoi_uniformity(points)
        # Maximize ratio while maintaining good uniformity
        # Return negative since scipy minimizes
        return -(ratio + 0.3 * uniformity)
    
    # Flatten points for optimization
    x0 = current_points.flatten()
    
    # Bounds for coordinates
    bounds = [(0, 1) for _ in range(len(x0))]
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            # Final uniformity improvement
            return improve_voronoi_uniformity(refined_points)
        else:
            # Fall back to simple refinement if optimization fails
            return improve_voronoi_uniformity(current_points)
            
    except Exception:
        # Fallback to simple refinement if optimization fails
        return improve_voronoi_uniformity(current_points)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Start with hexagonal lattice initialization for better geometric properties
    initial_points = create_hexagonal_lattice(16)
    
    # Apply Voronoi-guided optimization
    optimized_points = optimize_with_voronoi_guidance(initial_points, max_iter=500)
    
    return optimized_points

# EVOLVE-BLOCK-END
