# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    def fibonacci_sphere(n):
        """Generate n points on a unit sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def compute_voronoi_score(points):
        """Compute score based on Voronoi cell uniformity"""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            # Variance of areas - lower variance means more uniform cells
            return np.var(areas)
        except:
            return np.inf
    
    def initialize_spherical_points(n):
        """Initialize points with improved spherical distribution"""
        # Start with Fibonacci spiral
        points = fibonacci_sphere(n)
        
        # Apply iterative refinement to improve uniformity
        for _ in range(10):
            # Compute Voronoi score to check uniformity
            voronoi_score = compute_voronoi_score(points)
            
            # If not sufficiently uniform, adjust points
            if voronoi_score > 0.1:  # Threshold for uniformity
                # Simple adjustment: move points to reduce variance
                try:
                    sv = SphericalVoronoi(points)
                    areas = sv.voronoi_cell_areas()
                    
                    # Adjust points to make areas more uniform
                    adjustments = np.zeros_like(points)
                    for i in range(len(points)):
                        if areas[i] > np.mean(areas) * 1.1:  # Overlarge cell
                            # Move away from neighbors with large cells
                            neighbor_indices = sv._get_voronoi_regions()[i]
                            if len(neighbor_indices) > 0:
                                adjustments[i] = np.mean(points[list(neighbor_indices)], axis=0)
                                adjustments[i] = adjustments[i] - points[i]
                                adjustments[i] = adjustments[i] * 0.1  # Small adjustment
                                
                    # Apply adjustments
                    points = points + adjustments
                    # Renormalize to unit sphere
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    points = points / norms
                    
                except:
                    pass
            
        return points
    
    def voronoi_guided_optimization(initial_points, max_iter=300):
        """Optimize using Voronoi-guided approach"""
        points = initial_points.copy()
        
        # Convert to unit sphere if not already
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        
        best_points = points.copy()
        best_ratio = 0
        
        # Multi-phase optimization with different strategies
        for phase in range(3):
            if phase == 0:
                # Phase 1: Coarse optimization with larger steps
                iter_count = max_iter // 3
                step_size = 0.05
                tolerance = 1e-4
            elif phase == 1:
                # Phase 2: Medium optimization
                iter_count = max_iter // 3
                step_size = 0.01
                tolerance = 1e-6
            else:
                # Phase 3: Fine optimization
                iter_count = max_iter // 3
                step_size = 0.001
                tolerance = 1e-8
            
            # Iterative refinement
            for iteration in range(iter_count):
                # Compute current distances and ratio
                distances = cdist(points, points)
                np.fill_diagonal(distances, np.inf)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    current_ratio = d_min / d_max
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = points.copy()
                
                # Simple gradient-based update using Voronoi information
                try:
                    sv = SphericalVoronoi(points)
                    areas = sv.voronoi_cell_areas()
                    
                    # Compute forces based on Voronoi cell uniformity
                    forces = np.zeros_like(points)
                    for i in range(len(points)):
                        # Repulsion force from overlarge cells
                        if areas[i] > np.mean(areas) * 1.1:
                            # Find neighbors with smaller cells and repel toward them
                            neighbor_indices = list(sv._get_voronoi_regions()[i])
                            if len(neighbor_indices) > 0:
                                avg_neighbor_area = np.mean([areas[j] for j in neighbor_indices])
                                if avg_neighbor_area < np.mean(areas) * 0.9:
                                    # Move towards neighbors with smaller cells
                                    avg_neighbor_point = np.mean(points[neighbor_indices], axis=0)
                                    direction = avg_neighbor_point - points[i]
                                    norm_dir = np.linalg.norm(direction)
                                    if norm_dir > 1e-10:
                                        forces[i] = direction / norm_dir * step_size
                                
                    # Apply forces and renormalize
                    points = points + forces
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    points = points / norms
                    
                except Exception:
                    # Fallback: simple gradient ascent
                    try:
                        # Simple gradient update
                        distances = cdist(points, points)
                        np.fill_diagonal(distances, np.inf)
                        
                        # For each point, compute gradient based on nearest neighbors
                        gradients = np.zeros_like(points)
                        for i in range(len(points)):
                            # Get distances to other points
                            dist_vec = distances[i]
                            # Find closest point
                            closest_idx = np.argmin(dist_vec)
                            if closest_idx != i:
                                closest_point = points[closest_idx]
                                diff = points[i] - closest_point
                                norm_diff = np.linalg.norm(diff)
                                if norm_diff > 1e-10:
                                    gradients[i] = diff / norm_diff * step_size * 0.1
                        
                        points = points + gradients
                        norms = np.linalg.norm(points, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        points = points / norms
                        
                    except:
                        pass
                        
                # Early stopping based on improvement
                if iteration > 10 and iteration % 20 == 0:
                    # Check if improvement is minimal
                    distances = cdist(points, points)
                    np.fill_diagonal(distances, np.inf)
                    d_min_new = np.min(distances)
                    d_max_new = np.max(distances)
                    if d_max_new > 0:
                        new_ratio = d_min_new / d_max_new
                        if abs(new_ratio - current_ratio) < tolerance:
                            break
        
        return best_points, best_ratio
    
    # Generate initial points with improved spherical distribution
    initial_points = initialize_spherical_points(14)
    
    # Convert to unit cube [0,1]^3
    # First center around origin and scale appropriately
    centered = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(centered))
    if max_coord > 0:
        scaled = centered / max_coord * 0.5
    else:
        scaled = centered
    # Then shift to [0,1]^3
    final_points = scaled + 0.5
    
    # Apply Voronoi-guided optimization
    optimized_points, ratio = voronoi_guided_optimization(final_points)
    
    # Ensure final points are in [0,1]^3
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

# EVOLVE-BLOCK-END