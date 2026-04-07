# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_voronoi_min_area_ratio(points):
        """Compute ratio using Voronoi cell areas as a measure of dispersion quality"""
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            
            # Calculate areas of finite Voronoi cells
            areas = []
            for i, region in enumerate(vor.point_region):
                if region != -1:  # Skip infinite regions
                    vertices = vor.vertices[vor.regions[region]]
                    if len(vertices) >= 3:  # Need at least 3 vertices for a polygon
                        # Compute area of polygon using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
            
            if not areas:
                return 0.0
                
            min_area = np.min(areas)
            max_area = np.max(areas)
            
            if max_area == 0:
                return 0.0
                
            return min_area / max_area
            
        except:
            # Fallback to distance-based ratio if Voronoi computation fails
            distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=2))
            np.fill_diagonal(distances, np.inf)
            if distances.size > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    return min_dist / max_dist
            return 0.0
    
    def compute_min_max_ratio(points):
        """Original distance-based ratio computation"""
        distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        if distances.size > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                return min_dist / max_dist
        return 0.0
    
    def energy_objective_function(x_flat):
        """Energy-based objective function using inverse distance repulsion"""
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(distances, 1e-10)  # Avoid division by zero
        
        # Compute total repulsive energy (inverse distance)
        # Sum of 1/d_ij for all pairs (i,j) where i<j
        energy = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                energy += 1.0 / distances[i, j]
        
        # Penalize boundary violations
        boundary_penalty = 0
        for point in points:
            for coord in point:
                if coord <= 0.01 or coord >= 0.99:
                    boundary_penalty += 1000.0
        
        return -energy + boundary_penalty * 1000.0
    
    def voronoi_objective_function(x_flat):
        """Voronoi-based objective function focused on cell area uniformity"""
        points = x_flat.reshape(-1, 2)
        
        # Apply boundary constraints via reflection
        points = np.clip(points, 0, 1)
        points = np.clip(points, 0, 1)
        
        # Compute Voronoi-based quality
        area_ratio = compute_voronoi_min_area_ratio(points)
        
        # Add distance-based penalty for very small minimum distances
        distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        if distances.size > 0:
            min_dist = np.min(distances)
            if min_dist < 0.05:  # Penalize very small distances
                area_ratio -= (0.05 - min_dist) * 1000.0
        
        return -area_ratio
    
    def hexagonal_grid_init():
        """Create a proper hexagonal lattice arrangement"""
        points = []
        
        # Parameters for hexagonal lattice
        hex_spacing = 1.0
        row_spacing = hex_spacing * np.sqrt(3) / 2.0
        col_spacing = hex_spacing

        # Place points in hexagonal pattern (4 rows, 4 columns)
        for row in range(4):
            for col in range(4):
                if len(points) >= 16:
                    break
                x = col * col_spacing
                if row % 2 == 1:
                    x += col_spacing / 2.0
                y = row * row_spacing
                points.append([x, y])

        # Convert to numpy array and normalize
        points = np.array(points[:16])
        
        # Normalize to fit within unit square
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        if max_x > min_x and max_y > min_y:
            scale_x = 1.0 / (max_x - min_x)
            scale_y = 1.0 / (max_y - min_y)
            scale = min(scale_x, scale_y, 1.0)
            
            points[:, 0] = (points[:, 0] - min_x) * scale
            points[:, 1] = (points[:, 1] - min_y) * scale

        # Center the points
        center_shift = 0.5 - np.mean(points, axis=0)
        points = points + center_shift

        # Ensure bounds
        points = np.clip(points, 0, 1)

        # Apply randomness to break symmetries
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.005, points.shape)
        points += perturbations
        points = np.clip(points, 0, 1)
        
        return points
    
    def initialize_points():
        """Initialize points with multiple strategies"""
        # Strategy 1: Hexagonal grid
        points = hexagonal_grid_init()
        
        # Strategy 2: Random with some clustering avoidance
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        
        # Strategy 3: Perturbed hexagonal
        perturbed_points = points.copy()
        np.random.seed(42)
        perturbed_points += np.random.normal(0, 0.02, points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Evaluate which initialization performs better
        orig_ratio = compute_min_max_ratio(points)
        random_ratio = compute_min_max_ratio(random_points)
        perturbed_ratio = compute_min_max_ratio(perturbed_points)
        
        # Choose the best initialization
        if random_ratio > orig_ratio and random_ratio > perturbed_ratio:
            return random_points
        elif perturbed_ratio > orig_ratio:
            return perturbed_points
        else:
            return points
    
    # Main optimization loop with multiple restarts
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple initialization strategies
    num_restarts = 10
    for restart in range(num_restarts):
        # Initialize points
        points = initialize_points()
        
        # Apply different optimization approaches
        x0 = points.flatten()
        
        # Approach 1: Voronoi-based optimization
        try:
            # Use bounds for constrained optimization
            bounds = [(0, 1) for _ in range(32)]
            
            # Try different optimization methods
            result = minimize(
                voronoi_objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                current_ratio = compute_min_max_ratio(optimized_points)
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()
        except:
            pass
            
        # Approach 2: Energy-based optimization as backup
        if best_points is None:
            try:
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    energy_objective_function,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    current_ratio = compute_min_max_ratio(optimized_points)
                    
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = optimized_points.copy()
            except:
                pass
    
    # Final fallback to hexagonal grid if nothing worked
    if best_points is None:
        points = hexagonal_grid_init()
        return points
    
    return best_points

# EVOLVE-BLOCK-END