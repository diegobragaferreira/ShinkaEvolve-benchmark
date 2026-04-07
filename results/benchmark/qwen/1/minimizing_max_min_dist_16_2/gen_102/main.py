# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time
import math
from numba import jit

@jit(nopython=True)
def calculate_min_max_ratio_fast(distances):
    """Fast calculation of min/max ratio using numba-compiled function"""
    if len(distances) == 0:
        return 0.0
    
    min_dist = distances[0]
    max_dist = distances[0]
    
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
        if distances[i] > max_dist:
            max_dist = distances[i]
    
    if max_dist <= 0:
        return 0.0
    
    return min_dist / max_dist

class SphericalVoronoiEvolutionOptimizer:
    """Spherical Voronoi-based optimizer for point distribution maximizing min/max distance ratio."""
    
    def __init__(self, n_points: int = 16, dimensions: int = 2):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # 0.2786
        self.max_time = 180.0  # seconds
        
    def initialize_fibonacci_sphere(self) -> np.ndarray:
        """Initialize points using Fibonacci sphere algorithm for optimal 3D distribution."""
        points_3d = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points_3d.append([x, y, z])
        
        return np.array(points_3d)
    
    def stereographic_project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        points_2d = np.array(points_2d)
        
        # Normalize to unit square
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d
    
    def initialize_spherical_projection(self) -> np.ndarray:
        """Initialize points using spherical arrangement projected to 2D."""
        points_3d = self.initialize_fibonacci_sphere()
        return self.stereographic_project(points_3d)
    
    def calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        return calculate_min_max_ratio_fast(distances)
    
    def voronoi_refinement(self, points: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Improve point distribution using iterative Voronoi refinement."""
        current_points = points.copy()
        
        for _ in range(iterations):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)
                
                # Compute centroids of Voronoi cells
                new_points = []
                for i in range(len(current_points)):
                    region = vor.regions[vor.point_region[i]]
                    if not region or -1 in region:
                        # Keep original point if region is invalid
                        new_points.append(current_points[i])
                        continue
                    
                    # Find vertices of Voronoi cell
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) > 0:
                        # Compute centroid of cell vertices
                        vertices = np.array(vertices)
                        centroid = np.mean(vertices, axis=0)
                        new_points.append(centroid)
                    else:
                        # Keep original point if no vertices
                        new_points.append(current_points[i])
                
                # Update points
                current_points = np.array(new_points)
                
                # Keep within bounds
                current_points = np.clip(current_points, 0, 1)
                
            except Exception:
                # If Voronoi computation fails, return original points
                break
        
        return current_points
    
    def local_optimization(self, points: np.ndarray, max_iter: int = 300) -> np.ndarray:
        """Apply local optimization using L-BFGS-B."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            distances = pdist(points_candidate)
            return -calculate_min_max_ratio_fast(distances)
        
        try:
            # Use L-BFGS-B for local refinement with appropriate bounds
            bounds = [(0, 1) for _ in range(len(points.flatten()))]
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                return np.clip(optimized_points, 0, 1)
        except Exception:
            pass
            
        return points
    
    def hybrid_optimization(self, initial_points: np.ndarray) -> np.ndarray:
        """Apply hybrid optimization combining different strategies."""
        current_points = initial_points.copy()
        
        # Stage 1: Voronoi refinement for global improvement
        voronoi_points = self.voronoi_refinement(current_points, iterations=3)
        
        # Stage 2: Local optimization on Voronoi result
        local_points = self.local_optimization(voronoi_points)
        
        # Stage 3: Additional Voronoi refinement
        final_points = self.voronoi_refinement(local_points, iterations=2)
        
        # Stage 4: Final local optimization
        final_points = self.local_optimization(final_points, max_iter=200)
        
        return final_points
    
    def optimize(self) -> np.ndarray:
        """Main optimization process using spherical Voronoi approach."""
        # Initialize using spherical projection
        initial_points = self.initialize_spherical_projection()
        
        # Apply hybrid optimization
        optimized_points = self.hybrid_optimization(initial_points)
        
        # Additional refinement if needed
        refined_points = self.voronoi_refinement(optimized_points, iterations=5)
        refined_points = self.local_optimization(refined_points, max_iter=150)
        
        return refined_points
    
    def evolve(self) -> np.ndarray:
        """Main evolutionary optimization loop."""
        return self.optimize()

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = SphericalVoronoiEvolutionOptimizer(n_points=16, dimensions=2)
    return optimizer.evolve()

# EVOLVE-BLOCK-END
