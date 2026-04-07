# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple


class PointOptimizer:
    """Handles the optimization of point configurations in 3D space."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.total_variables = num_points * dimension
        
    def initialize_points(self) -> np.ndarray:
        """Initialize points using spherical packing heuristic."""
        # Generate points using Fibonacci spiral on sphere
        points = []
        for i in range(self.num_points):
            phi = np.arccos(1 - 2 * (i / (self.num_points - 1)))
            theta = np.sqrt(self.num_points) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        
        # Convert to numpy array and scale to unit cube
        points_array = np.array(points)
        # Normalize to unit sphere first
        norms = np.linalg.norm(points_array, axis=1, keepdims=True)
        points_array = points_array / np.max(norms)
        # Then scale and shift to [0,1]^3
        points_array = points_array * 0.5 + 0.5
        
        # Add small random perturbation
        np.random.seed(42)
        points_array += np.random.normal(0, 0.01, points_array.shape)
        
        return np.clip(points_array, 0, 1)
    
    def compute_distances(self, points: np.ndarray) -> Tuple[float, float]:
        """Compute min and max distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        
        # Handle edge case where all points are identical
        if len(distances) == 0:
            return 0.0, 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return 0.0, 0.0
            
        return min_dist, max_dist
    
    def objective_function(self, x_flat: np.ndarray) -> float:
        """Objective function to maximize min/max distance ratio."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, self.dimension)
        
        # Compute distances
        min_dist, max_dist = self.compute_distances(points)
        
        # Return negative ratio to minimize (since we want to maximize ratio)
        if max_dist <= 0:
            return 0.0
            
        return -min_dist / max_dist
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        # Initialize points
        initial_points = self.initialize_points()
        x0 = initial_points.flatten()
        
        # Define bounds for each coordinate [0,1]
        bounds = [(0, 1)] * self.total_variables
        
        # Run optimization with L-BFGS-B
        start_time = time.time()
        
        # First pass with L-BFGS-B
        result = minimize(
            self.objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        end_time = time.time()
        
        # Extract optimized points
        optimized_points = result.x.reshape(-1, self.dimension)
        
        # Ensure points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)
        
        return optimized_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(num_points=14, dimension=3)
    return optimizer.optimize()


# EVOLVE-BLOCK-END