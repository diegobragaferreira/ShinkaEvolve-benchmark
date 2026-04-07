# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from typing import Tuple, Optional
import time

class PointArrangementOptimizer:
    """Optimizes point arrangements to maximize min/max distance ratio."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_score = -np.inf
        self.best_points = None
        
    def fibonacci_spiral_on_sphere(self, n: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method."""
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
    
    def calculate_distances(self, points: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Calculate distance matrix and extract min/max distances."""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return distances, min_dist, max_dist
    
    def objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function to maximize min/max distance ratio."""
        points = points_flat.reshape(-1, 3)
        _, min_dist, max_dist = self.calculate_distances(points)
        
        if max_dist == 0:
            return -np.inf
            
        ratio = min_dist / max_dist
        return ratio
    
    def constraint_sphere(self, points_flat: np.ndarray) -> float:
        """Constraint function to keep points on unit sphere."""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return np.mean((norms - 1.0)**2)  # Mean squared deviation from unit sphere
    
    def optimize_with_lbfgsb(self, initial_points: np.ndarray) -> np.ndarray:
        """Optimize using L-BFGS-B algorithm."""
        initial_flat = initial_points.flatten()
        
        # Define bounds for coordinates (-1, 1) for each coordinate
        bounds = [(-1, 1) for _ in range(self.num_points * self.dimension)]
        
        # Optimization parameters
        options = {'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        
        # Run optimization
        result = minimize(
            lambda x: -self.objective_function(x),  # Minimize negative to maximize
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-9
        )
        
        # Extract optimized points
        final_points = result.x.reshape(-1, 3)
        
        # Ensure points are normalized to unit sphere
        norms = np.linalg.norm(final_points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        final_points = final_points / safe_norms
        
        return final_points
    
    def validate_and_score(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Validate solution and compute performance metrics."""
        distances, min_dist, max_dist = self.calculate_distances(points)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
            
        benchmark_ratio = ratio / 0.4898
            
        return ratio, benchmark_ratio, max_dist
    
    def optimize(self) -> Tuple[np.ndarray, dict]:
        """Main optimization loop."""
        # Phase 1: Initialization
        initial_points = self.fibonacci_spiral_on_sphere(self.num_points)
        
        # Phase 2: Optimization
        optimized_points = self.optimize_with_lbfgsb(initial_points)
        
        # Phase 3: Validation and Scoring
        min_max_ratio, benchmark_ratio, max_dist = self.validate_and_score(optimized_points)
        
        stats = {
            'min_max_ratio': min_max_ratio,
            'benchmark_ratio': benchmark_ratio,
            'max_distance': max_dist,
            'eval_time': 0.0
        }
        
        return optimized_points, stats

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create optimizer instance
    optimizer = PointArrangementOptimizer(num_points=14, dimension=3)
    
    # Perform optimization
    points, _ = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END