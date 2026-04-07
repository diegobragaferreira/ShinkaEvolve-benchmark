# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time
from typing import Tuple, Optional

class PointOptimizer:
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0
        
    def fibonacci_sphere(self, samples: int = 14) -> np.ndarray:
        """Generate points distributed evenly on a sphere using Fibonacci method"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        # Calculate pairwise distances efficiently
        try:
            distances = pdist(points)
            
            if len(distances) == 0:
                return 0.0
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Avoid division by zero
            if d_max <= 0:
                return 0.0
                
            return d_min / d_max
        except Exception:
            return 0.0
    
    def apply_constraints(self, points: np.ndarray) -> np.ndarray:
        """Ensure points stay within unit cube [0,1]^3"""
        return np.clip(points, 0, 1)
    
    def perturb_point(self, point: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Perturb a single point slightly"""
        noise = np.random.normal(0, delta, 3)
        return point + noise
    
    def initialize_points(self, seed: int) -> np.ndarray:
        """Initialize points using Fibonacci sphere method"""
        np.random.seed(seed)
        
        # Initialize using Fibonacci sphere method
        initial_points = self.fibonacci_sphere(self.num_points)
        
        # Scale and shift to unit cube [0.05, 0.95]^3 to avoid boundary issues
        initial_points = initial_points * 0.9 + 0.05
        
        # Apply constraints to ensure they're within bounds
        return self.apply_constraints(initial_points)
    
    def optimize_single_start(self, seed: int) -> Tuple[np.ndarray, float]:
        """Perform optimization from a single starting configuration"""
        # Initialize points
        points = self.initialize_points(seed)
        
        # Simulated Annealing parameters
        current_ratio = self.calculate_ratio(points)
        best_ratio_local = current_ratio
        best_points_local = points.copy()
        
        # Cooling schedule parameters
        T = 1.0  # Initial temperature
        T_min = 1e-8  # Minimum temperature
        alpha = 0.999  # Cooling rate
        max_iter = 10000  # Max iterations
        iter_without_improvement = 0
        max_no_improvement = 1000  # Early stopping threshold
        
        for iteration in range(max_iter):
            # Perturb one random point
            idx = np.random.randint(0, self.num_points)
            old_point = points[idx].copy()
            
            # Create new candidate point
            new_point = self.perturb_point(old_point, 0.005)
            points[idx] = new_point
            
            # Apply constraints
            points = self.apply_constraints(points)
            
            # Calculate new ratio
            new_ratio = self.calculate_ratio(points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio_local:
                    best_ratio_local = new_ratio
                    best_points_local = points.copy()
                iter_without_improvement = 0
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.rand() < np.exp(delta / T):
                    current_ratio = new_ratio
                    if new_ratio > best_ratio_local:
                        best_ratio_local = new_ratio
                        best_points_local = points.copy()
                    iter_without_improvement = 0
                else:
                    # Revert the change
                    points[idx] = old_point
            
            # Update temperature
            T = max(T * alpha, T_min)
            
            # Early stopping if no improvement
            iter_without_improvement += 1
            if iter_without_improvement > max_no_improvement:
                break
                
        return best_points_local, best_ratio_local
    
    def run_optimization(self) -> np.ndarray:
        """Run multi-start optimization to find best configuration"""
        # Try multiple starting configurations
        seeds = [42, 123, 456, 789]
        
        for seed in seeds:
            try:
                points, ratio = self.optimize_single_start(seed)
                
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = points.copy()
            except Exception:
                continue
        
        # Final validation
        if self.best_points is None:
            # Fallback to random initialization if something went wrong
            np.random.seed(42)
            self.best_points = np.random.rand(self.num_points, self.dimension)
            
        return self.best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(num_points=14, dimension=3)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END