# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random
from typing import Tuple, Optional

class PointGenerator:
    """Handles generation and initialization of points on a sphere."""
    
    @staticmethod
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate n points on a sphere using Fibonacci spiral method."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    @classmethod
    def initialize_points(cls, n: int, seed: int = 42) -> np.ndarray:
        """Initialize points using Fibonacci sphere and perturbation."""
        # Start with Fibonacci sphere points
        points = cls.fibonacci_sphere(n)

        # Add some randomness to avoid perfect symmetry issues
        np.random.seed(seed)
        points += 0.01 * np.random.randn(n, 3)

        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        return points

class Evaluation:
    """Handles evaluation of point configurations."""
    
    @staticmethod
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

class Optimizer:
    """Handles the optimization process using simulated annealing."""
    
    def __init__(self, max_iterations: int = 10000, initial_temp: float = 0.1, 
                 min_temp: float = 1e-6, cooling_rate: float = 0.999):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.patience = 100
        
    def perturb_point(self, point: np.ndarray, step_size: float) -> np.ndarray:
        """Perturb a single point on the unit sphere."""
        # Generate random perturbation
        delta = np.random.randn(3)
        # Project to tangent plane and normalize
        delta = delta - np.dot(delta, point) * point
        delta = delta / np.linalg.norm(delta)

        # Apply perturbation
        new_point = point + step_size * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)

        return new_point
    
    def adaptive_cooling(self, temperature: float, iteration: int, 
                        last_improvement: int, improvement_threshold: float = 1e-6) -> float:
        """Apply adaptive cooling based on improvement trends."""
        if iteration - last_improvement > self.patience:
            return max(self.min_temp, temperature * 0.95)
        return max(self.min_temp, temperature * self.cooling_rate)
    
    def optimize(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run simulated annealing optimization."""
        # Initialize
        points = initial_points.copy()
        current_ratio = Evaluation.compute_min_max_ratio(points)
        
        T = self.initial_temp
        step_size = 0.01
        best_points = points.copy()
        best_ratio = current_ratio
        last_improvement = 0
        iteration = 0
        
        # Vectorized operations for efficiency
        all_indices = list(range(len(points)))
        
        while iteration < self.max_iterations:
            # Adaptive cooling
            T = self.adaptive_cooling(T, iteration, last_improvement)
            
            # Select random point to perturb
            idx = random.choice(all_indices)
            
            # Save current point
            old_point = points[idx].copy()
            
            # Perturb selected point
            points[idx] = self.perturb_point(points[idx], step_size)
            
            # Compute new ratio
            new_ratio = Evaluation.compute_min_max_ratio(points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()
                    last_improvement = iteration
            else:
                # Calculate acceptance probability
                delta = new_ratio - current_ratio
                acceptance_prob = math.exp(delta / T)
                
                if random.random() < acceptance_prob:
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = points.copy()
                        last_improvement = iteration
                else:
                    # Revert change
                    points[idx] = old_point
            
            # Periodic local refinement
            if iteration % 500 == 0 and iteration > 0:
                refined_points, refined_ratio = self.local_refinement(points.copy())
                if refined_ratio > current_ratio:
                    points = refined_points.copy()
                    current_ratio = refined_ratio
                    if refined_ratio > best_ratio:
                        best_ratio = refined_ratio
                        best_points = points.copy()
                        last_improvement = iteration
            
            # Reduce step size over time
            step_size = max(0.001, step_size * 0.9999)
            
            # Early stopping if we're not improving
            if iteration - last_improvement > 5000:
                break
                
            iteration += 1
        
        return best_points, best_ratio
    
    def local_refinement(self, points: np.ndarray, iterations: int = 10, 
                        step_size: float = 0.1) -> Tuple[np.ndarray, float]:
        """Perform local refinement using gradient estimation."""
        refined_points = points.copy()
        for _ in range(iterations):
            improved = False
            for i in range(len(refined_points)):
                old_ratio = Evaluation.compute_min_max_ratio(refined_points)
                old_point = refined_points[i].copy()
                
                # Estimate gradient using finite differences
                grad = np.zeros(3)
                for j in range(3):
                    eps = 1e-4
                    test_points = refined_points.copy()
                    test_points[i, j] += eps
                    test_points[i] = test_points[i] / np.linalg.norm(test_points[i])
                    new_ratio = Evaluation.compute_min_max_ratio(test_points)
                    grad[j] = (new_ratio - old_ratio) / eps
                
                # Move along gradient
                if np.linalg.norm(grad) > 1e-10:
                    refined_points[i] = refined_points[i] + step_size * grad
                    refined_points[i] = refined_points[i] / np.linalg.norm(refined_points[i])
                    improved = True
            
            # If no improvement, stop early
            if not improved:
                break
        
        new_ratio = Evaluation.compute_min_max_ratio(refined_points)
        return refined_points, new_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize components
    generator = PointGenerator()
    optimizer = Optimizer()
    
    # Generate initial points
    initial_points = generator.initialize_points(14)
    
    # Optimize using simulated annealing
    best_points, _ = optimizer.optimize(initial_points)
    
    return best_points

# EVOLVE-BLOCK-END