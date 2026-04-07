# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import math
import random
import time
from typing import Tuple, Optional

class PointGenerator:
    """Handles generation and initialization of points on a sphere."""
    
    @staticmethod
    def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
        """Generate n points on a sphere using Fibonacci spiral method."""
        np.random.seed(seed)
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
        points = cls.fibonacci_sphere(n, seed)

        # Add some randomness to avoid perfect symmetry issues
        np.random.seed(seed)
        points += 0.01 * np.random.randn(n, 3)

        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        return points

class Evaluation:
    """Handles evaluation of point configurations."""
    
    @staticmethod
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float, float]:
        """Compute the ratio of minimum to maximum distances and return all values."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0.0, d_min, d_max
            
        ratio = d_min / d_max
        return ratio, d_min, d_max

class Optimizer:
    """Handles the optimization process using simulated annealing with enhancements."""
    
    def __init__(self, max_iterations: int = 15000, initial_temp: float = 0.1, 
                 min_temp: float = 1e-8, cooling_rate: float = 0.9995):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.patience = 500
        
    def perturb_point_guided(self, points: np.ndarray, idx: int, 
                           current_ratio: float, temp: float) -> np.ndarray:
        """Perturb a point based on its local geometric context."""
        try:
            # Get distances to all other points
            distances = cdist([points[idx]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance
            
            if len(distances) == 0:
                # Fallback to random perturbation
                delta = np.random.randn(3)
                delta = delta - np.dot(delta, points[idx]) * points[idx]
                return points[idx] + 0.02 * temp * delta / np.linalg.norm(delta)
            
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            
            # Analyze the point's local geometry
            # If point is too close to others, push it away
            if min_distance < avg_distance * 0.4:
                # Compute repulsion force from close neighbors
                repulsion = np.zeros(3)
                for i in range(len(points)):
                    if i != idx:
                        diff = points[idx] - points[i]
                        dist = np.linalg.norm(diff)
                        if dist > 0 and dist < avg_distance * 0.7:
                            # Inverse distance weighted repulsion
                            repulsion += diff / dist * (1.0 / dist**2)
                
                # If there's repulsion, normalize and apply
                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    # Magnitude depends on how close it is
                    magnitude = 0.05 * (1.0 - min_distance / avg_distance) * temp
                    return points[idx] + repulsion * magnitude
                
            elif max_distance > avg_distance * 1.5:
                # If point is far from others, maybe pull it closer to balance
                # Compute attraction toward average position
                attraction = np.mean(points, axis=0) - points[idx]
                attraction_norm = np.linalg.norm(attraction)
                if attraction_norm > 0:
                    attraction = attraction / attraction_norm
                    # Magnitude inversely related to distance from center
                    center_distance = np.linalg.norm(points[idx])
                    magnitude = 0.01 * (1.0 - center_distance * 0.5) * temp
                    return points[idx] - attraction * magnitude
                    
            # Default: small random perturbation guided by temperature
            delta = np.random.randn(3)
            delta = delta - np.dot(delta, points[idx]) * points[idx]
            magnitude = 0.02 * temp
            return points[idx] + magnitude * delta / np.linalg.norm(delta)
            
        except Exception:
            # Fallback to simple random perturbation
            delta = np.random.randn(3)
            delta = delta - np.dot(delta, points[idx]) * points[idx]
            return points[idx] + 0.02 * temp * delta / np.linalg.norm(delta)
    
    def adaptive_cooling(self, temperature: float, iteration: int, 
                        last_improvement: int, recent_ratios: list) -> float:
        """Apply adaptive cooling based on recent performance trends."""
        # Check for stagnation
        if len(recent_ratios) >= 5:
            recent_change = recent_ratios[-1] - recent_ratios[0]
            if recent_change < 1e-8:
                # Very slow progress, cool faster
                return max(self.min_temp, temperature * 0.98)
            elif recent_change > 1e-5:
                # Fast progress, cool slower but not in final phase
                return max(self.min_temp, temperature * 0.995)
        
        # Regular cooling with dynamic adjustment
        if iteration - last_improvement > self.patience:
            # Slow progress detected, increase cooling rate
            return max(self.min_temp, temperature * 0.95)
        
        # Normal cooling rate
        return max(self.min_temp, temperature * self.cooling_rate)
    
    def local_refinement(self, points: np.ndarray, max_iter: int = 20, 
                        step_size: float = 0.05) -> Tuple[np.ndarray, float]:
        """Perform local refinement using gradient estimation."""
        refined_points = points.copy()
        current_ratio, _, _ = Evaluation.compute_min_max_ratio(refined_points)
        
        for _ in range(max_iter):
            improved = False
            for i in range(len(refined_points)):
                old_ratio, _, _ = Evaluation.compute_min_max_ratio(refined_points)
                old_point = refined_points[i].copy()
                
                # Estimate gradient using finite differences
                grad = np.zeros(3)
                for j in range(3):
                    eps = 1e-5
                    test_points = refined_points.copy()
                    test_points[i, j] += eps
                    # Project back to sphere
                    norm = np.linalg.norm(test_points[i])
                    if norm > 0:
                        test_points[i] = test_points[i] / norm
                    new_ratio, _, _ = Evaluation.compute_min_max_ratio(test_points)
                    grad[j] = (new_ratio - old_ratio) / eps
                
                # Move along gradient
                if np.linalg.norm(grad) > 1e-10:
                    refined_points[i] = refined_points[i] + step_size * grad
                    # Project back to sphere
                    norm = np.linalg.norm(refined_points[i])
                    if norm > 0:
                        refined_points[i] = refined_points[i] / norm
                    improved = True
            
            # If no improvement, stop early
            if not improved:
                break
        
        new_ratio, _, _ = Evaluation.compute_min_max_ratio(refined_points)
        return refined_points, new_ratio
    
    def optimize_single(self, initial_points: np.ndarray, 
                       max_iter_override: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Run simulated annealing optimization for one instance."""
        # Initialize
        points = initial_points.copy()
        current_ratio, _, _ = Evaluation.compute_min_max_ratio(points)
        
        T = self.initial_temp
        best_points = points.copy()
        best_ratio = current_ratio
        last_improvement = 0
        iteration = 0
        recent_ratios = []
        max_iter = max_iter_override if max_iter_override is not None else self.max_iterations
        
        # Vectorized operations for efficiency
        all_indices = list(range(len(points)))
        
        while iteration < max_iter:
            # Adaptive cooling
            T = self.adaptive_cooling(T, iteration, last_improvement, recent_ratios)
            
            # Select random point to perturb
            idx = random.choice(all_indices)
            
            # Save current point
            old_point = points[idx].copy()
            
            # Perturb selected point using guided mechanism
            points[idx] = self.perturb_point_guided(points, idx, current_ratio, T)
            
            # Compute new ratio
            new_ratio, _, _ = Evaluation.compute_min_max_ratio(points)
            
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
                if T > 0:
                    acceptance_prob = math.exp(delta / T)
                else:
                    acceptance_prob = 0.0
                    
                if random.random() < acceptance_prob:
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = points.copy()
                        last_improvement = iteration
                else:
                    # Revert change
                    points[idx] = old_point
            
            # Track recent ratios for adaptive cooling
            recent_ratios.append(current_ratio)
            if len(recent_ratios) > 30:
                recent_ratios.pop(0)
            
            # Periodic local refinement
            if iteration % 1000 == 0 and iteration > 0:
                refined_points, refined_ratio = self.local_refinement(points.copy())
                if refined_ratio > current_ratio:
                    points = refined_points.copy()
                    current_ratio = refined_ratio
                    if refined_ratio > best_ratio:
                        best_ratio = refined_ratio
                        best_points = points.copy()
                        last_improvement = iteration
            
            # Early stopping if we're not improving
            if iteration - last_improvement > 10000:
                break
                
            iteration += 1
        
        return best_points, best_ratio
    
    def optimize_multi_start(self, initial_points: np.ndarray, 
                           num_starts: int = 5) -> Tuple[np.ndarray, float]:
        """Run multiple optimization instances to find better solutions."""
        best_points = None
        best_ratio = 0.0
        
        for start in range(num_starts):
            # Use different random seed for each start
            np.random.seed(start * 1000 + 42)
            
            # Create fresh initial points for this run
            if start == 0:
                # First start with provided points
                points = initial_points.copy()
            else:
                # Generate new initialization with different seed
                points = PointGenerator.initialize_points(14, start * 1000 + 42)
            
            # Run optimization for this start
            current_points, current_ratio = self.optimize_single(points, max_iter_override=10000)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
        
        # Run final refinement on best solution found
        if best_points is not None:
            final_points, final_ratio = self.local_refinement(best_points.copy(), max_iter=50)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
        
        return best_points, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize components
    generator = PointGenerator()
    optimizer = Optimizer()
    
    # Generate initial points using Fibonacci sphere
    initial_points = generator.initialize_points(14)
    
    # Optimize using multi-start simulated annealing
    best_points, _ = optimizer.optimize_multi_start(initial_points)
    
    return best_points

# EVOLVE-BLOCK-END