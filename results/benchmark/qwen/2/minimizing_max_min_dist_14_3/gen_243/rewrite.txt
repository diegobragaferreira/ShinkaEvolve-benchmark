# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import math
import random
import time
from typing import Tuple, Optional, List

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
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]

        return points

class Evaluator:
    """Handles evaluation of point configurations and distance computations."""
    
    @staticmethod
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float]:
        """Compute the ratio of minimum to maximum distances efficiently."""
        if len(points) < 2:
            return 0.0, 0.0
        
        # Use pdist for efficient pairwise distance calculation
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0.0, 0.0
        
        return d_min / d_max, d_min

class Optimizer:
    """Handles the optimization process with enhanced strategies."""
    
    def __init__(self, max_iterations: int = 20000, initial_temp: float = 1.0, 
                 min_temp: float = 1e-12, cooling_rate: float = 0.9995):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.patience = 1000
        self.improvement_window = 50
        
    def _compute_repulsion_force(self, points: np.ndarray, idx: int, 
                               avg_distance: float, min_distance: float) -> np.ndarray:
        """Compute repulsion force from close neighbors."""
        repulsion = np.zeros(3)
        threshold = avg_distance * 0.7
        
        for i in range(len(points)):
            if i != idx:
                diff = points[idx] - points[i]
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < threshold:
                    # Inverse distance weighted repulsion
                    repulsion += diff / dist * (1.0 / dist**2)
        
        return repulsion
    
    def _compute_attract_force(self, points: np.ndarray, idx: int, 
                             avg_distance: float, max_distance: float) -> np.ndarray:
        """Compute attraction force toward average position."""
        attraction = np.mean(points, axis=0) - points[idx]
        attraction_norm = np.linalg.norm(attraction)
        
        if attraction_norm > 0:
            attraction = attraction / attraction_norm
            return attraction
        return np.zeros(3)
    
    def _perturb_point_guided(self, points: np.ndarray, idx: int, 
                            current_ratio: float, temp: float, 
                            iteration: int = 0) -> np.ndarray:
        """Enhanced perturb point with geometric guidance."""
        try:
            # Get distances to all other points
            distances = cdist([points[idx]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance
            
            if len(distances) == 0:
                # Fallback to random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return points[idx] + 0.02 * temp * direction
                
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            
            # Analyze the point's local geometry
            if min_distance < avg_distance * 0.4:
                # Compute repulsion force from close neighbors
                repulsion = self._compute_repulsion_force(points, idx, avg_distance, min_distance)
                
                # If there's repulsion, normalize and apply
                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    # Magnitude depends on how close it is
                    magnitude = 0.05 * (1.0 - min_distance / avg_distance) * temp
                    return points[idx] + repulsion * magnitude
                else:
                    # Fallback to random perturbation
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    return points[idx] + 0.02 * temp * direction
                    
            elif max_distance > avg_distance * 1.5:
                # Compute attraction toward average position
                attraction = self._compute_attract_force(points, idx, avg_distance, max_distance)
                attraction_norm = np.linalg.norm(attraction)
                if attraction_norm > 0:
                    # Magnitude inversely related to distance from center
                    center_distance = np.linalg.norm(points[idx])
                    magnitude = 0.01 * (1.0 - center_distance * 0.5) * temp
                    return points[idx] - attraction * magnitude
                else:
                    # Random perturbation
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    return points[idx] + 0.01 * temp * direction
                    
            else:
                # Moderate distances - small adjustment
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                # Magnitude based on how balanced the distances are
                balance_score = abs(min_distance - avg_distance) / avg_distance + \
                               abs(max_distance - avg_distance) / avg_distance
                magnitude = 0.01 * (1.0 - balance_score * 0.5) * temp
                return points[idx] + magnitude * direction
                
        except Exception:
            # Fallback to simple random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            return points[idx] + 0.01 * temp * direction
    
    def _adaptive_cooling(self, temperature: float, iteration: int,
                         last_improvement: int, recent_ratios: List[float]) -> float:
        """Apply adaptive cooling based on performance trends."""
        # Check recent improvement
        if len(recent_ratios) >= self.improvement_window:
            recent_change = recent_ratios[-1] - recent_ratios[-self.improvement_window]
            if recent_change < 1e-10:
                # Very slow progress, increase cooling rate
                return max(self.min_temp, temperature * 0.98)
            elif recent_change > 1e-6:
                # Fast progress, decrease cooling rate
                return max(self.min_temp, temperature * 0.995)
        
        # Check for stagnation
        if iteration - last_improvement > self.patience:
            # No improvement for a while, increase cooling rate
            return max(self.min_temp, temperature * 0.95)
            
        # Normal cooling
        return max(self.min_temp, temperature * self.cooling_rate)
    
    def _optimize_single(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run simulated annealing optimization for one instance."""
        points = initial_points.copy()
        current_ratio, _ = Evaluator.compute_min_max_ratio(points)
        
        T = self.initial_temp
        best_points = points.copy()
        best_ratio = current_ratio
        last_improvement = 0
        iteration = 0
        recent_ratios = []
        
        # Main optimization loop
        while iteration < self.max_iterations and T > self.min_temp:
            # Adaptive cooling
            T = self._adaptive_cooling(T, iteration, last_improvement, recent_ratios)
            
            # Select random point to perturb
            point_to_move = random.randint(0, len(points) - 1)
            
            # Create new candidate point with guided perturbation
            new_points = points.copy()
            new_points[point_to_move] = self._perturb_point_guided(
                points, point_to_move, current_ratio, T, iteration)
            
            # Project back onto sphere
            norm = np.linalg.norm(new_points[point_to_move])
            if norm > 0:
                new_points[point_to_move] = new_points[point_to_move] / norm
            
            # Compute new ratio
            new_ratio, _ = Evaluator.compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                points = new_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()
                    last_improvement = iteration
            else:
                # Revert change
                pass  # points unchanged
            
            # Track recent ratios for adaptive cooling
            recent_ratios.append(current_ratio)
            if len(recent_ratios) > 100:
                recent_ratios.pop(0)
                
            iteration += 1
            
            # Early stopping
            if iteration - last_improvement > self.patience * 2:
                break
        
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
            current_points, current_ratio = self._optimize_single(points)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
        
        return best_points, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Initialize components
    generator = PointGenerator()
    optimizer = Optimizer()
    
    # Generate initial points using Fibonacci sphere
    initial_points = generator.initialize_points(14)
    
    # Optimize using multi-start simulated annealing
    best_points, _ = optimizer.optimize_multi_start(initial_points)
    
    return best_points

# EVOLVE-BLOCK-END