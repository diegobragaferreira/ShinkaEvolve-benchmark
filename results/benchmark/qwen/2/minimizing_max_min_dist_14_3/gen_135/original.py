# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time
from typing import Tuple, Optional

class SphericalPointOptimizer:
    """Enhanced optimizer for maximizing min/max distance ratio on unit sphere."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_solution = None
        self.best_ratio = 0.0
        self.start_time = time.time()
    
    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        try:
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 0.0
            
            return min_dist / max_dist
        except Exception:
            return 0.0
    
    def project_to_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere while maintaining relative positions."""
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms[:, np.newaxis]
    
    def fibonacci_sphere_initialization(self, seed: int = 42) -> np.ndarray:
        """Initialize points using enhanced Fibonacci sphere method."""
        np.random.seed(seed)
        points = np.zeros((self.num_points, self.dimension))
        
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(self.num_points):
            # Adjusted Fibonacci approach for better distribution
            y = 1 - (i / float(self.num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            # Refined phi calculation
            phi = (i * golden_ratio) % self.num_points * (2 * math.pi / self.num_points)
            
            # Convert to Cartesian coordinates
            if self.dimension >= 2:
                x = radius * math.cos(phi)
                z = radius * math.sin(phi)
                points[i] = [x, y, z] if self.dimension == 3 else [x, y]
            
            # Apply small random perturbations to break symmetries
            if self.dimension == 3:
                points[i] += np.random.normal(0, 0.05, 3)
            else:
                points[i] += np.random.normal(0, 0.05, 2)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        points = points / safe_norms[:, np.newaxis]
        
        return points
    
    def adaptive_perturbation(self, current_points: np.ndarray, temperature: float) -> Tuple[np.ndarray, int]:
        """Generate adaptive perturbation for one random point."""
        point_idx = np.random.randint(0, self.num_points)
        
        # Adaptive perturbation scale based on temperature
        perturbation_scale = temperature * 0.01
        
        # Create perturbation vector
        perturbation = np.random.normal(0, perturbation_scale, self.dimension)
        
        # Apply perturbation
        new_points = current_points.copy()
        new_points[point_idx] += perturbation
        
        return new_points, point_idx
    
    def adaptive_cooling(self, temperature: float, stagnation_count: int, 
                        max_stagnation: int = 500) -> float:
        """Apply adaptive cooling strategy."""
        if stagnation_count > max_stagnation and temperature > 1e-8:
            # Faster cooling when stagnating
            return temperature * 0.95
        return temperature * 0.9995
    
    def optimize_single_run(self, initial_points: np.ndarray, 
                           max_iterations: int = 100000) -> Tuple[np.ndarray, float]:
        """Perform single optimization run with adaptive strategies."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = self.compute_min_max_ratio(current_points)
        
        temperature = 1.0
        min_temperature = 1e-8
        stagnation_count = 0
        last_improvement = 0
        
        # Track recent ratios for adaptive learning
        recent_ratios = []
        
        for iteration in range(max_iterations):
            # Check for termination conditions
            if temperature < min_temperature:
                break
                
            # Apply adaptive cooling
            temperature = self.adaptive_cooling(temperature, stagnation_count)
            
            # Generate perturbation
            test_points, _ = self.adaptive_perturbation(current_points, temperature)
            
            # Project back to sphere
            test_points = self.project_to_unit_sphere(test_points)
            
            # Compute new ratio
            test_ratio = self.compute_min_max_ratio(test_points)
            
            # Metropolis acceptance criterion
            if test_ratio > best_ratio or np.random.random() < math.exp((test_ratio - best_ratio) / temperature):
                current_points = test_points.copy()
                if test_ratio > best_ratio:
                    best_ratio = test_ratio
                    best_points = test_points.copy()
                    last_improvement = iteration
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1
            
            # Track recent ratios
            recent_ratios.append(test_ratio)
            if len(recent_ratios) > 20:
                recent_ratios.pop(0)
            
            # Progress reporting
            if iteration % 10000 == 0 and iteration > 0:
                elapsed = time.time() - self.start_time
                if elapsed > 360:  # Time limit reached
                    break
        
        return best_points, best_ratio
    
    def multi_start_optimization(self, max_runs: int = 4) -> Tuple[np.ndarray, float]:
        """Run multiple optimization starts with different seeds."""
        best_global_points = None
        best_global_ratio = 0.0
        
        seeds = [42, 123, 456, 789]
        
        for seed in seeds[:max_runs]:
            try:
                # Initialize with Fibonacci method
                initial_points = self.fibonacci_sphere_initialization(seed)
                
                # Optimize
                points, ratio = self.optimize_single_run(initial_points)
                
                if ratio > best_global_ratio:
                    best_global_ratio = ratio
                    best_global_points = points.copy()
                    
            except Exception as e:
                # Continue with other seeds if one fails
                continue
        
        return best_global_points, best_global_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Initialize optimizer
    optimizer = SphericalPointOptimizer(num_points=14, dimension=3)
    
    # Run multi-start optimization
    points, ratio = optimizer.multi_start_optimization(max_runs=4)
    
    # Fallback to initial configuration if nothing worked
    if points is None:
        points = optimizer.fibonacci_sphere_initialization(42)
    
    return points

# EVOLVE-BLOCK-END