# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple, Optional
import warnings

class SphericalVoronoiInitializer:
    """Initializes points using spherical Voronoi diagram principles and energy minimization."""
    
    @staticmethod
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Initialize points using Fibonacci spiral on sphere for better even distribution."""
        points = []
        for i in range(n):
            phi = np.arccos(1 - 2 * (i / (n - 1)))
            theta = np.sqrt(n) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    @staticmethod
    def energy_minimization_initialization(n: int, d: int = 3, max_iter: int = 1000) -> np.ndarray:
        """Initialize points using energy minimization approach."""
        # Start with Fibonacci distribution on unit sphere
        points = SphericalVoronoiInitializer.fibonacci_sphere(n)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    @classmethod
    def initialize_points(cls, n: int = 14, d: int = 3, seed: int = 42) -> np.ndarray:
        """Initialize points using energy-based spherical approach."""
        np.random.seed(seed)
        
        # Use energy minimization approach
        points = cls.energy_minimization_initialization(n, d)
        
        # Scale to unit cube [0,1]^3
        # First map to [-1,1]^3 then scale to [0,1]^3
        points = points * 0.5 + 0.5
        
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        return points

class EnergyBasedDistanceCalculator:
    """Handles distance computations using energy-based metrics and optimizations."""
    
    @staticmethod
    def compute_min_max_distances(points: np.ndarray) -> Tuple[float, float]:
        """Compute minimum and maximum distances between all point pairs efficiently."""
        if len(points) < 2:
            return 0.0, 0.0
            
        try:
            # Use scipy's distance functions for efficiency
            distances = cdist(points, points, 'euclidean')
            
            # Zero out diagonal (distance to self)
            np.fill_diagonal(distances, np.inf)
            
            if distances.size == 0:
                return 0.0, 0.0
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Avoid division by zero
            if max_dist <= 0:
                return 0.0, 0.0
                
            return min_dist, max_dist
        except Exception:
            return 0.0, 0.0

class EnergyOptimizationStrategy:
    """Manages energy-based optimization approaches using Voronoi principles."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.total_variables = num_points * dimension
    
    def _energy_penalty_objective(self, x_flat: np.ndarray) -> float:
        """Energy-based objective function with penalty for constraint violations."""
        points = x_flat.reshape(-1, self.dimension)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Calculate distances
        min_dist, max_dist = EnergyBasedDistanceCalculator.compute_min_max_distances(points)
        
        # Energy-based penalty for constraint violations (boundary effects)
        penalty = 0.0
        for coord in points.flat:
            if coord < 0:
                penalty += 1000 * (0 - coord)**2
            elif coord > 1:
                penalty += 1000 * (coord - 1)**2
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf') + penalty
            
        # Energy-based objective: maximize the ratio (min/max) 
        # We minimize the negative ratio to maximize it
        ratio = -min_dist / max_dist
        return ratio + penalty
    
    def _voronoi_based_local_search(self, initial_points: np.ndarray, 
                                  max_time: float = 300.0) -> np.ndarray:
        """Perform local search using Voronoi-based energy minimization."""
        x0 = initial_points.flatten()
        bounds = [(0, 1)] * self.total_variables
        
        # Use Nelder-Mead for initial local search (faster for this problem)
        try:
            result = minimize(
                self._energy_penalty_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True}
            )
            return result.x.reshape(-1, self.dimension)
        except Exception:
            return initial_points.reshape(-1, self.dimension)
    
    def _hybrid_global_optimization(self, initial_points: np.ndarray,
                                  max_time: float = 300.0) -> np.ndarray:
        """Hybrid global optimization combining multiple strategies."""
        # Start with our best initialization
        current_points = initial_points.copy()
        
        # Try multiple local searches with different starting points
        best_points = current_points.copy()
        best_ratio = -float('inf')
        
        # Perform several local optimizations with different random restarts
        for restart in range(3):
            # Randomly perturb the current solution slightly
            if restart > 0:
                np.random.seed(42 + restart)
                perturbation = np.random.normal(0, 0.01, current_points.shape)
                temp_points = current_points + perturbation
                temp_points = np.clip(temp_points, 0, 1)
            else:
                temp_points = current_points.copy()
            
            # Local optimization
            local_result = self._voronoi_based_local_search(temp_points)
            
            # Evaluate the result
            min_dist, max_dist = EnergyBasedDistanceCalculator.compute_min_max_distances(local_result)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = local_result.copy()
        
        return best_points

class PointOptimizer:
    """Main optimization class using energy-based spherical Voronoi approach."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.initializer = SphericalVoronoiInitializer()
        self.optimizer = EnergyOptimizationStrategy(num_points, dimension)
        self.evaluator = EnergyBasedDistanceCalculator()
    
    def validate_and_correct(self, points: np.ndarray) -> np.ndarray:
        """Ensure points are valid and within bounds."""
        if points.shape != (self.num_points, self.dimension):
            raise ValueError("Invalid point array shape")
            
        # Clip to valid range
        points = np.clip(points, 0, 1)
        
        # Handle degenerate cases
        if np.allclose(points[0], points):
            # All points are identical, regenerate
            points = self.initializer.initialize_points(self.num_points, self.dimension)
            
        return points
    
    def evaluate_solution(self, points: np.ndarray) -> float:
        """Evaluate solution quality using energy-based metrics."""
        min_dist, max_dist = self.evaluator.compute_min_max_distances(points)
        
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine using energy-based spherical Voronoi approach."""
        # Phase 1: Initialize points using spherical energy-based method
        try:
            initial_points = self.initializer.initialize_points(
                self.num_points, self.dimension
            )
        except Exception:
            # Fallback to random initialization
            np.random.seed(42)
            initial_points = np.random.rand(self.num_points, self.dimension)
        
        # Phase 2: Hybrid optimization with multiple restarts
        try:
            optimized_points = self.optimizer._hybrid_global_optimization(
                initial_points
            )
        except Exception:
            optimized_points = initial_points
        
        # Phase 3: Final validation and correction
        final_points = self.validate_and_correct(optimized_points)
        
        # Phase 4: Quality check and potential fallback
        ratio = self.evaluate_solution(final_points)
        
        # If quality is poor, fallback to initialization
        if ratio < 1e-6:
            try:
                fallback_points = self.initializer.initialize_points(
                    self.num_points, self.dimension
                )
                fallback_ratio = self.evaluate_solution(fallback_points)
                if fallback_ratio > ratio:
                    final_points = fallback_points
            except Exception:
                pass
        
        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    try:
        optimizer = PointOptimizer(num_points=14, dimension=3)
        return optimizer.optimize()
    except Exception:
        # Last resort: random initialization
        np.random.seed(42)
        return np.random.rand(14, 3)

# EVOLVE-BLOCK-END