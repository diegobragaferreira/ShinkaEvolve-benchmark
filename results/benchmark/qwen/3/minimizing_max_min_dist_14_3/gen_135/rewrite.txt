# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import warnings
import time
from typing import Tuple, List, Optional, Union

class DistanceCalculator:
    """Efficiently calculates and caches pairwise distances"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        """Calculate and cache pairwise distances"""
        # Use a hashable representation for caching
        key = str(points.shape) + str(points.dtype) + str(points.tobytes()[:100])
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        self.cache_misses += 1
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        self.cache[key] = distances
        return distances
    
    def clear_cache(self):
        """Clear the distance cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

class PointValidator:
    """Validates and normalizes point configurations"""
    
    @staticmethod
    def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere"""
        if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
            return points
            
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]
    
    @staticmethod
    def clamp_to_bounds(points: np.ndarray, bounds: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """Clamp points to valid bounds"""
        if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
            return points
        return np.clip(points, bounds[0], bounds[1])

class ConfigurationGenerator:
    """Generates diverse initial point configurations"""
    
    @staticmethod
    def fibonacci_sphere_points(n: int) -> np.ndarray:
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.arccos(y)  # angle from z-axis
            phi = (i * 2 * np.pi) / golden_ratio  # azimuthal angle
            
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    @classmethod
    def generate_configurations(cls, n_points: int = 14) -> List[np.ndarray]:
        """Generate multiple high-quality initial configurations"""
        configs = []
        
        # Method 1: Fibonacci sphere distribution
        fib_points = cls.fibonacci_sphere_points(n_points)
        configs.append(fib_points)
        
        # Method 2: Random points normalized to sphere
        np.random.seed(42)
        rand_points = np.random.randn(n_points, 3)
        configs.append(rand_points)
        
        # Method 3: Slightly perturbed Fibonacci
        perturbed_fib = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        configs.append(perturbed_fib)
        
        # Method 4: Random points in unit cube then projected to sphere
        np.random.seed(123)
        cube_points = np.random.rand(n_points, 3)
        configs.append(cube_points)
        
        # Method 5: Perturbed random points
        np.random.seed(456)
        perturbed_rand = np.random.rand(n_points, 3) * 2 - 1
        configs.append(perturbed_rand)
        
        return configs

class OptimizationEngine:
    """Handles the optimization with multiple strategies"""
    
    def __init__(self, distance_calc: DistanceCalculator, validator: PointValidator):
        self.distance_calc = distance_calc
        self.validator = validator
        self.max_iterations = 500
        self.tolerance = 1e-10
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function: minimize negative ratio of min/max distances"""
        points = x.reshape(-1, 3)
        
        # Normalize to unit sphere
        points_normalized = self.validator.normalize_to_unit_sphere(points)
        
        # Calculate distances
        distances = self.distance_calc.calculate_pairwise_distances(points_normalized)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max < 1e-12:
            return 1e12  # Penalty for invalid configurations
            
        # Return negative ratio (we minimize to maximize ratio)
        return -d_min / d_max
    
    def constraint_sphere(self, x: np.ndarray) -> np.ndarray:
        """Constraint to keep points on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def optimize_single(self, initial_points: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Perform single optimization with L-BFGS-B"""
        try:
            x0 = initial_points.flatten()
            
            # Define constraints for unit sphere
            cons = [
                {'type': 'eq', 'fun': self.constraint_sphere}
            ]
            
            # Use L-BFGS-B for local refinement with strict tolerances
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': self.tolerance, 'gtol': self.tolerance}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                optimized_points = self.validator.normalize_to_unit_sphere(optimized_points)
                
                # Calculate final ratio
                distances = self.distance_calc.calculate_pairwise_distances(optimized_points)
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    return optimized_points, ratio
                    
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
        
        # Return input if optimization fails
        return initial_points, 0.0
    
    def optimize_with_restarts(self, initial_configs: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Run optimization with multiple restarts and selection"""
        best_points = None
        best_ratio = -np.inf
        
        for i, config in enumerate(initial_configs):
            try:
                # Apply slight perturbation for diversity
                if i > 0:
                    np.random.seed(i * 42)
                    perturbation = np.random.normal(0, 0.01, config.shape)
                    config = config + perturbation
                    config = self.validator.clamp_to_bounds(config)
                
                # Optimize this configuration
                optimized_points, ratio = self.optimize_single(config, self.max_iterations)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Configuration {i} optimization failed: {str(e)}")
                continue
        
        # Return best solution or fallback
        if best_points is not None:
            return best_points, best_ratio
        
        # Fallback to first configuration
        return initial_configs[0], 0.0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Initialize components
    distance_calc = DistanceCalculator()
    validator = PointValidator()
    config_generator = ConfigurationGenerator()
    optimizer = OptimizationEngine(distance_calc, validator)
    
    # Generate diverse initial configurations
    initial_configs = config_generator.generate_configurations(14)
    
    # Perform optimization with multiple restarts
    final_points, final_ratio = optimizer.optimize_with_restarts(initial_configs)
    
    # Final validation and normalization
    final_points = validator.normalize_to_unit_sphere(final_points)
    final_points = validator.clamp_to_bounds(final_points)
    
    # Clear cache to free memory
    distance_calc.clear_cache()
    
    return final_points

# EVOLVE-BLOCK-END