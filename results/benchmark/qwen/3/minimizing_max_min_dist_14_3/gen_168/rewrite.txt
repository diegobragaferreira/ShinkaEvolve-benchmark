# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple, Optional, List
import warnings

class PointInitializer:
    """Handles various point initialization strategies for 3D point layouts."""
    
    @staticmethod
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    @staticmethod
    def spherical_code_initialization(n_points: int = 14) -> np.ndarray:
        """Create initial point configuration based on spherical code principles."""
        # Known good configuration for 14 points on sphere from literature
        spherical_points = np.array([
            [0.0000, 0.0000, 1.0000],
            [0.0000, 0.0000, -1.0000],
            [0.9343, 0.0000, 0.3564],
            [-0.9343, 0.0000, 0.3564],
            [0.0000, 0.9343, 0.3564],
            [0.0000, -0.9343, 0.3564],
            [0.0000, 0.9343, -0.3564],
            [0.0000, -0.9343, -0.3564],
            [0.9343, 0.0000, -0.3564],
            [-0.9343, 0.0000, -0.3564],
            [0.3564, 0.9343, 0.0000],
            [-0.3564, 0.9343, 0.0000],
            [0.3564, -0.9343, 0.0000],
            [-0.3564, -0.9343, 0.0000]
        ])
        
        # Normalize to unit sphere if needed
        norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
        spherical_points = spherical_points / np.where(norms == 0, 1, norms)
        
        # Add small perturbations to escape local optima
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.01, spherical_points.shape)
        spherical_points = spherical_points + perturbation
        
        # Normalize again after perturbation
        norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
        spherical_points = spherical_points / np.where(norms == 0, 1, norms)
        
        return spherical_points
    
    @classmethod
    def initialize_multiple_strategies(cls, n_points: int = 14, num_starts: int = 6) -> Tuple[np.ndarray, float]:
        """Initialize points using multiple strategies and return the best configuration."""
        strategies = [
            cls._fibonacci_strategy,
            cls._spherical_code_strategy,
            cls._random_strategy,
            cls._perturbed_fibonacci_strategy,
            cls._perturbed_spherical_strategy,
            cls._relaxed_random_strategy
        ]
        
        best_points = None
        best_ratio = -float('inf')
        
        for start_idx, strategy in enumerate(strategies):
            try:
                points = strategy(n_points, start_idx)
                if points is not None:
                    ratio = cls._calculate_ratio(points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
            except Exception as e:
                warnings.warn(f"Strategy {start_idx} failed: {str(e)}")
                continue
        
        # Fallback to random if nothing worked
        if best_points is None:
            np.random.seed(42)
            best_points = np.random.rand(n_points, 3)
            best_ratio = cls._calculate_ratio(best_points)
            
        return best_points, best_ratio
    
    @classmethod
    def _fibonacci_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Fibonacci sphere initialization strategy."""
        points = cls.fibonacci_sphere(n_points)
        points = (points + 1) / 2  # Map from [-1,1] to [0,1]
        return points
    
    @classmethod
    def _spherical_code_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Spherical code initialization strategy."""
        points = cls.spherical_code_initialization(n_points)
        points = (points + 1) / 2  # Map from [-1,1] to [0,1]
        return points
    
    @classmethod
    def _random_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Random initialization strategy."""
        np.random.seed(42 + seed_offset)
        return np.random.rand(n_points, 3)
    
    @classmethod
    def _perturbed_fibonacci_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Perturbed Fibonacci strategy."""
        points = cls.fibonacci_sphere(n_points)
        points = (points + 1) / 2
        np.random.seed(42 + seed_offset)
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    @classmethod
    def _perturbed_spherical_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Perturbed spherical code strategy."""
        points = cls.spherical_code_initialization(n_points)
        points = (points + 1) / 2
        np.random.seed(42 + seed_offset)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    @classmethod
    def _relaxed_random_strategy(cls, n_points: int, seed_offset: int = 0) -> np.ndarray:
        """Geometric relaxation on random points strategy."""
        np.random.seed(42 + seed_offset)
        points = np.random.rand(n_points, 3)
        return cls._geometric_relaxation_step(points, iterations=25)
    
    @staticmethod
    def _geometric_relaxation_step(points: np.ndarray, iterations: int = 20) -> np.ndarray:
        """Apply geometric relaxation using force-based repulsion model."""
        points = points.copy()
        
        for _ in range(iterations):
            # Calculate pairwise distances
            n = len(points)
            forces = np.zeros_like(points)
            
            # Compute repulsive forces between all pairs
            for i in range(n):
                for j in range(i+1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    
                    # Avoid singularity
                    if dist_sq > 1e-10:
                        force_magnitude = 1.0 / dist_sq
                        forces[i] += force_magnitude * diff
                        forces[j] -= force_magnitude * diff
            
            # Apply forces and project back to unit cube
            points += 0.005 * forces  # Smaller step size for more stable convergence
            points = np.clip(points, 0, 1)
        
        return points
    
    @staticmethod
    def _calculate_ratio(points: np.ndarray) -> float:
        """Calculate the min/max distance ratio for a given layout."""
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
        
        return min_dist / max_dist

class PointOptimizer:
    """Handles the optimization of point layouts to maximize min/max distance ratio."""
    
    def __init__(self, n_points: int = 14):
        self.n_points = n_points
        self.best_layout = None
        self.best_ratio = 0.0
        
    def _objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function to maximize the min/max distance ratio."""
        points = points_flat.reshape(self.n_points, 3)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Calculate distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf')
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf')
        
        # Return negative ratio to minimize (maximize the ratio)
        return -min_dist / max_dist
    
    def _adaptive_optimization(self, initial_points: np.ndarray) -> np.ndarray:
        """Perform adaptive optimization with changing population sizes."""
        # Flatten initial points for optimization
        initial_flat = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0.0, 1.0)] * len(initial_flat)
        
        try:
            # Phase 1: Global search with large population
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=200,
                popsize=25,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            # Phase 2: Refinement with smaller population
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=250,
                popsize=15,
                tol=1e-7,
                mutation=(0.7, 1.0),
                recombination=0.8,
                seed=43,
                disp=False
            )
            
            # Phase 3: Local refinement using L-BFGS-B
            def local_objective(x_flat):
                points = x_flat.reshape(-1, 3)
                points = np.clip(points, 0, 1)
                distances = pdist(points)
                if len(distances) == 0:
                    return float('inf')
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist <= 0:
                    return float('inf')
                return -min_dist / max_dist
            
            x0 = result.x.reshape(-1, 3).flatten()
            local_result = minimize(
                local_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            result = local_result
            
        except Exception as e:
            # Fallback to initial points if optimization fails
            warnings.warn(f"Optimization failed: {str(e)}")
            return initial_points
            
        # Reshape optimized result
        optimized_points = result.x.reshape(self.n_points, 3)
        
        # Ensure all points are within valid range
        optimized_points = np.clip(optimized_points, 0, 1)
        
        return optimized_points
    
    def optimize(self, initial_points: np.ndarray) -> np.ndarray:
        """Main optimization process."""
        # Apply geometric relaxation to improve initial distribution
        relaxed_points = PointInitializer._geometric_relaxation_step(initial_points, iterations=30)
        
        # Optimize with adaptive strategy
        final_layout = self._adaptive_optimization(relaxed_points)
        
        # Final validation
        final_ratio = PointInitializer._calculate_ratio(final_layout)
        
        # Additional optimization if needed
        if final_ratio < 0.15:
            try:
                bounds = [(0.0, 1.0)] * (self.n_points * 3)
                result = differential_evolution(
                    self._objective_function,
                    bounds,
                    maxiter=400,
                    popsize=30,
                    tol=1e-10,
                    mutation=(0.8, 1.0),
                    recombination=0.9,
                    seed=42,
                    disp=False
                )
                final_layout = result.x.reshape(self.n_points, 3)
                final_layout = np.clip(final_layout, 0, 1)
                final_ratio = PointInitializer._calculate_ratio(final_layout)
            except Exception as e:
                warnings.warn(f"Final optimization failed: {str(e)}")
                pass
        
        # Final validation check
        if final_ratio < 0.05:
            np.random.seed(42)
            final_layout = np.random.rand(self.n_points, 3)
            
        return final_layout

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize points using multiple strategies
    initial_points, _ = PointInitializer.initialize_multiple_strategies(14, num_starts=6)
    
    # Optimize the layout
    optimizer = PointOptimizer(14)
    result = optimizer.optimize(initial_points)
    
    return result

# EVOLVE-BLOCK-END