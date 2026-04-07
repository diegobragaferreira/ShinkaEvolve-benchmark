# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple, List, Optional, Callable
import warnings

class PointInitializationStrategy:
    """Base class for point initialization strategies."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        raise NotImplementedError

class FibonacciSphereInitializer(PointInitializationStrategy):
    """Initialize points using Fibonacci sphere method."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        points_array = np.array(points)
        # Scale to unit cube [0,1]^3
        points_array = (points_array + 1) / 2  
        return points_array

class SphericalCodeInitializer(PointInitializationStrategy):
    """Initialize points using known spherical code configurations."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
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
        
        # Scale to unit cube [0,1]^3
        spherical_points = (spherical_points + 1) / 2
        return spherical_points

class RandomInitializer(PointInitializationStrategy):
    """Initialize points randomly."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        np.random.seed(42)
        return np.random.rand(n_points, d)

class PerturbedFibonacciInitializer(PointInitializationStrategy):
    """Initialize with perturbed Fibonacci points."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        fib_initializer = FibonacciSphereInitializer()
        points = fib_initializer.initialize(n_points, d)
        # Add small perturbation
        np.random.seed(42)
        points += np.random.normal(0, 0.005, points.shape)
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

class PerturbedSphericalInitializer(PointInitializationStrategy):
    """Initialize with perturbed spherical code points."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        sph_initializer = SphericalCodeInitializer()
        points = sph_initializer.initialize(n_points, d)
        # Add small perturbation
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

class RelaxedRandomInitializer(PointInitializationStrategy):
    """Initialize with geometric relaxation on random points."""
    
    def initialize(self, n_points: int, d: int = 3) -> np.ndarray:
        np.random.seed(42)
        points = np.random.rand(n_points, d)
        # Apply relaxation
        relaxation_steps = 25
        for _ in range(relaxation_steps):
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
            
            # Apply forces and project back to cube
            points += 0.005 * forces  # Smaller step size for more stable convergence
            points = np.clip(points, 0, 1)
        return points

class PointInitializer:
    """Handles various point initialization strategies."""
    
    def __init__(self):
        self.strategies = [
            FibonacciSphereInitializer(),
            SphericalCodeInitializer(), 
            RandomInitializer(),
            PerturbedFibonacciInitializer(),
            PerturbedSphericalInitializer(),
            RelaxedRandomInitializer()
        ]
    
    def initialize_best(self, n_points: int = 14, d: int = 3, num_starts: int = 6) -> np.ndarray:
        """Initialize points using multiple strategies and return the best."""
        best_points = None
        best_ratio = -float('inf')
        
        for start_idx, strategy in enumerate(self.strategies[:num_starts]):
            try:
                points = strategy.initialize(n_points, d)
                # Calculate initial ratio
                distances = pdist(points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = points.copy()
            except Exception:
                continue
        
        # Fallback to random if nothing worked
        if best_points is None:
            np.random.seed(42)
            best_points = np.random.rand(n_points, d)
        
        return best_points

class GeometricRelaxer:
    """Handles geometric relaxation of point sets."""
    
    @staticmethod
    def relax_points(points: np.ndarray, iterations: int = 20) -> np.ndarray:
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

class DistanceCalculator:
    """Handles distance calculations and metrics."""
    
    @staticmethod
    def calculate_metrics(points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate minimum, maximum, and ratio of distances between all point pairs."""
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0, 0.0, 0.0
        
        ratio = min_dist / max_dist
        return min_dist, max_dist, ratio

class ObjectiveFunction:
    """Handles objective function calculation for optimization."""
    
    def __init__(self, n_points: int = 14, d: int = 3):
        self.n_points = n_points
        self.d = d
    
    def calculate(self, points_flat: np.ndarray) -> float:
        """Calculate objective function value (negative ratio)."""
        points = points_flat.reshape(self.n_points, self.d)
        
        # Penalty for out-of-bounds points
        penalty = 0.0
        for i in range(self.n_points):
            for j in range(self.d):  # x, y, z coordinates
                if points[i,j] < 0:
                    penalty += 1000 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1000 * (points[i,j] - 1)**2
        
        # Calculate distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf') + penalty
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf') + penalty
        
        # Return negative ratio plus penalty
        return -min_dist / max_dist + penalty

class SymmetryGenerator:
    """Generates symmetric variants of point configurations."""
    
    @staticmethod
    def generate_variants(points: np.ndarray, num_variants: int = 4) -> np.ndarray:
        """Create symmetric variants by rotating around different axes."""
        variants = [points]
        
        # Create rotations around different axes
        for i in range(num_variants):
            angle = 2 * np.pi * (i + 1) / (num_variants + 1)
            
            # Rotation around z-axis
            rot_z = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            rotated = points @ rot_z.T
            variants.append(rotated)
            
            # Rotation around x-axis
            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            
            rotated_x = points @ rot_x.T
            variants.append(rotated_x)
            
            # Rotation around y-axis
            rot_y = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            
            rotated_y = points @ rot_y.T
            variants.append(rotated_y)
        
        return np.vstack(variants)

class AdaptiveOptimizer:
    """Performs adaptive optimization using multiple strategies."""
    
    def __init__(self, n_points: int = 14, d: int = 3):
        self.objective_func = ObjectiveFunction(n_points, d)
        self.n_points = n_points
        self.d = d
    
    def optimize(self, initial_points: np.ndarray) -> np.ndarray:
        """Perform adaptive optimization with changing population sizes and strategies."""
        start_time = time.time()
        
        # Flatten initial points for optimization
        initial_flat = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0.0, 1.0)] * len(initial_flat)
        
        # Progressive optimization phases
        current_result = None
        
        # Phase 1: Global search with large population
        try:
            result = differential_evolution(
                self.objective_func.calculate,
                bounds,
                maxiter=200,
                popsize=25,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
            current_result = result
        except Exception:
            # Fallback to initial points if first phase fails
            return initial_points
        
        # Phase 2: Refinement with smaller population
        try:
            result = differential_evolution(
                self.objective_func.calculate,
                bounds,
                maxiter=250,
                popsize=15,
                tol=1e-7,
                mutation=(0.7, 1.0),
                recombination=0.8,
                seed=43,
                disp=False
            )
            current_result = result
        except Exception:
            pass
        
        # Phase 3: Local refinement using L-BFGS-B with adaptive tolerance tightening
        def local_objective(x_flat):
            points = x_flat.reshape(-1, self.d)
            # Add penalty for out-of-bounds
            penalty = 0.0
            for i in range(self.n_points):
                for j in range(self.d):
                    if points[i,j] < 0:
                        penalty += 1000 * (0 - points[i,j])**2
                    elif points[i,j] > 1:
                        penalty += 1000 * (points[i,j] - 1)**2
                        
            distances = pdist(points)
            if len(distances) == 0:
                return float('inf') + penalty
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 0:
                return float('inf') + penalty
            return -min_dist / max_dist + penalty
        
        try:
            if current_result is not None:
                x0 = current_result.x.reshape(-1, self.d).flatten()
                
                # Adaptive tolerance tightening for L-BFGS-B
                # Start with looser tolerances for faster initial convergence
                looser_options = {'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
                tighter_options = {'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
                
                # First run with looser tolerances
                local_result = minimize(
                    local_objective,
                    x0,
                    method='L-BFGS-B',
                    options=looser_options
                )
                
                # If improvement was significant, try with tighter tolerances
                if (current_result is not None and 
                    abs(local_result.fun - current_result.fun) > 1e-8):
                    local_result = minimize(
                        local_objective,
                        local_result.x,
                        method='L-BFGS-B',
                        options=tighter_options
                    )
                
                current_result = local_result
        except Exception:
            pass
        
        # Reshape optimized result
        if current_result is not None:
            optimized_points = current_result.x.reshape(self.n_points, self.d)
        else:
            optimized_points = initial_points.copy()
        
        # Ensure all points are within valid range (final safety check)
        optimized_points = np.clip(optimized_points, 0, 1)
        
        # Explore symmetric variants to find potentially better solutions
        try:
            variants = SymmetryGenerator.generate_variants(optimized_points, num_variants=3)
            _, _, best_ratio = DistanceCalculator.calculate_metrics(optimized_points)
            
            # Evaluate all variants
            for i in range(len(variants) // self.n_points):
                variant_points = variants[i*self.n_points:(i+1)*self.n_points]
                _, _, ratio = DistanceCalculator.calculate_metrics(variant_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    optimized_points = variant_points.copy()
        except Exception:
            pass
        
        return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initialize points with multiple strategies
    initializer = PointInitializer()
    initial_points = initializer.initialize_best(14, 3, num_starts=6)
    
    # Phase 2: Apply geometric relaxation for better distribution
    relaxer = GeometricRelaxer()
    relaxed_points = relaxer.relax_points(initial_points, iterations=30)
    
    # Phase 3: Optimize points with adaptive strategy
    optimizer = AdaptiveOptimizer(14, 3)
    optimized_points = optimizer.optimize(relaxed_points)
    
    # Phase 4: Final validation and adjustment
    final_points = optimized_points.copy()
    
    # Calculate final metrics
    min_dist, max_dist, ratio = DistanceCalculator.calculate_metrics(final_points)
    
    # If optimization didn't work well, try final global optimization
    if max_dist <= 0 or min_dist <= 0 or ratio < 0.15:
        # Try one more optimization pass with better parameters if needed
        try:
            # Reconstruct objective for this fallback
            objective_func = ObjectiveFunction(14, 3)
            bounds = [(0.0, 1.0)] * (14 * 3)
            result = differential_evolution(
                objective_func.calculate,
                bounds,
                maxiter=400,
                popsize=30,
                tol=1e-10,
                mutation=(0.8, 1.0),
                recombination=0.9,
                seed=42,
                disp=False
            )
            final_points = result.x.reshape(14, 3)
            final_points = np.clip(final_points, 0, 1)
        except Exception:
            pass
    
    # Final validation
    _, _, final_ratio = DistanceCalculator.calculate_metrics(final_points)
    if final_ratio < 0.05:  # Very poor result, use another fallback
        np.random.seed(42)
        final_points = np.random.rand(14, 3)
    
    return final_points

# EVOLVE-BLOCK-END