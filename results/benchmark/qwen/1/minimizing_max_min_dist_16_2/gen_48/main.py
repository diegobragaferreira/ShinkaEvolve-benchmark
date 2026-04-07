# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time
import warnings
from typing import Tuple, Optional


class PointDispersionProblem:
    """Encapsulates the point dispersion optimization problem."""
    
    def __init__(self, n_points: int = 16, dimension: int = 2):
        self.n_points = n_points
        self.dimension = dimension
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # AlphaEvolve benchmark
        
    def compute_distance_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances efficiently using squareform for numerical stability
        try:
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0 or np.isinf(min_dist):
                return 0.0
                
            return min_dist / max_dist
        except Exception:
            return 0.0
    
    def objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function to maximize (returns negative since scipy minimizes)."""
        points = points_flat.reshape(-1, self.dimension)
        return -self.compute_distance_ratio(points)
    
    def constraint_function(self, points_flat: np.ndarray) -> np.ndarray:
        """Constraint function ensuring points stay within [0,1]^d bounds."""
        points = points_flat.reshape(-1, self.dimension)
        
        # Check bounds for each dimension
        violations = []
        
        # x bounds (first dimension)
        violations.append(np.min(points[:, 0]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 0]))  # Should be >= 0
        
        # y bounds (second dimension)  
        violations.append(np.min(points[:, 1]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 1]))  # Should be >= 0
        
        return np.array(violations)


class PointInitializer:
    """Handles different point initialization strategies."""
    
    @staticmethod
    def hexagonal_grid(n_points: int = 16, seed: int = 42) -> np.ndarray:
        """Create hexagonal grid initialization."""
        np.random.seed(seed)
        
        # Create a structured 4x4 grid with offset for hexagonal pattern
        points = []
        rows = cols = 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                
                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                points.append([x, y])
                
        points = np.array(points[:n_points])
        
        # Normalize and scale to [0,1] x [0,1]
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale and shift to fit nicely in unit square
        points[:, 0] *= 0.95
        points[:, 1] *= 0.95
        points[:, 0] += 0.025
        points[:, 1] += 0.025
        
        return np.clip(points, 0, 1)
    
    @staticmethod
    def spiral_pattern(n_points: int = 16, seed: int = 42) -> np.ndarray:
        """Create spiral initialization."""
        np.random.seed(seed)
        
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            radius = 0.4 * (i / n_points)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            points.append([x, y])
            
        return np.clip(np.array(points), 0, 1)
    
    @staticmethod
    def random_points(n_points: int = 16, seed: int = 42) -> np.ndarray:
        """Create purely random initialization."""
        np.random.seed(seed)
        return np.random.rand(n_points, 2)


class Optimizer:
    """Handles the optimization process with multiple strategies."""
    
    def __init__(self, problem: PointDispersionProblem):
        self.problem = problem
        self.bounds = [(0, 1) for _ in range(problem.n_points * problem.dimension)]
    
    def bounded_objective(self, x: np.ndarray) -> float:
        """Objective function with boundary clamping."""
        # Clamp points to valid bounds
        points = np.clip(x.reshape(-1, self.problem.dimension), 0, 1).flatten()
        return self.problem.objective_function(points)
    
    def differential_evolution_optimize(self, x0: np.ndarray, maxiter: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize using differential evolution."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    self.bounded_objective,
                    self.bounds,
                    seed=42,
                    maxiter=maxiter,
                    popsize=15,
                    tol=1e-6,
                    disp=False
                )
                
            if de_result.success:
                return de_result.x, -de_result.fun
        except Exception:
            pass
            
        return x0, self.problem.objective_function(x0)
    
    def local_optimize(self, x0: np.ndarray, maxiter: int = 500) -> Tuple[np.ndarray, float]:
        """Perform local optimization using L-BFGS-B."""
        try:
            result = minimize(
                self.bounded_objective,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                return result.x, -result.fun
        except Exception:
            pass
            
        return x0, self.problem.objective_function(x0)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize problem
    problem = PointDispersionProblem(n_points=16, dimension=2)
    optimizer = Optimizer(problem)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Best solution tracker
    best_points = None
    best_ratio = -np.inf
    
    # Multiple initialization strategies
    init_strategies = [
        PointInitializer.hexagonal_grid,
        PointInitializer.spiral_pattern, 
        PointInitializer.random_points
    ]
    
    # Multi-start optimization
    for strategy in init_strategies:
        for restart in range(3):  # 3 restarts per strategy
            # Generate initial points
            initial_points = strategy(problem.n_points, seed=42 + restart)
            x0 = initial_points.flatten()
            
            # Apply differential evolution for global search
            de_points, de_ratio = optimizer.differential_evolution_optimize(x0, maxiter=100)
            
            # Local refinement
            refined_points, refined_ratio = optimizer.local_optimize(de_points, maxiter=500)
            
            # Update best solution
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.reshape(-1, problem.dimension)
    
    # Final refinement with more iterations if we found a good solution
    if best_points is not None:
        final_points = best_points.flatten()
        final_points, final_ratio = optimizer.local_optimize(final_points, maxiter=1000)
        
        if final_ratio > best_ratio:
            best_points = final_points.reshape(-1, problem.dimension)
    
    # Fallback to hexagonal grid if no good solution found
    if best_points is None:
        initial_points = PointInitializer.hexagonal_grid(problem.n_points, seed=42)
        best_points = initial_points
    
    # Ensure all points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points

# EVOLVE-BLOCK-END