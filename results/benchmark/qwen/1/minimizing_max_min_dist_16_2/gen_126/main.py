# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple, List
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_distance_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Use squareform for numerical stability
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
    
    def objective_function(points_flat: np.ndarray) -> float:
        """Objective function to maximize (negative since scipy minimizes)."""
        points = points_flat.reshape(-1, 2)
        return -compute_distance_ratio(points)
    
    def constraint_function(points_flat: np.ndarray) -> float:
        """Constraint function ensuring points stay within [0,1]^2 bounds."""
        points = points_flat.reshape(-1, 2)
        
        # Check bounds with epsilon padding for numerical stability
        violations = 0
        epsilon = 1e-8
        
        # Check x bounds
        violations += np.sum(points[:, 0] < 0 + epsilon)
        violations += np.sum(points[:, 0] > 1 - epsilon)
        
        # Check y bounds
        violations += np.sum(points[:, 1] < 0 + epsilon)
        violations += np.sum(points[:, 1] > 1 - epsilon)
        
        return -violations
    
    def initialize_points(method: str = 'hexagonal') -> np.ndarray:
        """Initialize points with various structured patterns."""
        np.random.seed(42)
        
        if method == 'hexagonal':
            # Create hexagonal grid pattern
            points = []
            rows, cols = 4, 4
            
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= 16:
                        break
                    x = j * 0.25 + (i % 2) * 0.125
                    y = i * 0.25
                    # Add random perturbation
                    x += np.random.normal(0, 0.01)
                    y += np.random.normal(0, 0.01)
                    points.append([x, y])
            
            points = np.array(points[:16])
            
            # Normalize and scale
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
            points[:, 0] *= 0.95
            points[:, 1] *= 0.95
            points[:, 0] += 0.025
            points[:, 1] += 0.025
            
        elif method == 'spiral':
            # Create spiral pattern
            points = []
            for i in range(16):
                angle = 2 * np.pi * i / 16
                radius = 0.4 * (i / 16)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                # Add random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                points.append([x, y])
            points = np.array(points)
            
        elif method == 'random':
            # Pure random initialization
            points = np.random.rand(16, 2)
            
        else:
            # Default to hexagonal
            points = initialize_points('hexagonal')
            
        return np.clip(points, 0, 1)
    
    def bounded_objective(x: np.ndarray) -> float:
        """Objective function with bounded input."""
        # Clamp points to valid bounds
        points = np.clip(x.reshape(-1, 2), 0, 1).flatten()
        return objective_function(points)
    
    def optimize_with_de(x0: np.ndarray, maxiter: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize using differential evolution."""
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    bounded_objective,
                    bounds,
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
            
        return x0, objective_function(x0)
    
    def local_optimize(x0: np.ndarray, maxiter: int = 500) -> Tuple[np.ndarray, float]:
        """Perform local optimization using L-BFGS-B."""
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = minimize(
                bounded_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                return result.x, -result.fun
        except Exception:
            pass
            
        return x0, objective_function(x0)
    
    # Set up optimization parameters
    np.random.seed(42)
    best_points = None
    best_ratio = -np.inf
    
    # Multiple initialization strategies
    init_strategies = ['hexagonal', 'spiral', 'random']
    
    # Multi-start optimization with different strategies
    for strategy in init_strategies:
        for restart in range(3):  # 3 restarts per strategy
            # Generate initial points
            initial_points = initialize_points(strategy)
            x0 = initial_points.flatten()
            
            # Global optimization with DE
            de_points, de_ratio = optimize_with_de(x0, maxiter=100)
            
            # Local refinement
            refined_points, refined_ratio = local_optimize(de_points, maxiter=500)
            
            # Update best solution
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.reshape(-1, 2)
    
    # Final refinement with more iterations if we found a good solution
    if best_points is not None:
        final_points = best_points.flatten()
        final_points, final_ratio = local_optimize(final_points, maxiter=1000)
        
        if final_ratio > best_ratio:
            best_points = final_points.reshape(-1, 2)
    
    # Fallback to hexagonal grid if no good solution found
    if best_points is None:
        initial_points = initialize_points('hexagonal')
        best_points = initial_points
    
    # Ensure all points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points

# EVOLVE-BLOCK-END