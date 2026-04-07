# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time
from typing import List, Tuple, Optional, Callable

class ConfigurableInitializer:
    """Manages multiple geometric initialization strategies for point placement."""
    
    def __init__(self, n_points: int = 16, dimension: int = 2, seed: int = 42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)
    
    def _hexagonal_grid_init(self) -> np.ndarray:
        """Initialize points using a hexagonal grid pattern with perturbations."""
        points = []
        rows, cols = 4, 4
        sqrt3 = np.sqrt(3)
        spacing = 0.8
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * sqrt3 / 2
                
                # Scale to fit within [0.05, 0.95] range
                x_scaled = 0.05 + (x / (spacing * cols)) * 0.9
                y_scaled = 0.05 + (y / (spacing * rows * sqrt3 / 2)) * 0.9
                
                # Add slight random perturbation
                x_scaled += np.random.normal(0, 0.02)
                y_scaled += np.random.normal(0, 0.02)
                
                points.append([x_scaled, y_scaled])
        
        return np.array(points[:self.n_points])
    
    def _circle_arrangement_init(self) -> np.ndarray:
        """Initialize points in a circular arrangement with radial variation."""
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False)
        radii = 0.4 + 0.1 * np.sin(np.arange(self.n_points) * np.pi / 8)
        center = np.array([0.5, 0.5])
        points = np.column_stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ])
        points += np.random.normal(0, 0.01, points.shape)
        return np.clip(points, 0.05, 0.95)
    
    def _grid_offset_init(self) -> np.ndarray:
        """Initialize points in a grid with hexagonal offset pattern."""
        points = np.zeros((self.n_points, self.dimension))
        rows, cols = 4, 4
        row_spacing = 0.9 / (rows - 1) if rows > 1 else 0.9
        col_spacing = 0.9 / (cols - 1) if cols > 1 else 0.9
        
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= self.n_points:
                    break
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points[i * cols + j] = [x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005)]
        
        return np.clip(points, 0.05, 0.95)
    
    def _spiral_init(self) -> np.ndarray:
        """Initialize points in a spiral pattern."""
        points = np.zeros((self.n_points, self.dimension))
        for i in range(self.n_points):
            angle = i * 0.4
            radius = i * 0.04
            points[i] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]
        return np.clip(points, 0.05, 0.95)
    
    def generate_initial_configs(self) -> List[np.ndarray]:
        """Generate multiple initial configurations."""
        configs = []
        
        # Configuration 1: Hexagonal grid
        configs.append(self._hexagonal_grid_init())
        
        # Configuration 2: Circle arrangement
        configs.append(self._circle_arrangement_init())
        
        # Configuration 3: Grid with offset
        configs.append(self._grid_offset_init())
        
        # Configuration 4: Spiral pattern
        configs.append(self._spiral_init())
        
        return configs

class OptimizerPipeline:
    """Orchestrates the complete optimization pipeline."""
    
    def __init__(self, n_points: int = 16, dimension: int = 2, seed: int = 42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        self.initializer = ConfigurableInitializer(n_points, dimension, seed)
        self.bounds = [(0.01, 0.99) for _ in range(n_points * dimension)]
        
    def _objective_function(self, x: np.ndarray) -> float:
        """Computes the negative ratio of minimum to maximum distances."""
        points = x.reshape(-1, self.dimension)
        
        # Use squareform for better numerical stability
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    def _adaptive_objective_with_penalty(self, x: np.ndarray, penalty_weight: float = 1000.0) -> float:
        """Enhanced objective with geometric penalties."""
        points = x.reshape(-1, self.dimension)
        
        # Compute pairwise distances using squareform for better numerical stability
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # If all points are identical or near identical, penalize heavily
        if d_max == 0:
            return -1.0
        
        # Calculate ratio to maximize
        ratio = d_min / d_max
        
        # Add penalty for points near boundaries
        boundary_penalty = 0.0
        margin = 0.01
        for point in points:
            if (point[0] < margin or point[0] > 1-margin or 
                point[1] < margin or point[1] > 1-margin):
                boundary_penalty += penalty_weight * (margin - min(point[0], 1-point[0], point[1], 1-point[1]))
        
        # Add penalty for very small distances (close point clustering)
        min_distance_penalty = 0.0
        if d_min < 0.05:  # Threshold for clustering penalty
            min_distance_penalty = penalty_weight * (0.05 - d_min)
        
        total_penalty = boundary_penalty + min_distance_penalty
        return -(ratio - total_penalty / penalty_weight)
    
    def _global_optimization(self, x0: np.ndarray) -> Optional[minimize.OptimizeResult]:
        """Perform global optimization using differential evolution."""
        try:
            result = differential_evolution(
                self._adaptive_objective_with_penalty,
                self.bounds,
                seed=self.seed,
                maxiter=150,
                popsize=25,
                tol=1e-9,
                recombination=0.9,
                mutation=(0.8, 1.0),
                disp=False
            )
            return result
        except Exception as e:
            warnings.warn(f"Global optimization failed: {e}")
            return None
    
    def _local_refinement(self, x0: np.ndarray) -> Optional[minimize.OptimizeResult]:
        """Perform local refinement with fallback options."""
        try:
            # Try L-BFGS-B first
            result = minimize(
                self._adaptive_objective_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if not result.success:
                # Fallback to SLSQP
                result = minimize(
                    self._adaptive_objective_with_penalty,
                    x0,
                    method='SLSQP',
                    bounds=self.bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            
            return result
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            return None
    
    def _final_refinement(self, x0: np.ndarray) -> np.ndarray:
        """Perform final refinement with standard objective."""
        try:
            result = minimize(
                self._objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            return result.x.reshape(-1, self.dimension)
        except Exception as e:
            warnings.warn(f"Final refinement failed: {e}")
            return x0.reshape(-1, self.dimension)
    
    def optimize(self) -> np.ndarray:
        """Execute the complete optimization pipeline."""
        # Generate initial configurations
        initial_configs = self.initializer.generate_initial_configs()
        
        best_result = None
        best_value = float('inf')
        
        # Try each initial configuration
        for i, config in enumerate(initial_configs):
            x0 = config.flatten()
            
            # Global optimization
            de_result = self._global_optimization(x0)
            if de_result is None:
                continue
                
            # Local refinement
            lr_result = self._local_refinement(de_result.x)
            if lr_result is None:
                continue
                
            # Track best result
            if lr_result.fun < best_value:
                best_value = lr_result.fun
                best_result = lr_result
        
        # If all attempts failed, use the first configuration
        if best_result is None:
            x0 = initial_configs[0].flatten()
            de_result = self._global_optimization(x0)
            if de_result is not None:
                best_result = self._local_refinement(de_result.x)
        
        # Return final result
        if best_result is not None:
            points = best_result.x.reshape(-1, self.dimension)
        else:
            # Fallback to first initial configuration
            points = initial_configs[0]
            
        # Final refinement
        points = self._final_refinement(points.flatten())
        
        # Ensure final points are within bounds
        points = np.clip(points, 0.01, 0.99)
        return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = OptimizerPipeline(n_points=16, dimension=2, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END