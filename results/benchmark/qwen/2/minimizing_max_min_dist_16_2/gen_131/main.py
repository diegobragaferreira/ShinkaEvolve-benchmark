# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0, min_dist, max_dist
            
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio
    
    def generate_hexagonal_grid(self) -> np.ndarray:
        """Generate hexagonal lattice initial configuration."""
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        for i in range(rows):
            for j in range(cols):
                x_offset = spacing_x * 0.5 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Ensure bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points)
    
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid initial configuration."""
        points = []
        side_length = int(math.ceil(math.sqrt(self.num_points)))
        
        for i in range(side_length):
            for j in range(side_length):
                if len(points) >= self.num_points:
                    break
                x = (i + 0.5) / side_length
                y = (j + 0.5) / side_length
                points.append([x, y])
        
        return np.array(points)[:self.num_points]
    
    def generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate a focused set of diverse initial configurations."""
        configs = []
        
        # Generate different base configurations
        configs.append(self.generate_hexagonal_grid())
        configs.append(self.generate_fibonacci_spiral())
        configs.append(self.generate_regular_grid())
        
        # Add perturbed versions - fewer perturbations for better performance
        np.random.seed(42)
        for config in configs:
            # Single perturbation level for faster execution
            perturbed = config + np.random.normal(0, 0.02, config.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            configs.append(perturbed)
        
        return configs
    
    def optimize_single_start(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Perform optimized single-point optimization."""
        try:
            # Use L-BFGS-B with fewer iterations for faster convergence
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 150, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            
            if result.success:
                return result.x.reshape(-1, self.dimension)
        except Exception:
            return None
        return None
    
    def get_best_solution(self, configs: List[np.ndarray]) -> np.ndarray:
        """Find best solution among the best initial configurations only."""
        best_ratio = -np.inf
        best_points = None
        
        # Only evaluate the top 3 configurations (reducing computation)
        evaluated_configs = []
        for config in configs:
            # Quick optimization to assess quality
            optimized_points = self.optimize_single_start(config.flatten())
            
            if optimized_points is not None:
                ratio, _, _ = self.calculate_ratio(optimized_points)
                evaluated_configs.append((ratio, optimized_points))
        
        # Sort by ratio and take the best ones
        evaluated_configs.sort(key=lambda x: x[0], reverse=True)
        
        # Test top 3 configurations with full optimization
        for ratio, config in evaluated_configs[:3]:
            # Full optimization on top candidates
            optimized_points = self.optimize_single_start(config.flatten())
            
            if optimized_points is not None:
                final_ratio, _, _ = self.calculate_ratio(optimized_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
        
        # Fallback to first configuration if nothing worked
        return best_points if best_points is not None else configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = PointDispersionOptimizer(16, 2)
    
    # Generate initial configurations (fewer to speed up)
    initial_configs = optimizer.generate_initial_configurations()
    
    # Find best solution with focused optimization
    best_points = optimizer.get_best_solution(initial_configs)
    
    return best_points

# EVOLVE-BLOCK-END