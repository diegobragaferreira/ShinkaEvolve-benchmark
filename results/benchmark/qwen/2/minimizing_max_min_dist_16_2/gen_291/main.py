# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from numba import jit
import warnings
import time
from typing import Tuple, List, Optional
import math

@jit(nopython=True)
def fast_pdist_squared(points):
    """Fast computation of squared pairwise distances using numba"""
    n = points.shape[0]
    distances_squared = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            distances_squared[i, j] = dist_sq
            distances_squared[j, i] = dist_sq
    return distances_squared

class PointConfiguration:
    """Class to manage point configurations and their evaluation metrics"""
    
    def __init__(self, points: np.ndarray):
        self.points = points.copy()
        self._cached_ratio = None
        self._cached_distances = None
    
    def compute_min_max_ratio(self) -> float:
        """Compute the minimum to maximum distance ratio for given points."""
        if len(self.points) < 2:
            return 0.0

        # Use cached distances if available
        if self._cached_distances is not None:
            distances = self._cached_distances
        else:
            # Use faster distance calculation with numba
            distances_squared = fast_pdist_squared(self.points)
            distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])
            self._cached_distances = distances
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        ratio = min_dist / max_dist
        self._cached_ratio = ratio
        return ratio
    
    def get_ratio_cached(self) -> float:
        """Get cached ratio or compute if not cached"""
        if self._cached_ratio is not None:
            return self._cached_ratio
        return self.compute_min_max_ratio()
    
    def reset_cache(self):
        """Clear cached values"""
        self._cached_ratio = None
        self._cached_distances = None

class OptimizationStrategy:
    """Base class for optimization strategies"""
    
    def __init__(self, max_time: float = 170):
        self.max_time = max_time
        self.start_time = None
    
    def initialize(self) -> PointConfiguration:
        """Initialize configuration"""
        raise NotImplementedError
    
    def optimize(self, initial_config: PointConfiguration) -> PointConfiguration:
        """Optimize the configuration"""
        raise NotImplementedError
    
    def check_time_remaining(self, threshold: float = 5.0) -> bool:
        """Check if there's enough time remaining"""
        if self.start_time is None:
            self.start_time = time.time()
        return (time.time() - self.start_time) < (self.max_time - threshold)

class ProgressiveOptimizationStrategy(OptimizationStrategy):
    """Multi-stage progressive optimization strategy"""
    
    def __init__(self, max_time: float = 170):
        super().__init__(max_time)
        self.stage_timing = {
            'coarse': 0.3,
            'medium': 0.3,
            'fine': 0.3
        }
    
    def initialize(self) -> PointConfiguration:
        """Create initial configuration with structured pattern"""
        # Start with regular 4x4 grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i / 3.0  # Normalized to [0,1] range
                y = j / 3.0
                grid_points.append([x, y])
        
        points = np.array(grid_points)
        
        # Apply adaptive perturbations based on location
        # Emphasize corners and edges to encourage better spreading
        for i in range(16):
            row, col = i // 4, i % 4
            
            # More aggressive perturbations for corners to break symmetry
            if (row in [0, 3] and col in [0, 3]):
                std = 0.035
            elif (row in [0, 3] or col in [0, 3]):
                std = 0.02
            else:
                std = 0.01
                
            # Apply perturbation
            points[i, 0] += np.random.normal(0, std)
            points[i, 1] += np.random.normal(0, std)
        
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        
        # Fix specific points to break symmetry
        points[0] = [0.0, 0.0]      # Bottom-left corner
        points[3] = [1.0, 0.0]      # Bottom-right corner
        points[12] = [0.0, 1.0]     # Top-left corner
        points[15] = [1.0, 1.0]     # Top-right corner
        points[5] = [0.25, 0.25]    # Interior point
        points[10] = [0.75, 0.75]   # Interior point
        
        return PointConfiguration(points)
    
    def optimize_stage(self, points: PointConfiguration, 
                      max_iter: int, ftol: float, gtol: float) -> PointConfiguration:
        """Perform single optimization stage"""
        # Reset cache before optimization
        points.reset_cache()
        
        try:
            result = minimize(
                self.objective_with_regularization,
                points.points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            
            if result.success:
                optimized_points = PointConfiguration(result.x.reshape(-1, 2))
                return optimized_points
        except Exception:
            pass
        
        return points
    
    def objective_with_regularization(self, x):
        """Objective function with regularization"""
        points = x.reshape(-1, 2)
        distances_squared = fast_pdist_squared(points)
        distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio
    
    def optimize(self, initial_config: PointConfiguration) -> PointConfiguration:
        """Apply progressive optimization with multiple stages"""
        current_points = initial_config.copy()
        
        # Stage 1: Coarse optimization with relaxed tolerances
        if self.check_time_remaining(50):
            current_points = self.optimize_stage(
                current_points, 300, 1e-8, 1e-6
            )
        
        # Stage 2: Refinement with medium precision
        if self.check_time_remaining(30):
            # Apply local search to improve quality
            current_points = self.local_search(current_points, 30)
            
            # Fine optimization
            current_points = self.optimize_stage(
                current_points, 500, 1e-10, 1e-8
            )
        
        # Stage 3: Final high precision optimization
        if self.check_time_remaining(10):
            # Apply final local search
            current_points = self.local_search(current_points, 50)
            
            # Very tight optimization
            current_points = self.optimize_stage(
                current_points, 800, 1e-12, 1e-10
            )
        
        return current_points
    
    def local_search(self, points: PointConfiguration, max_iterations: int) -> PointConfiguration:
        """Perform targeted local search to improve minimum distance"""
        current_points = points.points.copy()
        best_points = points.points.copy()
        best_ratio = points.get_ratio_cached()
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try moving each point to improve the minimum distance
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                best_move = original_point.copy()
                best_improvement = 0.0
                
                # Test small movements in different directions
                movements = [
                    [-0.01, -0.01], [-0.01, 0], [-0.01, 0.01],
                    [0, -0.01], [0, 0.01],
                    [0.01, -0.01], [0.01, 0], [0.01, 0.01]
                ]
                
                # Test each movement
                for dx, dy in movements:
                    test_point = original_point.copy()
                    test_point[0] += dx
                    test_point[1] += dy
                    
                    # Clip to bounds
                    test_point[0] = np.clip(test_point[0], 0.001, 0.999)
                    test_point[1] = np.clip(test_point[1], 0.001, 0.999)
                    
                    # Temporarily update this point
                    temp_points = current_points.copy()
                    temp_points[i] = test_point
                    
                    # Create temporary configuration for ratio check
                    temp_config = PointConfiguration(temp_points)
                    ratio = temp_config.get_ratio_cached()
                    improvement = ratio - best_ratio
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = test_point.copy()
                
                # Apply the best move if it improves the solution
                if best_improvement > 0:
                    current_points[i] = best_move
                    improved = True
                    
                    # Update best solution if this is better
                    temp_config = PointConfiguration(current_points)
                    ratio = temp_config.get_ratio_cached()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
            
            # Early stopping if no improvement
            if not improved:
                break
                
        return PointConfiguration(best_points)

class MultiStartStrategy(OptimizationStrategy):
    """Strategy that uses multiple initial configurations with restarts"""
    
    def __init__(self, max_time: float = 170):
        super().__init__(max_time)
        self.initial_generators = [
            self._create_structured_grid,
            self._create_golden_spiral,
            self._create_hexagonal_lattice,
            self._create_perturbed_grid,
            self._create_random_uniform
        ]
    
    def _create_structured_grid(self) -> PointConfiguration:
        """Create structured 4x4 grid with symmetry breaking"""
        points = np.array([[i/3, j/3] for i in range(4) for j in range(4)])
        # Add small random perturbations
        points += np.random.normal(0, 0.02, (16, 2))
        points = np.clip(points, 0, 1)
        
        # Fix corner and some interior points to break symmetry
        points[0] = [0.0, 0.0]      # Bottom-left corner
        points[3] = [1.0, 0.0]      # Bottom-right corner
        points[12] = [0.0, 1.0]     # Top-left corner
        points[15] = [1.0, 1.0]     # Top-right corner
        points[5] = [0.25, 0.25]    # Interior point
        points[10] = [0.75, 0.75]   # Interior point
        
        return PointConfiguration(points)
    
    def _create_golden_spiral(self) -> PointConfiguration:
        """Create golden spiral pattern"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(16):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / 15) if i > 0 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        spiral_points = np.array(points)
        # Scale and center the spiral
        spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
            np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        
        return PointConfiguration(spiral_points)
    
    def _create_hexagonal_lattice(self) -> PointConfiguration:
        """Create hexagonal lattice pattern"""
        rows = 4
        cols = 4
        points = []
        spacing = 1.0 / max(rows, cols)

        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Offset every other row
                x_offset = (i % 2) * spacing / 2
                x = (j * spacing) + x_offset
                y = i * spacing
                points.append([x, y])

        hex_points = np.array(points[:16])
        # Normalize to [0.1, 0.9] range
        hex_points = (hex_points - np.min(hex_points, axis=0)) / (
            np.max(hex_points, axis=0) - np.min(hex_points, axis=0) + 1e-12)
        hex_points = hex_points * 0.8 + 0.1
        
        return PointConfiguration(hex_points)
    
    def _create_perturbed_grid(self) -> PointConfiguration:
        """Create perturbed grid pattern"""
        points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        points += np.random.normal(0, 0.05, (16, 2))
        points = np.clip(points, 0, 1)
        
        # Fix corners and interior points
        points[0] = [0.0, 0.0]
        points[3] = [1.0, 0.0]
        points[12] = [0.0, 1.0]
        points[15] = [1.0, 1.0]
        points[5] = [0.25, 0.25]
        points[10] = [0.75, 0.75]
        
        return PointConfiguration(points)
    
    def _create_random_uniform(self) -> PointConfiguration:
        """Create random uniform distribution"""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return PointConfiguration(points)
    
    def initialize(self) -> PointConfiguration:
        """Initialize with best among multiple strategies"""
        # Use a smaller subset of strategies for quick initial assessment
        best_ratio = -np.inf
        best_config = None
        
        # Run a quick optimization on each initial configuration
        for generator in self.initial_generators[:3]:  # Use first 3 for quick test
            try:
                config = generator()
                # Quick rough optimization
                if self.check_time_remaining(50):
                    simple_strategy = ProgressiveOptimizationStrategy(self.max_time * 0.1)
                    optimized_config = simple_strategy.optimize(config)
                    ratio = optimized_config.get_ratio_cached()
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_config = optimized_config
            except Exception:
                continue
        
        # If no good initial config found, return first one
        if best_config is None:
            try:
                config = self.initial_generators[0]()
                return config
            except Exception:
                np.random.seed(42)
                return PointConfiguration(np.random.rand(16, 2))
        
        return best_config
    
    def optimize(self, initial_config: PointConfiguration) -> PointConfiguration:
        """Apply progressive optimization to the best initial configuration"""
        strategy = ProgressiveOptimizationStrategy(self.max_time)
        return strategy.optimize(initial_config)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def execute_optimization_strategy() -> PointConfiguration:
        """Execute the main optimization strategy"""
        # Initialize with multi-start approach
        strategy = MultiStartStrategy(max_time=170)
        
        try:
            # Get best initial configuration
            best_initial = strategy.initialize()
            
            # Perform full optimization
            final_result = strategy.optimize(best_initial)
            
            return final_result
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Fallback to simple approach  
            np.random.seed(42)
            return PointConfiguration(np.random.rand(16, 2))
    
    # Execute main optimization
    try:
        final_config = execute_optimization_strategy()
        return final_config.points
    except Exception as e:
        warnings.warn(f"Failed completely: {e}")
        np.random.seed(42)
        return np.random.rand(16, 2)

# EVOLVE-BLOCK-END