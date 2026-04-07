# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
from typing import List, Tuple, Optional
import time

class PointOptimizer:
    """Structured optimizer for maximizing min/max distance ratio of 16 points in 2D."""
    
    def __init__(self):
        self.best_points = None
        self.best_ratio = -np.inf
        self.start_time = None
        
    def golden_spiral_2d(self, n_points: int) -> np.ndarray:
        """Generate points on a 2D golden spiral."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)
    
    def hexagonal_lattice_2d(self, n_points: int) -> np.ndarray:
        """Generate points on a 2D hexagonal lattice."""
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))

        points = []
        spacing = 1.0 / max(rows, cols)

        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x_offset = (i % 2) * spacing / 2
                x = (j * spacing) + x_offset
                y = i * spacing
                points.append([x, y])

        return np.array(points[:n_points])
    
    def structured_grid_initializer(self) -> np.ndarray:
        """Create structured 4x4 grid with strategic perturbations."""
        # Create regular grid
        grid_points = np.array([[i/3, j/3] for i in range(4) for j in range(4)])[:16]
        
        # Apply adaptive perturbations
        for i in range(16):
            row, col = i // 4, i % 4
            # More aggressive perturbations for corners and edges
            if row in [0, 3] or col in [0, 3]:
                std = 0.03
            else:
                std = 0.01
            grid_points[i] += np.random.normal(0, std, 2)
            
        # Ensure bounds
        grid_points = np.clip(grid_points, 0, 1)
        return grid_points
    
    def corner_based_initializer(self) -> np.ndarray:
        """Create a configuration with strategic corner placement."""
        # Corner points
        corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        # Edge midpoints
        edges = np.array([[0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]])
        # Cross pattern in center
        cross = np.array([[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]])
        # Diamond pattern
        diamond = np.array([[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]])
        
        points = np.vstack([corners, edges, cross, diamond])
        return points
    
    def compute_ratio(self, points: np.ndarray) -> float:
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def regularized_objective(self, x: np.ndarray) -> float:
        """Objective function with regularization to avoid numerical issues."""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        eps = 1e-12
        if max_dist < eps:
            return -1.0

        ratio = min_dist / (max_dist + eps)
        return -ratio
    
    def adaptive_optimization(self, points: np.ndarray) -> np.ndarray:
        """Perform adaptive optimization based on initial quality."""
        initial_ratio = -self.regularized_objective(points.flatten())
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        # Dynamic optimization parameters based on quality
        if initial_ratio > 0.25:
            options = {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
        elif initial_ratio > 0.20:
            options = {'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}
        elif initial_ratio > 0.15:
            options = {'maxiter': 1200, 'ftol': 1e-10, 'gtol': 1e-10}
        else:
            options = {'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
            
        try:
            result = minimize(
                self.regularized_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options=options
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
            
        return points
    
    def multi_stage_refinement(self, points: np.ndarray) -> np.ndarray:
        """Apply multi-stage refinement for better local improvement."""
        current_points = points.copy()
        
        # Stage 1: Basic optimization
        current_points = self.adaptive_optimization(current_points)
        
        # Stage 2: Local improvement using neighborhood search
        for _ in range(3):
            improved = False
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                best_point = original_point.copy()
                best_ratio = self.compute_ratio(current_points)
                
                # Test small perturbations
                for _ in range(10):
                    perturbation = np.random.normal(0, 0.005, 2)
                    test_point = original_point + perturbation
                    test_point = np.clip(test_point, 0, 1)
                    
                    temp_points = current_points.copy()
                    temp_points[i] = test_point
                    
                    test_ratio = self.compute_ratio(temp_points)
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_point = test_point.copy()
                
                if not np.array_equal(current_points[i], best_point):
                    current_points[i] = best_point
                    improved = True
                    
            if not improved:
                break
                
        return current_points
    
    def run_single_optimization(self, init_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run a single optimization from given initial points."""
        try:
            # Apply multi-stage refinement
            refined_points = self.multi_stage_refinement(init_points)
            ratio = self.compute_ratio(refined_points)
            
            return refined_points, ratio
        except Exception as e:
            warnings.warn(f"Single optimization failed: {e}")
            return init_points, self.compute_ratio(init_points)
    
    def generate_initial_strategies(self) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        strategies = []
        
        # 1. Golden spiral pattern
        spiral_points = self.golden_spiral_2d(16)
        if np.max(spiral_points) > np.min(spiral_points):
            spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
                np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1
        strategies.append(spiral_points.copy())
        
        # 2. Hexagonal lattice
        hex_points = self.hexagonal_lattice_2d(16)
        if np.max(hex_points) > np.min(hex_points):
            hex_points = (hex_points - np.min(hex_points, axis=0)) / (
                np.max(hex_points, axis=0) - np.min(hex_points, axis=0) + 1e-12)
        hex_points = hex_points * 0.8 + 0.1
        strategies.append(hex_points.copy())
        
        # 3. Structured grid with perturbations
        strategies.append(self.structured_grid_initializer())
        
        # 4. Corner-based pattern
        strategies.append(self.corner_based_initializer())
        
        # 5. Random uniform points
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        strategies.append(random_points)
        
        return strategies
    
    def optimize(self, max_time: int = 170) -> np.ndarray:
        """Main optimization loop with hierarchical approach."""
        self.start_time = time.time()
        strategies = self.generate_initial_strategies()
        
        # Evaluate all strategies
        results = []
        for i, strategy in enumerate(strategies):
            try:
                points, ratio = self.run_single_optimization(strategy)
                results.append((points, ratio, i))
            except Exception as e:
                warnings.warn(f"Strategy {i} failed: {e}")
                continue
        
        # Sort by quality and take top performers
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            best_initial_points = results[0][0]
            best_initial_ratio = results[0][1]
            
            # Save best so far
            self.best_points = best_initial_points
            self.best_ratio = best_initial_ratio
            
            # Run further optimization on best strategy
            final_points = self.multi_stage_refinement(best_initial_points)
            final_ratio = self.compute_ratio(final_points)
            
            if final_ratio > self.best_ratio:
                self.best_points = final_points
                self.best_ratio = final_ratio
                
        # Fallback to best initial configuration if nothing worked
        if self.best_points is None:
            self.best_points = strategies[0]
            
        return self.best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    try:
        optimizer = PointOptimizer()
        points = optimizer.optimize(max_time=170)
        return points
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to simple approach if something fails
        np.random.seed(42)
        return np.random.rand(16, 2)

# EVOLVE-BLOCK-END