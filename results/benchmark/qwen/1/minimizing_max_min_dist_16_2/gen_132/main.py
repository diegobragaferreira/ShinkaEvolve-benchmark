# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings

class PointArrangementOptimizer:
    """Optimizes point arrangements to maximize min/max distance ratio"""
    
    def __init__(self, n_points=16):
        self.n_points = n_points
        self.best_solution = None
        self.best_ratio = float('-inf')
        
    def _compute_distances(self, points):
        """Compute pairwise distances with numerical stability"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        return distances
    
    def _objective_function(self, x):
        """Objective function to maximize min/max distance ratio"""
        points = x.reshape(-1, 2)
        distances = self._compute_distances(points)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
            
        return -d_min / d_max
    
    def _create_hexagonal_initialization(self):
        """Create initial configuration using hexagonal packing principles"""
        np.random.seed(42)
        points = np.zeros((self.n_points, 2))
        
        # Create hexagonal grid with perturbations
        rows, cols = 4, 4
        sqrt3 = np.sqrt(3)
        row_spacing = 0.8 / (rows - 1) if rows > 1 else 0.8
        col_spacing = 0.8 / (cols - 1) if cols > 1 else 0.8
        
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= self.n_points:
                    break
                # Hexagonal packing with offset rows
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                
                # Scale to fit in [0.1, 0.9] range to avoid boundary issues
                x_scaled = 0.1 + x * 0.8
                y_scaled = 0.1 + y * 0.8
                
                # Add slight random perturbation
                x_scaled += np.random.normal(0, 0.01)
                y_scaled += np.random.normal(0, 0.01)
                
                points[i * cols + j] = [x_scaled, y_scaled]
        
        # Ensure bounds
        points = np.clip(points, 0.01, 0.99)
        return points
    
    def _create_multi_initialization(self):
        """Create multiple initial configurations for robust optimization"""
        initial_configs = []
        
        # Configuration 1: Hexagonal grid
        config1 = self._create_hexagonal_initialization()
        initial_configs.append(config1.flatten())
        
        # Configuration 2: Circular arrangement
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False)
        radii = 0.4 + 0.1 * np.sin(np.arange(self.n_points) * np.pi / 8)
        center = np.array([0.5, 0.5])
        config2 = np.column_stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ])
        config2 += np.random.normal(0, 0.01, config2.shape)
        config2 = np.clip(config2, 0.01, 0.99)
        initial_configs.append(config2.flatten())
        
        # Configuration 3: Random with bounds
        config3 = np.random.rand(self.n_points, 2) * 0.8 + 0.1
        config3 = np.clip(config3, 0.01, 0.99)
        initial_configs.append(config3.flatten())
        
        return initial_configs
    
    def _optimize_stage(self, objective_func, x0, bounds, method='L-BFGS-B'):
        """Perform optimization with given method and parameters"""
        try:
            if method == 'DE':
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42,
                    maxiter=200,
                    popsize=25,
                    tol=1e-9,
                    recombination=0.9,
                    mutation=(0.8, 1.0),
                    disp=False
                )
                return result.x
            else:
                result = minimize(
                    objective_func,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14},
                    callback=None
                )
                return result.x if result.success else x0
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            return x0
    
    def optimize(self):
        """Main optimization loop with multi-phase approach"""
        # Create bounds
        bounds = [(0.01, 0.99) for _ in range(2 * self.n_points)]
        
        # Get multiple initial configurations
        initial_configs = self._create_multi_initialization()
        
        # Phase 1: Global search with multiple starting points
        best_x = None
        best_value = float('inf')
        
        for i, x0 in enumerate(initial_configs):
            try:
                # Global optimization with DE
                global_x = self._optimize_stage(self._objective_function, x0, bounds, 'DE')
                
                # Local refinement
                local_x = self._optimize_stage(self._objective_function, global_x, bounds, 'L-BFGS-B')
                
                # Evaluate result
                current_value = self._objective_function(local_x)
                
                if current_value < best_value:
                    best_value = current_value
                    best_x = local_x
                    
            except Exception as e:
                warnings.warn(f"Optimization phase failed for config {i}: {e}")
                continue
        
        # Phase 2: Final polishing with enhanced objective
        if best_x is not None:
            final_x = self._optimize_stage(self._objective_function, best_x, bounds, 'L-BFGS-B')
            points = final_x.reshape(-1, 2)
        else:
            # Fallback to first configuration
            points = initial_configs[0].reshape(-1, 2)
        
        # Ensure final bounds
        points = np.clip(points, 0.01, 0.99)
        
        return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointArrangementOptimizer(n_points=16)
    return optimizer.optimize()

# EVOLVE-BLOCK-END