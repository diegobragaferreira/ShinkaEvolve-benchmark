# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings

class PointOptimizer:
    """
    A class-based approach to optimize point distribution for maximizing 
    minimum-to-maximum distance ratio in 2D space.
    """
    
    def __init__(self, n_points=16, dimension=2, seed=42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)
        
    def _initialize_hexagonal_grid(self):
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
    
    def _initialize_structured_configurations(self):
        """Generate multiple structured initial configurations."""
        configs = []
        
        # Configuration 1: Modified hexagonal grid
        config1 = self._initialize_hexagonal_grid()
        configs.append(config1)
        
        # Configuration 2: Circle arrangement with noise
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False)
        radii = 0.4 + 0.1 * np.sin(np.arange(self.n_points) * np.pi / 8)
        center = np.array([0.5, 0.5])
        config2 = np.column_stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ])
        config2 += np.random.normal(0, 0.01, config2.shape)
        configs.append(np.clip(config2, 0.01, 0.99))
        
        # Configuration 3: Random grid with padding
        config3 = np.random.rand(self.n_points, self.dimension) * 0.9 + 0.05
        configs.append(config3)
        
        return configs
    
    def _compute_objective(self, x):
        """Compute the negative ratio of min/max distances."""
        points = x.reshape(-1, self.dimension)
        
        # Use squareform for better numerical stability
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    def _run_global_optimization(self, x0, bounds):
        """Run differential evolution for global optimization."""
        try:
            result = differential_evolution(
                self._compute_objective,
                bounds,
                seed=self.seed,
                maxiter=250,
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
    
    def _run_local_refinement(self, x0, bounds):
        """Run local optimization with fallback options."""
        try:
            # Try L-BFGS-B first
            result = minimize(
                self._compute_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if not result.success:
                # Fallback to SLSQP
                result = minimize(
                    self._compute_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            
            return result
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            return None
    
    def optimize(self):
        """Main optimization routine with multi-stage approach."""
        # Generate initial configurations
        initial_configs = self._initialize_structured_configurations()
        
        best_result = None
        best_value = float('inf')
        
        # Try each initial configuration
        for i, config in enumerate(initial_configs):
            x0 = np.clip(config, 0.01, 0.99).flatten()
            bounds = [(0.01, 0.99) for _ in range(self.n_points * self.dimension)]
            
            # Global optimization
            de_result = self._run_global_optimization(x0, bounds)
            if de_result is None:
                continue
                
            # Local refinement
            lr_result = self._run_local_refinement(de_result.x, bounds)
            if lr_result is None:
                continue
                
            # Track best result
            if lr_result.fun < best_value:
                best_value = lr_result.fun
                best_result = lr_result
        
        # If all attempts failed, use the first configuration
        if best_result is None:
            x0 = initial_configs[0].flatten()
            bounds = [(0.01, 0.99) for _ in range(self.n_points * self.dimension)]
            de_result = self._run_global_optimization(x0, bounds)
            if de_result is not None:
                best_result = self._run_local_refinement(de_result.x, bounds)
        
        # Return final result
        if best_result is not None:
            points = best_result.x.reshape(-1, self.dimension)
        else:
            # Fallback to first initial configuration
            points = initial_configs[0]
            
        # Ensure final points are within bounds
        points = np.clip(points, 0.01, 0.99)
        return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer(n_points=16, dimension=2, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END