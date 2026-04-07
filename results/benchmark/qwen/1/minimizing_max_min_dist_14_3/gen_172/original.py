# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math

class PointDistributionOptimizer:
    """Optimizes point distribution on a 3D sphere to maximize min/max distance ratio."""
    
    def __init__(self, num_points=14):
        self.num_points = num_points
        self.best_ratio = -np.inf
        self.best_points = None
        
    def _generate_icosahedron_points(self):
        """Generate initial points using icosahedron vertices."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(points, axis=1)
        return points / norms[:, np.newaxis]
    
    def _add_remaining_points(self, base_points, remaining_count):
        """Add remaining points using spherical coordinate distribution."""
        points = base_points.copy()
        for i in range(remaining_count):
            theta = math.acos(1 - 2 * (i / (remaining_count - 1)))
            phi = math.sqrt(self.num_points * math.pi) * theta
            
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            points = np.vstack([points, [x, y, z]])
        return points
    
    def _initialize_points(self):
        """Create initial point configuration using hybrid approach."""
        base_points = self._generate_icosahedron_points()
        remaining = self.num_points - len(base_points)
        
        if remaining > 0:
            points = self._add_remaining_points(base_points, remaining)
        else:
            points = base_points[:self.num_points]
        
        # Add jitter to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        return points / norms[:, np.newaxis]
    
    def _calculate_distance_ratio(self, points_flat):
        """Calculate the ratio of minimum to maximum distance."""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def _objective_function(self, points_flat):
        """Minimize negative of distance ratio (since we want to maximize)."""
        return -self._calculate_distance_ratio(points_flat)
    
    def _create_constraints(self):
        """Create constraint functions for unit sphere."""
        def constraint_func(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0
        
        constraints = []
        for i in range(self.num_points):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]})
        return constraints
    
    def _optimize_phase(self, x0, bounds, maxiter, ftol, gtol):
        """Perform optimization with specific parameters."""
        constraints = self._create_constraints()
        
        result = minimize(
            self._objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            tol=ftol
        )
        return result.x if result.success else x0
    
    def _progressive_optimization(self, x0):
        """Perform multi-phase optimization with progressive tightening."""
        # Phase 1: Coarse optimization
        bounds = [(-1.2, 1.2)] * len(x0)
        x1 = self._optimize_phase(x0, bounds, 200, 1e-4, 1e-4)
        
        # Phase 2: Medium optimization
        bounds = [(-1.1, 1.1)] * len(x0)
        x2 = self._optimize_phase(x1, bounds, 300, 1e-6, 1e-6)
        
        # Phase 3: Fine optimization
        bounds = [(-1.05, 1.05)] * len(x0)
        x3 = self._optimize_phase(x2, bounds, 500, 1e-8, 1e-8)
        
        return x3
    
    def optimize(self):
        """Main optimization routine with multi-start approach."""
        for restart in range(8):
            np.random.seed(42 + restart)
            
            # Initialize points
            initial_points = self._initialize_points()
            x0 = initial_points.flatten()
            
            # Add small random perturbation
            perturbation = np.random.normal(0, 0.01, x0.shape)
            x0 += perturbation
            
            # Optimize
            try:
                optimized_points = self._progressive_optimization(x0)
                
                # Calculate final ratio
                ratio = self._calculate_distance_ratio(optimized_points)
                
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = optimized_points.copy()
                    
            except Exception:
                continue
        
        # Fallback to initialization if no good solution found
        if self.best_points is None:
            initial_points = self._initialize_points()
            self.best_points = initial_points.flatten()
        
        return self.best_points.reshape(-1, 3)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    optimizer = PointDistributionOptimizer(num_points=14)
    return optimizer.optimize()

# EVOLVE-BLOCK-END
