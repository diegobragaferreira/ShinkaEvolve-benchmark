# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math


class PointOptimizer:
    """Handles the optimization of point arrangements to maximize min/max distance ratio."""
    
    def __init__(self, n_points=16, seed=42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)
    
    def _initialize_hexagonal_grid(self):
        """Create initial hexagonal grid pattern."""
        points = np.zeros((self.n_points, 2))
        rows = 4
        cols = 4
        spacing = 0.25
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < self.n_points:
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * math.sqrt(3) / 2
                    points[idx] = [x, y]
                    idx += 1
        
        # Normalize to [0.1, 0.9] range
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1
        
        return points
    
    def _initialize_spiral_pattern(self):
        """Create initial spiral pattern."""
        points = np.zeros((self.n_points, 2))
        angles = np.linspace(0, 4*np.pi, self.n_points)
        radii = np.linspace(0.1, 0.4, self.n_points)
        
        for i in range(self.n_points):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])
            
        return points
    
    def _initialize_random(self):
        """Create random initial points."""
        return np.random.uniform(0.1, 0.9, (self.n_points, 2))
    
    def _compute_ratio(self, points):
        """Compute min/max distance ratio for given point configuration."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle case where all points might be coincident
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def _objective_function(self, points_vec):
        """Objective function for optimization (negative ratio to maximize)."""
        points = points_vec.reshape(-1, 2)
        ratio = self._compute_ratio(points)
        return -ratio  # Negative because scipy minimizes
    
    def _constraint_function(self, points_vec):
        """Constraint function to keep points within bounds [0,1] x [0,1]."""
        points = points_vec.reshape(-1, 2)
        # Check if any point is outside [0,1] bounds
        violations = 0
        
        # Check x bounds
        violations += np.sum(points[:, 0] < 0)
        violations += np.sum(points[:, 0] > 1)
        
        # Check y bounds
        violations += np.sum(points[:, 1] < 0)
        violations += np.sum(points[:, 1] > 1)
        
        # Return negative of violations (positive if all constraints satisfied)
        return -violations
    
    def _optimize_with_de(self, initial_points):
        """Use differential evolution for global search."""
        bounds = [(0, 1) for _ in range(2*self.n_points)]
        result = differential_evolution(
            self._objective_function,
            bounds,
            maxiter=1000,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.seed,
            disp=False
        )
        return result.x.reshape(-1, 2)
    
    def _optimize_with_local_refinement(self, initial_points):
        """Use local optimization (SLSQP) for fine-tuning."""
        bounds = [(0, 1) for _ in range(2*self.n_points)]
        constraints = {'type': 'ineq', 'fun': self._constraint_function}
        
        result = minimize(
            self._objective_function,
            initial_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10},
            tol=1e-10
        )
        return result.x.reshape(-1, 2)
    
    def optimize(self):
        """Main optimization procedure with multiple strategies."""
        # Try multiple initialization strategies
        initial_strategies = [
            self._initialize_hexagonal_grid(),
            self._initialize_spiral_pattern(),
            self._initialize_random()
        ]
        
        best_points = None
        best_ratio = -np.inf
        
        # Run optimization from each initialization
        for initial_config in initial_strategies:
            # Stage 1: Global search with Differential Evolution
            de_solution = self._optimize_with_de(initial_config)
            
            # Stage 2: Local refinement with SLSQP
            final_solution = self._optimize_with_local_refinement(de_solution)
            
            # Evaluate solution
            current_ratio = self._compute_ratio(final_solution)
            
            # Update best solution
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = final_solution.copy()
        
        return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer(n_points=16, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END
