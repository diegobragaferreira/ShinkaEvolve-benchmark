# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time
from typing import List, Tuple, Optional, Callable

class PointOptimizer:
    """Structured optimizer for maximizing min/max distance ratio of 16 points in 2D."""
    
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
        
        # Optimization parameters
        self.max_iter_local = 500
        self.tolerance = 1e-12
        self.de_maxiter = 30
        self.de_popsize = 20
        
    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimize negative ratio of min/max distances."""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -1e10
            
        return -min_dist / max_dist
    
    def evaluate_solution(self, points: np.ndarray) -> float:
        """Evaluate the quality of a solution by computing min/max distance ratio."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def create_initial_grid(self) -> np.ndarray:
        """Create initial 4x4 grid points."""
        grid_size = 4
        x_vals = np.linspace(0.05, 0.95, grid_size)
        y_vals = np.linspace(0.05, 0.95, grid_size)
        return np.array([[x, y] for x in x_vals for y in y_vals])
    
    def perturb_points(self, points: np.ndarray, magnitude: float) -> np.ndarray:
        """Add controlled random perturbations."""
        np.random.seed(self.seed)
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)
    
    def compute_current_ratio(self, points: np.ndarray) -> float:
        """Compute current min/max distance ratio for adaptive scaling."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def adaptive_perturb_points(self, points: np.ndarray, base_magnitude: float = 0.02) -> np.ndarray:
        """Add random perturbations with adaptive magnitude based on current configuration quality."""
        current_ratio = self.compute_current_ratio(points)
        
        # Scale perturbation based on configuration quality
        if current_ratio < 0.1:
            magnitude = base_magnitude * 2.0
        elif current_ratio < 0.2:
            magnitude = base_magnitude * 1.5
        else:
            magnitude = base_magnitude * 0.5
            
        return self.perturb_points(points, magnitude)
    
    def create_spiral_pattern(self) -> np.ndarray:
        """Create a spiral-like initial pattern."""
        np.random.seed(self.seed)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])
        return self.perturb_points(spiral_points, 0.02)
    
    def create_hexagonal_pattern(self) -> np.ndarray:
        """Create a hexagonal pattern."""
        np.random.seed(self.seed)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                points.append([x, y])
        return np.array(points)
    
    def evolutionary_multistart(self) -> List[np.ndarray]:
        """Use evolutionary algorithm for global search to find good starting points."""
        def de_objective(x):
            points = x.reshape(-1, 2)
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return -1e10
            return -min_dist / max_dist

        bounds = [(0, 1) for _ in range(32)]
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=self.de_maxiter,
                popsize=self.de_popsize,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=self.seed,
                workers=1
            )

            if de_result.success:
                best_de_points = de_result.x.reshape(-1, 2)
                return [best_de_points]
        except Exception:
            pass

        return []
    
    def local_refinement(self, points_list: List[np.ndarray]) -> List[np.ndarray]:
        """Refine point configurations using local optimization."""
        refined_results = []
        for points in points_list:
            try:
                x0 = points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Try multiple local optimizers for robustness
                for method in ['L-BFGS-B', 'SLSQP']:
                    result = minimize(
                        self.objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': self.max_iter_local, 'ftol': self.tolerance, 'gtol': self.tolerance}
                    )

                    if result.success:
                        refined_points = result.x.reshape(-1, 2)
                        refined_results.append(refined_points)
                        break
            except Exception:
                continue
        return refined_results
    
    def run_strategies(self) -> Tuple[np.ndarray, float]:
        """Execute all optimization strategies and return best result."""
        best_ratio = 0.0
        best_points = None
        
        # Strategy 1: Evolutionary algorithm
        try:
            evolutions = self.evolutionary_multistart()
            if evolutions:
                refined_evolutions = self.local_refinement(evolutions)
                for refined_points in refined_evolutions:
                    ratio = self.evaluate_solution(refined_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
        except Exception:
            pass
        
        # Strategy 2: Grid-based patterns with adaptive perturbations
        strategies = [
            ("Grid", self.create_initial_grid(), 0.02),
            ("Spiral", self.create_spiral_pattern(), 0.03),
            ("Hexagonal", self.create_hexagonal_pattern(), 0.02),
            ("Random", np.random.rand(16, 2), 0.03)
        ]
        
        for name, base_points, magnitude in strategies:
            try:
                # Apply adaptive perturbation
                perturbed = self.adaptive_perturb_points(base_points, magnitude)
                x0 = perturbed.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                result = minimize(
                    self.objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.max_iter_local, 'ftol': self.tolerance, 'gtol': self.tolerance}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = self.evaluate_solution(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception:
                continue
        
        # Strategy 3: Additional random starts
        for i in range(3):
            try:
                np.random.seed(self.seed + i)
                random_points = np.random.rand(16, 2)
                perturbed = self.adaptive_perturb_points(random_points, 0.03)
                x0 = perturbed.flatten()
                bounds = [(0, 1) for _ in range(32)]

                result = minimize(
                    self.objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.max_iter_local, 'ftol': self.tolerance, 'gtol': self.tolerance}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = self.evaluate_solution(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception:
                continue
                
        # Fallback to default if no good solution found
        if best_points is None:
            grid_points = self.create_initial_grid()
            best_points = grid_points.copy()
            
        return best_points, best_ratio

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer()
    best_points, _ = optimizer.run_strategies()
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    return best_points

# EVOLVE-BLOCK-END