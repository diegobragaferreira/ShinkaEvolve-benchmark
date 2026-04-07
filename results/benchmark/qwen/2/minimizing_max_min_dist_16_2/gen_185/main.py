# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List, Optional

class PointArrangementOptimizer:
    """Enhanced optimizer with hierarchical approach to maximize min/max distance ratio."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def calculate_min_max_ratio(self, points) -> float:
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0.0

        return dmin / dmax

    def objective_function(self, points) -> float:
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -self.calculate_min_max_ratio(points)

    def create_hexagonal_initialization(self) -> np.ndarray:
        """Create a hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < 16:
                    x = (j + 0.5 * (i % 2)) * spacing_x
                    y = i * spacing_y
                    points[idx] = [x, y]
                    idx += 1
        return points

    def create_symmetry_broken_initialization(self) -> np.ndarray:
        """Create an initialization that breaks common symmetries."""
        points = np.zeros((16, 2))
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        # Fix corner points to break rotational symmetry
        points[0] = [spacing_x * 0.5, spacing_y * 0.5]
        points[15] = [1.0 - spacing_x * 0.5, 1.0 - spacing_y * 0.5]

        idx = 1
        for i in range(rows):
            for j in range(cols):
                if idx < 16 and not (i == 0 and j == 0) and not (i == rows-1 and j == cols-1):
                    x = (j + 0.5 * (i % 2)) * spacing_x
                    y = i * spacing_y
                    # Add small random perturbation to break further symmetry
                    x += np.random.normal(0, 0.005)
                    y += np.random.normal(0, 0.005)
                    points[idx] = [x, y]
                    idx += 1

        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points

    def create_ring_initialization(self) -> np.ndarray:
        """Create a concentric ring-like arrangement."""
        points = np.zeros((16, 2))
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 4)
        layer_points = [4, 4, 4, 4]

        idx = 0
        for i, radius in enumerate(radii):
            num_points_in_layer = layer_points[i]
            layer_angles = np.linspace(0, 2*np.pi, num_points_in_layer, endpoint=False)
            for angle in layer_angles:
                if idx < 16:
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points[idx] = [x, y]
                    idx += 1
        return points

    def create_fibonacci_initialization(self) -> np.ndarray:
        """Create a Fibonacci-like arrangement for better point distribution."""
        points = np.zeros((16, 2))
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            r = np.sqrt(i / 15.0)
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points[i] = [x, y]
        return points

    def create_grid_initialization(self) -> np.ndarray:
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    points[idx] = [x, y]
                    idx += 1
        return points

    def create_random_initialization(self) -> np.ndarray:
        """Create a random initialization."""
        np.random.seed(self.seed)
        return np.random.rand(16, 2)

    def perturb_points(self, points, perturbation_magnitude=0.015) -> np.ndarray:
        """Apply random perturbation to points and ensure bounds."""
        perturbed = points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def _local_optimization_step(self, initial_points, max_iter=300) -> np.ndarray:
        """Perform local optimization refinement."""
        initial_flat = initial_points.flatten()
        bounds = [(0, 1) for _ in range(len(initial_flat))]
        
        try:
            result = minimize(
                lambda flat_points: self.objective_function(flat_points.reshape(-1, 2)),
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            optimized_points = result.x.reshape(-1, 2)
            return np.clip(optimized_points, 0, 1)
        except Exception:
            return initial_points

    def _global_optimization_step(self, initial_points, max_iter=200) -> np.ndarray:
        """Perform global optimization using differential evolution."""
        flat_points = initial_points.flatten()
        bounds = [(0, 1) for _ in range(len(flat_points))]
        
        try:
            de_result = differential_evolution(
                lambda x: self.objective_function(x.reshape(-1, 2)),
                bounds,
                seed=self.seed,
                maxiter=max_iter,
                popsize=25,
                mutation=(0.5, 1),
                recombination=0.7,
                tol=1e-8,
                disp=False
            )
            return de_result.x.reshape(-1, 2)
        except Exception:
            return initial_points

    def _adaptive_initialization_generation(self) -> List[Tuple[str, np.ndarray]]:
        """Generate diverse initial configurations with strategic diversity."""
        configs = []
        
        # Core initialization strategies
        strategies = [
            ("hexagonal", self.create_hexagonal_initialization()),
            ("symmetry_broken", self.create_symmetry_broken_initialization()),
            ("ring", self.create_ring_initialization()),
            ("fibonacci", self.create_fibonacci_initialization()),
            ("grid", self.create_grid_initialization()),
            ("random", self.create_random_initialization())
        ]
        
        # Add perturbed versions for diversity
        for name, base_config in strategies:
            # Only create a few perturbed versions to avoid explosion
            for perturbation_level in [0.01, 0.02, 0.03]:
                perturbed = self.perturb_points(base_config, perturbation_level)
                configs.append((f"{name}_perturbed_{perturbation_level}", perturbed))
        
        return configs[:12]  # Limit to manageable number

    def _selective_optimization(self, configurations: List[Tuple[str, np.ndarray]]) -> Tuple[float, np.ndarray]:
        """Run optimization on configurations with intelligent filtering."""
        best_ratio = -np.inf
        best_points = None
        evaluated_count = 0
        
        # Process configurations in batches
        for name, initial_config in configurations:
            try:
                evaluated_count += 1
                # Run sequential optimization stages
                global_result = self._global_optimization_step(initial_config, max_iter=150)
                local_result = self._local_optimization_step(global_result, max_iter=250)
                
                # Evaluate final result
                ratio = self.calculate_min_max_ratio(local_result)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = local_result
                    
            except Exception:
                continue
                
        return best_ratio, best_points

    def optimize(self) -> np.ndarray:
        """Main optimization routine with hierarchical approach."""
        # Generate initial configurations
        configurations = self._adaptive_initialization_generation()
        
        # Perform optimization with intelligent selection
        best_ratio, best_points = self._selective_optimization(configurations)
        
        # Fallback if nothing worked
        if best_points is None:
            fallback_config = self.create_hexagonal_initialization()
            best_points = self._local_optimization_step(fallback_config, max_iter=400)
            
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointArrangementOptimizer(seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END