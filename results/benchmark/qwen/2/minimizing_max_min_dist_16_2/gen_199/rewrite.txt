# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
from typing import List, Tuple, Optional, Callable
import time

class PointOptimizer:
    """
    Structured evolutionary optimizer for maximizing min/max distance ratio of 16 points in 2D.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.bounds = [(0.001, 0.999) for _ in range(32)]
        self.best_ratio = -np.inf
        self.best_points = None
        
    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative of min/max ratio)."""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        
        if len(distances) == 0 or np.max(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return -min_dist / max_dist
        
    def generate_hexagonal_initial(self) -> np.ndarray:
        """Generate initial configuration based on hexagonal lattice."""
        points = []
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)

        for i in range(rows):
            for j in range(cols):
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.5
                x = (j * spacing_x) + x_offset
                y = i * spacing_y

                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                points.append([x, y])

        return np.array(points)
        
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral for good distribution."""
        points = []
        phi = (1 + math.sqrt(5)) / 2
        
        for i in range(16):
            theta = math.acos(-1 + (2 * i) / 15)
            phi_angle = (i * 2 * math.pi) / (phi * phi)

            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)

            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)
        
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid initial configuration with symmetry breaking."""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        points[0] = [0.05, 0.05]      # Bottom-left corner
        points[15] = [0.95, 0.95]     # Top-right corner
        points[3] = [0.95, 0.05]      # Bottom-right corner
        points[12] = [0.05, 0.95]     # Top-left corner

        return np.array(points)
        
    def generate_adaptive_perturbation(self, base_points: np.ndarray, 
                                     iteration: int = 0, 
                                     magnitude_scale: float = 1.0) -> np.ndarray:
        """Generate perturbed points with adaptive magnitude."""
        base_magnitude = 0.05 * (1.0 - iteration * 0.1) * magnitude_scale
        base_magnitude = max(0.005, base_magnitude)

        perturbation = np.random.normal(0, base_magnitude, base_points.shape)
        perturbed_points = base_points + perturbation
        perturbed_points = np.clip(perturbed_points, 0.001, 0.999)
        return perturbed_points
        
    def create_diverse_initial_configs(self) -> List[np.ndarray]:
        """Create multiple diverse initial configurations."""
        initial_configs = [
            self.generate_hexagonal_initial(),
            self.generate_fibonacci_spiral(),
            self.generate_regular_grid()
        ]
        
        # Add perturbed versions with enhanced diversity
        for i, config in enumerate(initial_configs[:]):
            for j in range(4):
                np.random.seed(self.seed + i * 10 + j)
                perturbed = config.copy()
                
                if j == 0:
                    perturbation = np.random.normal(0, 0.01, config.shape)
                elif j == 1:
                    perturbation = np.random.normal(0, 0.03, config.shape)
                elif j == 2:
                    perturbation = np.random.normal(0, 0.015, config.shape)
                    center = np.array([0.5, 0.5])
                    for k in range(len(perturbed)):
                        perturbed[k] += (center - perturbed[k]) * 0.02
                else:
                    perturbation = np.random.normal(0, 0.04, config.shape)

                perturbed += perturbation
                perturbed = np.clip(perturbed, 0.001, 0.999)
                initial_configs.append(perturbed)
                
        return initial_configs
        
    def global_exploration_stage(self) -> Optional[np.ndarray]:
        """Stage 1: Global search with differential evolution."""
        try:
            de_result = differential_evolution(
                self.objective,
                self.bounds,
                maxiter=20,
                popsize=5,
                seed=self.seed,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            # Local refinement
            local_result = minimize(
                self.objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if local_result.success:
                return local_result.x.reshape(-1, 2)
                
        except Exception:
            return None
            
        return None
        
    def multi_start_refinement_stage(self, initial_configs: List[np.ndarray]) -> None:
        """Stage 2: Multi-start local optimization from various initial points."""
        for i, initial_config in enumerate(initial_configs[:10]):
            np.random.seed(self.seed + i)
            perturbed_config = initial_config + np.random.normal(0, 0.01, initial_config.shape)
            perturbed_config = np.clip(perturbed_config, 0.001, 0.999)

            try:
                result = minimize(
                    self.objective,
                    perturbed_config.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    distances = pdist(final_points)
                    
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > self.best_ratio:
                                self.best_ratio = ratio
                                self.best_points = final_points.copy()
                                
            except Exception:
                continue
                
    def targeted_search_stage(self) -> None:
        """Stage 3: Targeted grid search for improvement."""
        if self.best_points is None or self.best_ratio < 0.25:
            test_grid = np.linspace(0.1, 0.9, 5)
            for i in range(5):
                for j in range(5):
                    base_x = test_grid[i]
                    base_y = test_grid[j]

                    np.random.seed(self.seed + i * 5 + j)
                    base_points = np.array([[base_x, base_y]] * 16)
                    perturbation = np.random.normal(0, 0.03, (16, 2))
                    perturbed_points = base_points + perturbation
                    perturbed_points = np.clip(perturbed_points, 0.001, 0.999)

                    try:
                        result = minimize(
                            self.objective,
                            perturbed_points.flatten(),
                            method='L-BFGS-B',
                            bounds=self.bounds,
                            options={'maxiter': 50, 'ftol': 1e-8}
                        )

                        if result.success:
                            final_points = result.x.reshape(-1, 2)
                            distances = pdist(final_points)
                            min_dist = np.min(distances)
                            max_dist = np.max(distances)
                            
                            if max_dist > 0:
                                ratio = min_dist / max_dist
                                if ratio > self.best_ratio:
                                    self.best_ratio = ratio
                                    self.best_points = final_points.copy()
                                    
                    except Exception:
                        continue
                        
    def evaluate_solution(self, points: np.ndarray) -> float:
        """Evaluate solution and return ratio."""
        distances = pdist(points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            return min_dist / max_dist if max_dist > 0 else 0
        return 0
        
    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        # Global exploration stage
        global_result = self.global_exploration_stage()
        if global_result is not None:
            ratio = self.evaluate_solution(global_result)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_points = global_result.copy()
        
        # Create diverse initial configurations
        initial_configs = self.create_diverse_initial_configs()
        
        # Multi-start refinement stage
        self.multi_start_refinement_stage(initial_configs)
        
        # Targeted search stage
        self.targeted_search_stage()
        
        # Fallback to best initial configuration if no improvement found
        if self.best_points is None:
            fallback_config = self.generate_regular_grid()
            np.random.seed(self.seed)
            fallback_points = fallback_config + np.random.normal(0, 0.01, fallback_config.shape)
            fallback_points = np.clip(fallback_points, 0.001, 0.999)
            self.best_points = fallback_points
            
        return self.best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer(seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END