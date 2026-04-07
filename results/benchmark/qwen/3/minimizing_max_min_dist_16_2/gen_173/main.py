# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
import random
from typing import Tuple, Callable, List, Optional

class PointConfiguration:
    """Handles point configuration generation and optimization."""

    def __init__(self):
        self.points = None
        self.ratio = -np.inf

    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist

    def objective_function(self, x_flat: np.ndarray) -> float:
        """Objective function to maximize (negative because we minimize)"""
        points = x_flat.reshape(-1, 2)
        return -self.compute_min_max_ratio(points)

    def constraint_function(self, x_flat: np.ndarray) -> np.ndarray:
        """Constraint to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        violations = np.concatenate([
            np.minimum(points[:, 0], 0),
            np.minimum(points[:, 1], 0),
            np.maximum(points[:, 0] - 1, 0),
            np.maximum(points[:, 1] - 1, 0)
        ])
        return violations

class InitializationStrategies:
    """Collection of different initialization strategies."""

    @staticmethod
    def hexagonal_grid_init() -> np.ndarray:
        """Create a proper hexagonal lattice arrangement"""
        points = []

        # Parameters for hexagonal lattice
        hex_spacing = 1.0
        row_spacing = hex_spacing * np.sqrt(3) / 2.0
        col_spacing = hex_spacing

        # Place points in hexagonal pattern (4 rows, 4 columns)
        for row in range(4):
            for col in range(4):
                if len(points) >= 16:
                    break
                x = col * col_spacing
                if row % 2 == 1:
                    x += col_spacing / 2.0
                y = row * row_spacing
                points.append([x, y])

        # Convert to numpy array and normalize
        points = np.array(points[:16])

        # Normalize to fit within unit square
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        if max_x > min_x and max_y > min_y:
            scale_x = 1.0 / (max_x - min_x)
            scale_y = 1.0 / (max_y - min_y)
            scale = min(scale_x, scale_y, 1.0)

            points[:, 0] = (points[:, 0] - min_x) * scale
            points[:, 1] = (points[:, 1] - min_y) * scale

        # Center the points
        center_shift = 0.5 - np.mean(points, axis=0)
        points = points + center_shift

        # Ensure bounds
        points = np.clip(points, 0, 1)

        # Apply enhanced symmetry breaking:
        # 1. Apply deterministic rotations to break rotational symmetry
        # 2. Apply position-specific perturbations to break translational symmetry
        center = np.mean(points, axis=0)

        # Apply deterministic rotations to create asymmetric pattern
        angles = [0, np.pi/6, np.pi/3, np.pi/2]  # 0, 30, 60, 90 degrees
        for i in range(len(points)):
            # Rotate every 4th point differently to break rotational symmetry
            if i % 4 == 0:
                angle = angles[i % len(angles)]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                points[i] = rotation_matrix @ (points[i] - center) + center

        # Apply position-based perturbations to break translational symmetry
        np.random.seed(42)
        for i in range(len(points)):
            # Different perturbation magnitudes based on position relative to center
            distance_from_center = np.linalg.norm(points[i] - center)
            max_dist = np.sqrt(2)  # Max possible distance in [0,1]x[0,1]
            normalized_dist = distance_from_center / max_dist if max_dist > 0 else 0

            # Points farther from center get larger perturbations
            pert_mag = 0.005 * (1 - normalized_dist) + 0.002  # Base + distance-dependent component

            # Apply perturbation
            points[i] += np.random.normal(0, pert_mag, 2)

        points = np.clip(points, 0, 1)

        return points

    @staticmethod
    def random_init() -> np.ndarray:
        """Generate random point configuration"""
        np.random.seed(42)
        return np.random.rand(16, 2)

    @staticmethod
    def perturbed_hexagonal_init() -> np.ndarray:
        """Create perturbed hexagonal grid"""
        points = InitializationStrategies.hexagonal_grid_init()
        np.random.seed(42)

        # Apply adaptive perturbations based on position
        center = np.mean(points, axis=0)
        distances_from_center = np.sqrt(np.sum((points - center)**2, axis=1))
        max_distance = np.max(distances_from_center)

        if max_distance > 0:
            normalized_distances = distances_from_center / max_distance
            # Points closer to center get smaller perturbations to preserve internal structure
            # Points farther from center get larger perturbations for better exploration
            perturbation_magnitude = 0.01 * (0.5 + 0.5 * normalized_distances)

            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations

        return np.clip(points, 0, 1)

    @staticmethod
    def adaptive_hexagonal_init() -> np.ndarray:
        """Create adaptive hexagonal grid"""
        points = InitializationStrategies.hexagonal_grid_init()

        center = np.mean(points, axis=0)
        distances_from_center = np.sqrt(np.sum((points - center)**2, axis=1))
        max_distance = np.max(distances_from_center)

        # Get distances to boundaries for boundary-aware perturbations
        distances_to_boundaries = np.minimum(
            np.minimum(points[:, 0], 1 - points[:, 0]),
            np.minimum(points[:, 1], 1 - points[:, 1])
        )

        if max_distance > 0:
            # Combine distance from center and distance to boundaries for adaptive perturbation
            normalized_distances = distances_from_center / max_distance
            boundary_distances_normalized = distances_to_boundaries / 0.5  # Normalize to [0,1]

            # Points that are far from center AND far from boundaries get larger perturbations
            # Points that are near boundaries get smaller perturbations to avoid going out of bounds
            perturbation_magnitude = 0.01 * (0.3 + 0.7 * (1 - normalized_distances)) * (0.5 + 0.5 * boundary_distances_normalized)

            np.random.seed(42)
            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations

        return np.clip(points, 0, 1)

    @staticmethod
    def generate_initial_config() -> np.ndarray:
        """Generate a better initial configuration"""
        points = []

        # Create a 4x4 grid with strategic spacing
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125 + random.uniform(-0.01, 0.01)
                y = i * 0.25 + random.uniform(-0.01, 0.01)
                points.append([x, y])

        points = np.array(points)
        points += np.random.normal(0, 0.005, (16, 2))
        return np.clip(points, 0, 1)

class OptimizationEngine:
    """Main optimization engine with adaptive strategies."""

    def __init__(self):
        self.config = PointConfiguration()
        self.strategies = [
            InitializationStrategies.hexagonal_grid_init,
            InitializationStrategies.random_init,
            InitializationStrategies.perturbed_hexagonal_init,
            InitializationStrategies.adaptive_hexagonal_init,
            InitializationStrategies.generate_initial_config
        ]
        self.bounds = [(0, 1) for _ in range(32)]

    def optimize_with_differential_evolution(self, x0: np.ndarray, restart: int) -> Optional[np.ndarray]:
        """Attempt optimization using differential evolution"""
        try:
            # Adjust parameters based on restart number for efficiency
            if restart < 3:
                result = differential_evolution(
                    self.config.objective_function,
                    self.bounds,
                    maxiter=500,
                    popsize=15,
                    tol=1e-8,
                    seed=42 + restart,
                    callback=None
                )
            else:
                result = differential_evolution(
                    self.config.objective_function,
                    self.bounds,
                    maxiter=200,
                    popsize=10,
                    tol=1e-6,
                    seed=42 + restart,
                    callback=None
                )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        return None

    def optimize_with_local_search(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Attempt optimization using local search method"""
        try:
            result = minimize(
                self.config.objective_function,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints={'type': 'ineq', 'fun': self.config.constraint_function},
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
            )

            if result.success:
                return result.x.reshape(-1, 2)
        except Exception:
            pass
        return None

    def execute_multi_start_optimization(self) -> np.ndarray:
        """Execute multi-start optimization with adaptive strategy selection"""
        best_ratio = -np.inf
        best_points = None

        num_restarts = 5
        for restart in range(num_restarts):
            # Select initialization strategy
            init_func = self.strategies[restart % len(self.strategies)]
            points = init_func()

            # Flatten for optimization
            x0 = points.flatten()

            # Try differential evolution first (global search)
            optimized_points = self.optimize_with_differential_evolution(x0, restart)

            # If DE failed, try local optimization
            if optimized_points is None:
                optimized_points = self.optimize_with_local_search(x0)

            # Check results
            if optimized_points is not None:
                current_ratio = self.config.compute_min_max_ratio(optimized_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()

        # Fallback if all optimizations failed
        if best_points is None:
            points = InitializationStrategies.hexagonal_grid_init()
            x0 = points.flatten()
            best_points = self.optimize_with_local_search(x0)

            if best_points is None:
                # Final fallback to random points
                np.random.seed(42)
                best_points = np.random.rand(16, 2)

        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    engine = OptimizationEngine()
    return engine.execute_multi_start_optimization()

# EVOLVE-BLOCK-END