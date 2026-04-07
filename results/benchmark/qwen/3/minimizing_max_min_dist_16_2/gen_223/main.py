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

    def boundary_penalty_function(self, x_flat: np.ndarray, penalty_weight: float = 100.0) -> float:
        """Add penalty for points near boundaries to encourage interior placement"""
        points = x_flat.reshape(-1, 2)
        # Calculate distance to nearest boundary for each point
        distances_to_boundaries = np.minimum(
            np.minimum(points[:, 0], 1 - points[:, 0]),
            np.minimum(points[:, 1], 1 - points[:, 1])
        )
        # Apply penalty for being too close to boundaries
        penalty = np.sum(penalty_weight * (1 - distances_to_boundaries)**2)
        return penalty

class InitializationStrategies:
    """Collection of different initialization strategies."""

    @staticmethod
    def hexagonal_grid_init() -> np.ndarray:
        """Create a proper hexagonal lattice arrangement with enhanced precision"""
        # Create regular hexagonal packing with precise mathematical construction
        # For 16 points, we'll use a 4x4 arrangement but optimize positioning

        # Calculate the ideal hexagonal grid spacing for maximum packing density
        # The goal is to place points so that they form a nearly equilateral triangle pattern
        sqrt3 = np.sqrt(3)
        spacing = 1.0  # Base spacing parameter
        row_spacing = spacing * sqrt3 / 2.0  # Vertical spacing between rows
        col_spacing = spacing  # Horizontal spacing

        points = []

        # Create a hexagonal grid with sufficient points to allow for optimization
        # 4x4 grid gives us 16 points arranged in a hexagonal pattern
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Position according to hexagonal pattern
                x = j * col_spacing + (i % 2) * col_spacing / 2.0  # Offset odd rows
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array and normalize properly
        points = np.array(points[:16])

        # Normalize to fit perfectly within [0,1] x [0,1]
        # Preserve the hexagonal structure while fitting bounds
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0 and y_range > 0:
                # Scale to fit nicely in the unit square
                scale_factor = min(1.0 / x_range, 1.0 / y_range)
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * scale_factor
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * scale_factor

                # Center in unit square
                center_shift = 0.5 - np.mean(points, axis=0)
                points = points + center_shift

        # Ensure bounds
        points = np.clip(points, 0, 1)

        # Apply superior symmetry breaking with mathematical precision:
        # 1. Apply carefully chosen rotations to break rotational symmetry
        # 2. Apply position-dependent perturbations with mathematical basis
        center = np.mean(points, axis=0)

        # Apply rotations using golden ratio based angles for better distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        rotation_angles = [
            0.0,
            np.pi / golden_ratio,          # ~1.38 rad (~79 deg)
            np.pi / (golden_ratio * 2),    # ~0.69 rad (~39 deg)
            np.pi / (golden_ratio * 3),    # ~0.46 rad (~26 deg)
        ]

        # Apply rotations systematically
        np.random.seed(42)
        for i in range(len(points)):
            # Apply rotation based on position to prevent symmetric solutions
            rotation_group = i % 4
            if rotation_group < len(rotation_angles):
                angle = rotation_angles[rotation_group]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                points[i] = rotation_matrix @ (points[i] - center) + center

        # Apply sophisticated position-based perturbations with mathematical basis
        # These perturbations are designed to spread points while maintaining hexagonal structure
        np.random.seed(42)
        for i in range(len(points)):
            # Calculate distance from center to determine perturbation magnitude
            distance_from_center = np.linalg.norm(points[i] - center)
            max_dist = np.sqrt(2)  # Max possible distance in [0,1]x[0,1]
            normalized_dist = distance_from_center / max_dist if max_dist > 0 else 0

            # More sophisticated perturbation scheme:
            # 1. Base perturbation varies with distance from center (more at edges)
            # 2. Add harmonic components to avoid regular patterns
            base_perturbation = 0.01 * (0.5 + 0.5 * normalized_dist)  # Larger at edges

            # Add sine wave modulation based on angular position to break symmetry
            angle_from_center = np.arctan2(points[i][1] - center[1], points[i][0] - center[0])
            angular_modulation = 0.003 * np.sin(3 * angle_from_center)  # Triple frequency modulation

            # Apply perturbations
            perturbation_x = np.random.normal(0, base_perturbation, 1)[0] + angular_modulation
            perturbation_y = np.random.normal(0, base_perturbation, 1)[0] + angular_modulation

            points[i][0] += perturbation_x
            points[i][1] += perturbation_y

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
        """Create adaptive hexagonal grid with boundary awareness"""
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
            # Introduce a non-linear scaling for more interesting behavior
            center_weight = 0.3 + 0.7 * (1 - normalized_distances)
            boundary_weight = 0.5 + 0.5 * boundary_distances_normalized
            perturbation_magnitude = 0.01 * center_weight * boundary_weight

            np.random.seed(42)
            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations

        return np.clip(points, 0, 1)

    @staticmethod
    def simulated_annealing_init() -> np.ndarray:
        """Create initial configuration using a simplified simulated annealing approach"""
        # Start with a regular hexagonal grid
        points = InitializationStrategies.hexagonal_grid_init()

        # Apply a few rounds of simulated annealing style perturbations
        np.random.seed(42)

        # Temperature schedule parameters
        initial_temp = 0.1
        cooling_rate = 0.99
        min_temp = 0.001
        steps_per_temp = 10

        temp = initial_temp

        for step in range(100):
            if temp < min_temp:
                break

            # Try to make random moves
            test_points = points.copy()

            # Pick random point to move
            idx = np.random.randint(0, len(test_points))
            # Move it slightly
            test_points[idx] += np.random.normal(0, temp * 0.01, 2)

            # Keep within bounds
            test_points = np.clip(test_points, 0, 1)

            # Accept or reject based on energy difference (simplified version)
            # We don't actually compute energy here since we're just trying to avoid local minima
            points = test_points

            # Cool down
            temp *= cooling_rate

        return points

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
            InitializationStrategies.generate_initial_config,
            InitializationStrategies.simulated_annealing_init
        ]
        self.bounds = [(0, 1) for _ in range(32)]

    def optimize_with_differential_evolution(self, x0: np.ndarray, restart: int) -> Optional[np.ndarray]:
        """Attempt optimization using differential evolution"""
        try:
            # Adjust parameters based on restart number for efficiency
            if restart < 2:
                # More thorough optimization for early restarts
                result = differential_evolution(
                    self.config.objective_function,
                    self.bounds,
                    maxiter=800,
                    popsize=20,
                    tol=1e-8,
                    seed=42 + restart,
                    callback=None
                )
            elif restart < 4:
                # Medium thoroughness
                result = differential_evolution(
                    self.config.objective_function,
                    self.bounds,
                    maxiter=500,
                    popsize=15,
                    tol=1e-7,
                    seed=42 + restart,
                    callback=None
                )
            else:
                # Faster optimization for later restarts
                result = differential_evolution(
                    self.config.objective_function,
                    self.bounds,
                    maxiter=300,
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
        """Attempt optimization using local search method with adaptive parameters"""
        try:
            # Use different optimization settings based on how many attempts we've made
            # This helps with convergence to better local optima
            result = minimize(
                self.config.objective_function,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints={'type': 'ineq', 'fun': self.config.constraint_function},
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-5}
            )

            if result.success:
                return result.x.reshape(-1, 2)
        except Exception:
            pass
        return None

    def optimize_with_adaptive_local_search(self, x0: np.ndarray, iteration: int = 0) -> Optional[np.ndarray]:
        """Adaptive local search with dynamic parameters based on iteration"""
        try:
            # Adaptively choose optimization parameters based on iteration
            maxiter = 1000 if iteration < 2 else 500
            ftol = 1e-8 if iteration < 2 else 1e-6
            eps = 1e-5 if iteration < 2 else 1e-4

            # For later iterations, also try different methods
            methods = ['SLSQP', 'L-BFGS-B']
            method = methods[iteration % len(methods)] if iteration >= 2 else 'SLSQP'

            result = minimize(
                self.config.objective_function,
                x0,
                method=method,
                bounds=self.bounds,
                constraints={'type': 'ineq', 'fun': self.config.constraint_function},
                options={'maxiter': maxiter, 'ftol': ftol, 'eps': eps}
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

        num_restarts = 7  # Increased number of restarts for better exploration
        for restart in range(num_restarts):
            # Select initialization strategy
            init_func = self.strategies[restart % len(self.strategies)]
            points = init_func()

            # Flatten for optimization
            x0 = points.flatten()

            # Try differential evolution first (global search) with adaptive parameters
            optimized_points = self.optimize_with_differential_evolution(x0, restart)

            # If DE failed, try adaptive local optimization
            if optimized_points is None:
                optimized_points = self.optimize_with_adaptive_local_search(x0, restart)

            # Check results
            if optimized_points is not None:
                current_ratio = self.config.compute_min_max_ratio(optimized_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()

        # Additional refinement step with more aggressive optimization
        if best_points is not None:
            # Try one final intense optimization
            x0 = best_points.flatten()
            final_points = self.optimize_with_adaptive_local_search(x0, 10)
            if final_points is not None:
                final_ratio = self.config.compute_min_max_ratio(final_points)
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()

        # Fallback if all optimizations failed
        if best_points is None:
            points = InitializationStrategies.hexagonal_grid_init()
            x0 = points.flatten()
            best_points = self.optimize_with_adaptive_local_search(x0, 0)

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