# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import math
from typing import List, Tuple, Optional
import time

class PointConfiguration:
    """Handles point configuration generation and manipulation."""

    def __init__(self, n_points: int = 16, dimensions: int = 2, seed: int = 42):
        self.n_points = n_points
        self.dimensions = dimensions
        np.random.seed(seed)

    def generate_grid_points(self) -> np.ndarray:
        """Generate points in a grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points

    def generate_perturbed_grid_points(self) -> np.ndarray:
        """Generate a perturbed grid to break symmetries."""
        grid_points = self.generate_grid_points()
        # Add small random perturbations
        perturbation_magnitude = 0.05
        perturbed = grid_points + np.random.normal(0, perturbation_magnitude, grid_points.shape)
        # Clip to bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def generate_hexagonal_points(self) -> np.ndarray:
        """Generate points in a hexagonal pattern with improved triangular lattice."""
        points = []

        # Create a more precise hexagonal lattice for 16 points
        # Using 4 rows and 4 columns with proper triangular packing

        # Calculate optimal spacing for 16 points in unit square
        # For triangular packing, we want spacing ~ 1/sqrt(16) * sqrt(3/2) ≈ 0.433
        spacing = 0.4
        row_spacing = spacing * np.sqrt(3) / 2
        col_spacing = spacing

        # Generate hexagonal grid with proper offsetting
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    # Hexagonal packing with proper offsetting
                    x = j * col_spacing + (i % 2) * col_spacing / 2
                    y = i * row_spacing

                    # Add systematic asymmetry to break symmetries
                    # Use different noise patterns based on position
                    asym_factor = 0.02
                    if (i + j) % 3 == 0:
                        asym_factor *= 2.0  # Stronger asymmetry for some points
                    elif (i + j) % 2 == 0:
                        asym_factor *= 1.5  # Medium asymmetry

                    # Add asymmetric noise
                    noise_x = np.random.normal(0, asym_factor * spacing)
                    noise_y = np.random.normal(0, asym_factor * spacing)

                    # Apply different noise patterns for different positions
                    if (i * j) % 4 == 0:
                        noise_x *= 1.2
                        noise_y *= 1.2

                    x += noise_x
                    y += noise_y

                    points.append([x, y])

        # Normalize coordinates to [0,1] range
        points = np.array(points[:self.n_points])
        if len(points) > 0:
            # Center the points
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            points[:, 0] -= center_x
            points[:, 1] -= center_y

            # Scale to fit nicely in [0,1] range
            max_extent = max(np.max(np.abs(points[:, 0])), np.max(np.abs(points[:, 1])))
            if max_extent > 0:
                points[:, 0] /= max_extent * 2
                points[:, 1] /= max_extent * 2

            # Shift and scale to [0.1, 0.9] range to avoid edge effects
            points[:, 0] = np.clip(points[:, 0] * 0.8 + 0.45, 0.1, 0.9)
            points[:, 1] = np.clip(points[:, 1] * 0.8 + 0.45, 0.1, 0.9)

        return points

    def generate_random_points(self) -> np.ndarray:
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def generate_all_configurations(self) -> List[np.ndarray]:
        """Generate all types of initial configurations."""
        configurations = [
            self.generate_grid_points(),
            self.generate_perturbed_grid_points(),
            self.generate_hexagonal_points(),
            self.generate_random_points()
        ]
        return configurations

class FitnessEvaluator:
    """Evaluates the quality of point configurations."""

    @staticmethod
    def compute_ratio(points: np.ndarray) -> float:
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max

class OptimizationStrategy:
    """Base class for optimization strategies."""

    def optimize(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

class LBFGSBOptimizer(OptimizationStrategy):
    """L-BFGS-B optimization strategy."""

    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations

    def optimize(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize using L-BFGS-B method."""
        def objective(x):
            # Reshape x into points array
            reshaped_points = x.reshape(-1, 2)

            # Compute pairwise distances
            distances = pdist(reshaped_points)

            # Avoid division by zero
            if len(distances) == 0:
                return 0

            # Calculate min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Return negative ratio to maximize (since minimize minimizes)
            if d_max == 0:
                return 0
            return -d_min / d_max

        # Flatten initial guess
        x0 = points.flatten()

        # Set up bounds (each coordinate must be between 0 and 1)
        bounds = [(0, 1) for _ in range(len(x0))]

        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                return optimized_points, -result.fun
            else:
                warnings.warn(f"L-BFGS-B optimization failed: {result.message}")
                return points, FitnessEvaluator.compute_ratio(points)

        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization error: {str(e)}")
            return points, FitnessEvaluator.compute_ratio(points)

class SimulatedAnnealingOptimizer(OptimizationStrategy):
    """Simulated Annealing optimization strategy."""

    def __init__(self, max_iterations: int = 5000, initial_temp: float = 1.0, cooling_rate: float = 0.9995):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def optimize(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize using Simulated Annealing."""
        def compute_ratio(points_array):
            """Compute the actual ratio for given points"""
            distances = pdist(points_array)
            if len(distances) == 0:
                return 0
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max == 0:
                return 0
            return d_min / d_max

        current_points = points.copy()
        current_ratio = compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        temp = self.initial_temp

        for iteration in range(self.max_iterations):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))

            # Perturb the selected point with adaptive step size
            step_size = 0.02 if iteration < self.max_iterations//2 else 0.005
            neighbor_points[idx, 0] += np.random.normal(0, step_size)
            neighbor_points[idx, 1] += np.random.normal(0, step_size)

            # Keep within bounds
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)

            # Calculate neighbor ratio
            neighbor_ratio = compute_ratio(neighbor_points)

            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if delta < 0:  # Only accept worse solutions with probability
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio

            # Cool down
            temp *= self.cooling_rate

            # Early stopping condition
            if temp < 1e-8:
                break

        return best_points, best_ratio

class EvolutionaryPointOptimizer:
    """Main optimizer that orchestrates multiple strategies."""

    def __init__(self, n_points: int = 16, dimensions: int = 2, seed: int = 42, max_time_seconds: int = 180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        self.config_generator = PointConfiguration(n_points, dimensions, seed)
        self.fitness_evaluator = FitnessEvaluator()
        self.lbfgsb_optimizer = LBFGSBOptimizer()
        self.sa_optimizer = SimulatedAnnealingOptimizer()

    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        start_time = time.time()
        best_points = None
        best_ratio = -np.inf

        # Generate all initial configurations
        configurations = self.config_generator.generate_all_configurations()

        # Try each configuration with multiple optimization strategies
        for i, initial_config in enumerate(configurations):
            if time.time() - start_time > self.max_time_seconds - 5:
                break

            try:
                # Strategy 1: L-BFGS-B optimization
                lbfgsb_points, lbfgsb_ratio = self.lbfgsb_optimizer.optimize(initial_config)

                # Strategy 2: Simulated Annealing refinement (only if L-BFGS-B gave reasonable results)
                if lbfgsb_ratio > 0.1:  # Only refine if it's a decent solution
                    sa_points, sa_ratio = self.sa_optimizer.optimize(lbfgsb_points)
                    optimized_points = sa_points if sa_ratio > lbfgsb_ratio else lbfgsb_points
                    ratio = max(sa_ratio, lbfgsb_ratio)
                else:
                    optimized_points = lbfgsb_points
                    ratio = lbfgsb_ratio

                # Update best solution
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue

        # Final validation
        if best_points is not None:
            final_ratio = self.fitness_evaluator.compute_ratio(best_points)
            print(f"Final optimized ratio: {final_ratio:.6f}")
            return best_points
        else:
            # Fallback to the first configuration if nothing worked
            return configurations[0] if configurations else np.random.rand(self.n_points, self.dimensions)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    optimizer = EvolutionaryPointOptimizer(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END