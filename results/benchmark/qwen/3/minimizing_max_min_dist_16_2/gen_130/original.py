# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import time

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, n_points=16, dimensions=2, seed=42, max_time_seconds=180):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)

    def _compute_distances(self, points):
        """Compute pairwise distances between points."""
        if len(points.shape) == 1:
            points = points.reshape(-1, self.dimensions)
        distances = pdist(points)
        return distances

    def _objective_function(self, flat_points):
        """Objective function to maximize min/max distance ratio."""
        points = flat_points.reshape(-1, self.dimensions)
        distances = self._compute_distances(points)

        if len(distances) == 0:
            return -np.inf

        min_distance = np.min(distances)
        max_distance = np.max(distances)

        # Avoid division by zero
        if max_distance <= 1e-12:
            return -np.inf

        return min_distance / max_distance

    def _constraint_function(self, flat_points):
        """Constraint function to keep points within unit square."""
        points = flat_points.reshape(-1, self.dimensions)
        # Ensure all coordinates are in [0,1]
        return np.concatenate([
            points.flatten() - 1,  # x_i - 1 <= 0
            -points.flatten()     # x_i >= 0
        ])

    def _optimize_single_start(self, initial_points, max_iter=1000):
        """Perform optimization from single starting configuration."""
        # Flatten points for optimization
        flat_initial = initial_points.flatten()

        # Define bounds for each coordinate [0,1]
        bounds = [(0, 1) for _ in range(len(flat_initial))]

        # Constraints for boundary conditions
        constraints = {'type': 'ineq', 'fun': self._constraint_function}

        # Optimize using L-BFGS-B with tighter tolerances
        try:
            result = minimize(
                lambda x: -self._objective_function(x),  # Negative because we maximize
                flat_initial,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                return optimized_points, self._objective_function(result.x)
            else:
                warnings.warn(f"Optimization failed: {result.message}")
                return initial_points, self._objective_function(flat_initial)

        except Exception as e:
            warnings.warn(f"Optimization error: {str(e)}")
            return initial_points, self._objective_function(flat_initial)

    def _generate_grid_points(self):
        """Generate points in a grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points

    def _generate_hexagon_points(self):
        """Generate points in a hexagonal pattern."""
        # Create a hexagonal lattice pattern using triangular packing
        points = []
        center_x, center_y = 0.5, 0.5

        # Use triangular (hexagonal) packing with optimal spacing
        # For 16 points in a square, we want roughly equal spacing
        spacing = 0.3
        radius = spacing / np.sqrt(3)  # Correct spacing for triangular packing

        # Generate 4x4 hexagonal grid
        rows = 4
        cols = 4
        y_spacing = radius * np.sqrt(3)
        x_spacing = radius * 2

        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = center_x - (cols-1)*x_spacing/2 + j*x_spacing
                    # Offset every other row for hexagonal packing
                    if i % 2 == 1:
                        x += x_spacing/2
                    y = center_y - (rows-1)*y_spacing/2 + i*y_spacing
                    # Add some randomness to avoid perfect symmetry
                    x += np.random.normal(0, radius * 0.1)
                    y += np.random.normal(0, radius * 0.1)
                    # Clip to ensure within bounds
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    points.append([x, y])

        # Fill remaining points with random distribution
        while len(points) < self.n_points:
            points.append([np.random.rand(), np.random.rand()])

        return np.array(points[:self.n_points])

    def _generate_perturbed_grid_points(self):
        """Generate a perturbed grid to break symmetries."""
        grid_points = self._generate_grid_points()
        # Add small random perturbations
        perturbation_magnitude = 0.05
        perturbed = grid_points + np.random.normal(0, perturbation_magnitude, grid_points.shape)
        # Clip to bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def _generate_random_points(self):
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)

    def _generate_fibonacci_sphere_points(self):
        """Generate points using Fibonacci sphere algorithm (adapted for 2D)."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            # Map to 2D unit square
            x_mapped = (x + 1) / 2
            y_mapped = (z + 1) / 2

            points.append([np.clip(x_mapped, 0, 1), np.clip(y_mapped, 0, 1)])

        return np.array(points)

    def generate_initial_points(self):
        """Generate multiple diverse initial point configurations."""
        configurations = []

        # Grid-based initialization
        configurations.append(self._generate_grid_points())

        # Perturbed grid initialization
        configurations.append(self._generate_perturbed_grid_points())

        # Hexagonal-like pattern
        configurations.append(self._generate_hexagon_points())

        # Random initialization
        configurations.append(self._generate_random_points())

        # Fibonacci-inspired pattern
        configurations.append(self._generate_fibonacci_sphere_points())

        return configurations

    def optimize(self):
        """Main optimization routine with multi-start approach."""
        best_points = None
        best_ratio = -np.inf
        start_time = time.time()

        # Generate multiple initial configurations
        initial_configs = self.generate_initial_points()

        # Try multiple initial configurations
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 5:  # Leave buffer for final processing
                break

            try:
                optimized_points, ratio = self._optimize_single_start(initial_config)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue

        # If we haven't found a good solution yet, try more aggressive optimization
        if best_ratio < 0.1 and time.time() - start_time < self.max_time_seconds - 10:
            # Try additional optimization with higher iteration limits
            if best_points is not None:
                try:
                    # More intensive optimization
                    optimized_points, ratio = self._optimize_single_start(best_points, max_iter=2000)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                except Exception as e:
                    warnings.warn(f"Error in intensive optimization: {str(e)}")

        # Final validation
        if best_points is not None:
            final_ratio = self._objective_function(best_points.flatten())
            print(f"Final optimized ratio: {final_ratio:.6f}")
            return best_points
        else:
            # Return the last attempted configuration if nothing worked
            return initial_configs[-1] if initial_configs else np.random.rand(self.n_points, self.dimensions)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16, dimensions=2, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END