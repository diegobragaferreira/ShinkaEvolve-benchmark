# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
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

    def _optimize_single_start(self, initial_points):
        """Perform optimization from single starting configuration."""
        # Flatten points for optimization
        flat_initial = initial_points.flatten()

        # Define bounds for each coordinate [0,1]
        bounds = [(0, 1) for _ in range(len(flat_initial))]

        # Constraints for boundary conditions
        constraints = {'type': 'ineq', 'fun': self._constraint_function}

        # Optimize using L-BFGS-B
        try:
            result = minimize(
                lambda x: -self._objective_function(x),  # Negative because we maximize
                flat_initial,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
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

    def generate_initial_points(self):
        """Generate initial point configurations."""
        # Generate multiple random initial configurations
        configurations = []

        # Grid-based initialization for better coverage
        grid_points = self._generate_grid_points()
        configurations.append(grid_points.copy())

        # Random initialization
        random_points = np.random.rand(self.n_points, self.dimensions)
        configurations.append(random_points.copy())

        # Hexagonal-like pattern
        hex_points = self._generate_hexagon_points()
        configurations.append(hex_points.copy())

        return configurations

    def _generate_grid_points(self):
        """Generate points in a grid pattern."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points

    def _generate_hexagon_points(self):
        """Generate points in a more sophisticated hexagonal pattern."""
        import math

        # Mathematical constants for proper hexagonal packing
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2  # Vertical spacing between rows
        col_spacing = 1.0        # Horizontal spacing between columns

        points = []

        # Create hexagonal lattice pattern (4x4 grid for 16 points)
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Proper hexagonal offset for alternating rows
                x = j * col_spacing + (i % 2) * 0.5
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:self.n_points])

        # Normalize to fit within [0,1] bounds properly
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply systematic perturbations to break symmetry effectively
        # Different noise intensities based on position index and mathematical pattern
        for i in range(self.n_points):
            # Apply non-uniform noise to break various symmetries
            # Use sine/cosine patterns to create structured yet asymmetric perturbations
            noise_intensity = 0.01 + 0.005 * math.sin(i * 0.5)
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]

        # Clip to ensure all points stay within valid bounds
        points = np.clip(points, 0, 1)

        return points

    def optimize(self):
        """Main optimization routine."""
        best_points = None
        best_ratio = -np.inf

        # Try multiple initial configurations
        initial_configs = self.generate_initial_points()

        for i, initial_config in enumerate(initial_configs):
            try:
                optimized_points, ratio = self._optimize_single_start(initial_config)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                warnings.warn(f"Error in optimization round {i}: {str(e)}")
                continue

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
    optimizer = PointDispersionOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END