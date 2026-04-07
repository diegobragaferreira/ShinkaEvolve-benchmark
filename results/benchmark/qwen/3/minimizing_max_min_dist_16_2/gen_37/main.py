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
        # Create a hexagonal lattice pattern with better geometric properties
        points = []
        center_x, center_y = 0.5, 0.5

        # Use a proper triangular lattice that maximizes minimum distances
        # For 16 points, arrange in a 4x4 triangular grid pattern
        rows = 4
        cols = 4

        # Calculate proper spacing for triangular packing
        # In a triangular lattice, nearest neighbors are separated by distance s
        # The area per point should be s^2 * sqrt(3)/2
        # For 16 points in unit square, we estimate optimal spacing
        total_area = 1.0  # Area of unit square
        points_per_unit_area = 16.0  # We have 16 points
        optimal_spacing = np.sqrt(total_area / points_per_unit_area) * 1.2  # Add some margin

        # Hexagonal packing with proper spacing
        y_spacing = optimal_spacing * np.sqrt(3)
        x_spacing = optimal_spacing * 2

        # Create triangular lattice points
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = center_x - (cols-1)*x_spacing/2 + j*x_spacing
                    # Offset every other row for hexagonal packing
                    if i % 2 == 1:
                        x += x_spacing/2
                    y = center_y - (rows-1)*y_spacing/2 + i*y_spacing

                    # Add controlled asymmetry to break symmetries
                    # This helps escape local optima during optimization
                    asymmetry_factor = 0.05
                    x += np.random.normal(0, asymmetry_factor * optimal_spacing)
                    y += np.random.normal(0, asymmetry_factor * optimal_spacing)

                    # Clip to ensure within bounds
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    points.append([x, y])

        # Fill remaining points with more structured approach rather than pure random
        # Use a spiral pattern to distribute points more evenly
        if len(points) < self.n_points:
            # Add spiral points to fill remaining positions
            # This creates a more uniform distribution
            angle_step = 0.5
            radius_step = 0.01
            angle = 0
            radius = 0.05

            while len(points) < self.n_points:
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)

                # Add small noise to prevent perfect symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)

                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])

                angle += angle_step
                radius += radius_step

                # Prevent infinite loop
                if radius > 0.5:
                    break

        return np.array(points[:self.n_points])

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

        # Apply simulated annealing as final refinement if we found a good solution
        if best_points is not None and best_ratio > 0:
            refined_points, refined_ratio = self._simulated_annealing_refinement(best_points.copy())
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio

        # Final validation
        if best_points is not None:
            final_ratio = self._objective_function(best_points.flatten())
            print(f"Final optimized ratio: {final_ratio:.6f}")
            return best_points
        else:
            # Return the last attempted configuration if nothing worked
            return initial_configs[-1] if initial_configs else np.random.rand(self.n_points, self.dimensions)

    def _simulated_annealing_refinement(self, points, max_iter=2000, initial_temp=0.1, cooling_rate=0.999):
        """
        Refine the solution using simulated annealing to escape local optima
        """
        current_points = points.copy()
        current_ratio = self._objective_function(current_points.flatten())
        best_points = current_points.copy()
        best_ratio = current_ratio

        temp = initial_temp

        for iteration in range(max_iter):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))

            # Perturb the selected point with larger steps for better exploration
            neighbor_points[idx, 0] += np.random.normal(0, 0.02)
            neighbor_points[idx, 1] += np.random.normal(0, 0.02)

            # Keep within bounds
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)

            # Calculate neighbor ratio
            neighbor_ratio = self._objective_function(neighbor_points.flatten())

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
                    acceptance_prob = np.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio

            # Cool down
            temp *= cooling_rate

            # Early stopping condition
            if temp < 1e-8:
                break

        return best_points, best_ratio

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