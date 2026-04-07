# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import math
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

    def _optimize_single_start(self, initial_points, max_iter=2000):
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
                warnings.warn(f"L-BFGS-B optimization failed: {result.message}")
                return initial_points, self._objective_function(flat_initial)

        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization error: {str(e)}")
            return initial_points, self._objective_function(flat_initial)

    def _compute_ratio(self, points):
        """Compute the actual ratio for given points"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max

    def _simulated_annealing(self, points, max_iter=5000, initial_temp=1.0, cooling_rate=0.9995):
        """
        Simulated Annealing optimization for point dispersion
        """
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        temp = initial_temp

        for iteration in range(max_iter):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))

            # Perturb the selected point with adaptive step size
            step_size = 0.02 if iteration < max_iter//2 else 0.005
            neighbor_points[idx, 0] += np.random.normal(0, step_size)
            neighbor_points[idx, 1] += np.random.normal(0, step_size)

            # Keep within bounds
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)

            # Calculate neighbor ratio
            neighbor_ratio = self._compute_ratio(neighbor_points)

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
            temp *= cooling_rate

            # Early stopping condition
            if temp < 1e-8:
                break

        return best_points, best_ratio

    def generate_initial_points(self):
        """Generate multiple diverse initial point configurations."""
        configurations = []
        np.random.seed(self.seed)

        # 1. Grid configuration
        grid_points = []
        grid_size = 4  # 4x4 grid for 16 points
        spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) < self.n_points:
                    grid_points.append([i * spacing, j * spacing])
        configurations.append(np.array(grid_points))

        # 2. Perturbed grid configuration
        perturbed_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(perturbed_points) < self.n_points:
                    x = max(0, min(1, i * spacing + np.random.normal(0, 0.05 * spacing)))
                    y = max(0, min(1, j * spacing + np.random.normal(0, 0.05 * spacing)))
                    perturbed_points.append([x, y])
        configurations.append(np.array(perturbed_points))

        # 3. Random configuration
        configurations.append(np.random.rand(self.n_points, 2))

        # 4. Hexagonal-like configuration with better spacing
        hex_points = []
        # Center point
        hex_points.append([0.5, 0.5])

        # Surrounding points in hexagonal pattern with proper spacing
        radius = 0.3
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for angle in angles:
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            if len(hex_points) < self.n_points:
                hex_points.append([x, y])

        # Fill remaining points with triangular packing pattern
        if len(hex_points) < self.n_points:
            # Add points in triangular lattice pattern to maximize dispersion
            rows = 3
            cols = 3
            spacing = 0.25  # Adjusted spacing for better distribution
            y_spacing = spacing * np.sqrt(3)
            x_spacing = spacing * 2

            for i in range(rows):
                for j in range(cols):
                    if len(hex_points) < self.n_points:
                        x = 0.5 - (cols-1)*x_spacing/2 + j*x_spacing
                        # Offset every other row for hexagonal packing
                        if i % 2 == 1:
                            x += x_spacing/2
                        y = 0.5 - (rows-1)*y_spacing/2 + i*y_spacing

                        # Add noise to break symmetry
                        x += np.random.normal(0, spacing * 0.1)
                        y += np.random.normal(0, spacing * 0.1)

                        # Clip to bounds
                        x = np.clip(x, 0, 1)
                        y = np.clip(y, 0, 1)
                        hex_points.append([x, y])

        # Fill remaining points randomly
        for i in range(self.n_points - len(hex_points)):
            hex_points.append([np.random.rand(), np.random.rand()])

        configurations.append(np.array(hex_points[:self.n_points]))

        # 5. Fibonacci-inspired pattern for better distribution
        fib_points = []
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

            fib_points.append([np.clip(x_mapped, 0, 1), np.clip(y_mapped, 0, 1)])

        configurations.append(np.array(fib_points))

        return configurations

    def optimize(self):
        """Main optimization routine with multi-start approach."""
        best_points = None
        best_ratio = -np.inf
        start_time = time.time()

        # Generate multiple initial configurations
        initial_configs = self.generate_initial_points()

        # Try all initial configurations with optimization
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time_seconds - 5:  # Leave buffer for final processing
                break

            try:
                # First, try L-BFGS-B optimization
                lbfgsb_points, lbfgsb_ratio = self._optimize_single_start(initial_config.copy())

                # Then refine with simulated annealing if time allows and solution is reasonable
                if time.time() - start_time < self.max_time_seconds - 10 and lbfgsb_ratio > 0.1:
                    sa_points, sa_ratio = self._simulated_annealing(lbfgsb_points.copy())
                    final_points = sa_points if sa_ratio > lbfgsb_ratio else lbfgsb_points
                    final_ratio = sa_ratio if sa_ratio > lbfgsb_ratio else lbfgsb_ratio
                else:
                    final_points = lbfgsb_points
                    final_ratio = lbfgsb_ratio

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()

            except Exception as e:
                warnings.warn(f"Error optimizing initial config {i}: {str(e)}")
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