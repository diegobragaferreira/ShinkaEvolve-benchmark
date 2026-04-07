# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time
import math

class PointOptimizer:
    """Enhanced optimizer for maximizing min/max distance ratio of 16 points in 2D."""

    def __init__(self, n_points=16, max_time=180):
        self.n_points = n_points
        self.max_time = max_time
        self.best_solution = None
        self.best_ratio = -np.inf
        self.start_time = time.time()

    def compute_distance_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        try:
            # Use squareform for numerical stability
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist == 0 or np.isinf(min_dist):
                return 0.0

            return min_dist / max_dist
        except Exception:
            return 0.0

    def initialize_hexagonal_grid(self):
        """Initialize using hexagonal grid pattern."""
        np.random.seed(42)
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                points.append([x, y])

        points = np.array(points[:self.n_points])

        # Normalize to [0,1] bounds
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

            # Scale to fit in [0,1] x [0,1]
            points[:, 0] *= 0.95
            points[:, 1] *= 0.95
            points[:, 0] += 0.025
            points[:, 1] += 0.025

        return points

    def initialize_golden_spiral(self):
        """Initialize using golden spiral pattern."""
        np.random.seed(42)
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(self.n_points):
            angle = 2 * np.pi * i / phi
            radius = 0.4 * (i / (self.n_points - 1) if self.n_points > 1 else 0.5)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        return np.array(points)

    def initialize_fibonacci_spiral(self):
        """Initialize using Fibonacci spiral pattern."""
        np.random.seed(42)
        points = []
        for i in range(self.n_points):
            angle = 2 * np.pi * i / (self.n_points - 1) if self.n_points > 1 else 0
            radius = 0.4 * np.sqrt(i / (self.n_points - 1)) if self.n_points > 1 else 0.5
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        return np.array(points)

    def initialize_regular_polygon(self):
        """Initialize using regular polygon pattern."""
        np.random.seed(42)
        points = []
        for i in range(self.n_points):
            angle = 2 * np.pi * i / self.n_points
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        return np.array(points)

    def initialize_fibonacci_sphere(self):
        """Initialize using Fibonacci sphere pattern projected to 2D."""
        np.random.seed(42)
        points = []
        # Generate Fibonacci sphere points
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(self.n_points):
            # Fibonacci sphere distribution
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = np.arctan2(y, radius)  # actual angle
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            # Project to 2D (using x and z coordinates)
            x_proj = 0.5 + 0.4 * x
            y_proj = 0.5 + 0.4 * z

            # Add small random perturbation
            x_proj += np.random.normal(0, 0.01)
            y_proj += np.random.normal(0, 0.01)

            # Clamp to [0,1]
            x_proj = np.clip(x_proj, 0, 1)
            y_proj = np.clip(y_proj, 0, 1)

            points.append([x_proj, y_proj])
        return np.array(points)

    def initialize_random(self):
        """Initialize using completely random pattern."""
        np.random.seed(42)
        return np.random.rand(self.n_points, 2)

    def initialize_points(self, method='hexagonal'):
        """Create structured initialization for better starting configuration."""
        if method == 'hexagonal':
            return self.initialize_hexagonal_grid()
        elif method == 'golden_spiral':
            return self.initialize_golden_spiral()
        elif method == 'fibonacci_spiral':
            return self.initialize_fibonacci_spiral()
        elif method == 'regular_polygon':
            return self.initialize_regular_polygon()
        else:
            return self.initialize_random()

    def objective_function(self, x):
        """Objective function for optimization (minimize negative ratio)."""
        points = x.reshape(-1, 2)
        ratio = self.compute_distance_ratio(points)
        return -ratio

    def global_optimization_step(self, x0):
        """Perform global optimization using differential evolution."""
        bounds = [(0, 1)] * (self.n_points * 2)

        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=150,
                popsize=25,
                atol=1e-12,
                rtol=1e-12,
                mutation=(0.8, 1.0),
                recombination=0.9
            )
            return result.success, result.x if result.success else x0
        except Exception:
            return False, x0

    def simulated_annealing(self, x0, max_iter=1000):
        """Simulated annealing optimization as fallback method."""
        np.random.seed(int(time.time()) % 1000000)

        current_x = x0.copy()
        current_points = current_x.reshape(-1, 2)
        current_ratio = self.compute_distance_ratio(current_points)

        best_x = current_x.copy()
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Annealing parameters
        T_start = 1.0
        T_end = 1e-8
        alpha = 0.95
        T = T_start

        iter_count = 0
        while iter_count < max_iter and T > T_end:
            # Generate neighbor solution by perturbing one point
            neighbor_x = current_x.copy()
            point_idx = np.random.randint(0, self.n_points)
            coord_idx = np.random.randint(0, 2)

            # Perturb the selected coordinate
            step_size = 0.01 * T
            neighbor_x[point_idx * 2 + coord_idx] += np.random.normal(0, step_size)

            # Keep within bounds
            neighbor_x[point_idx * 2 + coord_idx] = np.clip(neighbor_x[point_idx * 2 + coord_idx], 0, 1)

            # Evaluate neighbor
            neighbor_points = neighbor_x.reshape(-1, 2)
            neighbor_ratio = self.compute_distance_ratio(neighbor_points)

            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_x = neighbor_x
                current_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current_x = neighbor_x
                    current_ratio = neighbor_ratio

            # Update best solution
            if current_ratio > best_ratio:
                best_x = current_x.copy()
                best_ratio = current_ratio
                best_points = neighbor_points.copy()

            # Cool down
            T *= alpha
            iter_count += 1

        return True, best_x

    def local_refinement_step(self, x0, method='L-BFGS-B'):
        """Perform local refinement optimization with fallback."""
        bounds = [(0, 1)] * (self.n_points * 2)

        try:
            # Try primary method
            result = minimize(
                self.objective_function,
                x0,
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if result.success:
                return True, result.x
            else:
                # Try with looser tolerances
                result = minimize(
                    self.objective_function,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                return result.success, result.x if result.success else x0

        except Exception:
            # Try alternate method as fallback
            if method == 'L-BFGS-B':
                try:
                    result = minimize(
                        self.objective_function,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-10}
                    )
                    if result.success:
                        return True, result.x
                    else:
                        # Fallback to simulated annealing
                        return self.simulated_annealing(x0)
                except Exception:
                    # Fallback to simulated annealing
                    return self.simulated_annealing(x0)
            # Fallback to simulated annealing for all other cases
            return self.simulated_annealing(x0)

    def voronoi_relaxation(self, points, max_iter=50):
        """Perform Voronoi relaxation to improve point distribution."""
        from scipy.spatial import Voronoi

        current_points = points.copy()

        for iteration in range(max_iter):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)

                # Calculate new positions as centroids of Voronoi cells
                new_points = np.zeros_like(current_points)
                converged = True

                # Process each point
                for i in range(len(current_points)):
                    # Get vertices of Voronoi cell for point i
                    region = vor.regions[vor.point_region[i]]

                    if -1 in region or len(region) < 3:
                        # Handle unbounded regions (use current position with slight adjustment)
                        new_points[i] = current_points[i] + np.random.normal(0, 0.001, 2)
                        continue

                    # Extract vertices of the Voronoi cell
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])

                    if len(vertices) < 3:
                        # Not enough vertices, use current position
                        new_points[i] = current_points[i]
                        continue

                    # Compute centroid of polygon (Voronoi cell)
                    centroid = np.mean(vertices, axis=0)

                    # Apply boundary constraints with epsilon padding
                    centroid = np.clip(centroid, 1e-8, 1-1e-8)

                    # Update point position
                    new_points[i] = centroid

                    # Check for convergence
                    if np.linalg.norm(new_points[i] - current_points[i]) > 1e-6:
                        converged = False

                # Apply damping factor for stable convergence
                damping = 0.9
                current_points = current_points + damping * (new_points - current_points)

                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)

                # Early stopping if converged
                if converged:
                    break

            except Exception:
                # If Voronoi computation fails, use simple perturbation
                current_points = current_points + np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)

        return current_points

    def validate_and_update_best(self, points):
        """Validate solution and update best if better."""
        # Apply Voronoi relaxation to potentially improve the solution
        relaxed_points = self.voronoi_relaxation(points)
        ratio = self.compute_distance_ratio(relaxed_points)
        if ratio > self.best_ratio:
            self.best_ratio = ratio
            self.best_solution = relaxed_points.copy()

    def has_timed_out(self):
        """Check if maximum time has been exceeded."""
        return time.time() - self.start_time > self.max_time * 0.95

    def optimize(self):
        """Main optimization loop with multiple strategies and diverse initialization."""
        # List of different initialization methods to try
        init_methods = ['hexagonal', 'golden_spiral', 'fibonacci_spiral', 'regular_polygon', 'fibonacci_sphere', 'random']

        # Multi-start optimization with diverse initializations
        for init_method in init_methods:
            # Try multiple restarts for each method to ensure thorough exploration
            for restart in range(5):
                if self.has_timed_out():
                    break

                try:
                    # Create diverse seed for each restart
                    np.random.seed(42 + restart + hash(init_method) % 1000)

                    # Initialize with different methods
                    initial_points = self.initialize_points(init_method)
                    x0 = initial_points.flatten()

                    # Global optimization
                    global_success, x_global = self.global_optimization_step(x0)

                    # Multiple local refinement strategies
                    if global_success:
                        # Strategy 1: L-BFGS-B refinement
                        local_success, x_local = self.local_refinement_step(x_global, 'L-BFGS-B')
                        if local_success:
                            final_points = x_local.reshape(-1, 2)
                            self.validate_and_update_best(final_points)

                        # Strategy 2: SLSQP refinement
                        slsqp_success, x_slsqp = self.local_refinement_step(x_global, 'SLSQP')
                        if slsqp_success:
                            final_points = x_slsqp.reshape(-1, 2)
                            self.validate_and_update_best(final_points)

                        # Strategy 3: Additional L-BFGS-B with different config
                        if not local_success and not slsqp_success:
                            # Try with more iterations
                            _, x_refined = self.local_refinement_step(x_global, 'L-BFGS-B')
                            final_points = x_refined.reshape(-1, 2)
                            self.validate_and_update_best(final_points)

                except Exception:
                    continue

        # If we haven't found a good solution yet, try direct optimization from random initialization
        if self.best_solution is None:
            try:
                np.random.seed(42)
                x0_random = np.random.uniform(0, 1, self.n_points * 2)

                # Global optimization
                global_success, x_global = self.global_optimization_step(x0_random)

                # Local refinement
                if global_success:
                    local_success, x_local = self.local_refinement_step(x_global)
                    if local_success:
                        final_points = x_local.reshape(-1, 2)
                        self.validate_and_update_best(final_points)

            except Exception:
                pass

        # Final fallback to initialization if nothing worked
        if self.best_solution is None:
            self.best_solution = self.initialize_points('hexagonal')

        return self.best_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer(n_points=16, max_time=180)
    return optimizer.optimize()

# EVOLVE-BLOCK-END