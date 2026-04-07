# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a geometry-inspired packing evolution algorithm with hierarchical refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_distance_ratios(points):
        """Helper function to compute the ratio metrics for evaluation."""
        if len(points) < 2:
            return 0, 0, 0
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0, 0, 0
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def objective_function(x):
        """Objective function that returns negative ratio to maximize ratio."""
        points = x.reshape(-1, 2)
        ratio, _, _ = compute_distance_ratios(points)
        return -ratio

    def create_hexagonal_lattice():
        """Create a hexagonal lattice pattern with perturbations."""
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])

        # Add small random perturbations for diversity
        points = np.array(points[:16])
        noise = np.random.normal(0, 0.015, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_golden_ratio_pattern():
        """Create pattern based on golden ratio properties for even distribution."""
        np.random.seed(42)
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Distribute points using golden angle spiral approach
        for i in range(16):
            angle = i * 2 * np.pi / phi
            radius = np.sqrt(i / 15.0)  # Scale to keep within unit square
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([x, y])

        # Add small perturbations to break perfect symmetry
        points = np.array(points)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_grid_pattern():
        """Create a regular grid pattern with slight perturbations."""
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])

        # Add noise for diversity
        points = np.array(points)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_refined_spiral_pattern():
        """Create a refined spiral pattern."""
        np.random.seed(42)

        # Create a base spiral
        points = []
        for i in range(16):
            t = i / 15.0 * 4 * np.pi
            r = 0.4 * (i / 15.0)
            x = 0.5 + r * np.cos(t) * 0.8
            y = 0.5 + r * np.sin(t) * 0.8
            points.append([x, y])

        # Add some randomization to break symmetry
        points = np.array(points)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_symmetry_breaking_pattern():
        """Create a pattern that breaks common symmetries."""
        # Start with a regular grid
        points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225
                y = 0.1 + i * 0.225
                points.append([x, y])

        points = np.array(points)

        # Apply non-linear transformation to break symmetry
        transformed = points.copy()
        for i in range(len(transformed)):
            # Apply small non-linear deformations per point
            rx = np.random.random() * 0.02
            ry = np.random.random() * 0.02
            transformed[i][0] += rx * (points[i][0] - 0.5)
            transformed[i][1] += ry * (points[i][1] - 0.5)

        # Add noise and clip
        noise = np.random.normal(0, 0.01, transformed.shape)
        transformed = transformed + noise
        transformed = np.clip(transformed, 0, 1)
        return transformed

    def adaptive_optimize(points, method='L-BFGS-B', max_iter=500, time_remaining=None):
        """Adaptive optimization with intelligent parameter selection."""
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        # Set optimization parameters based on available time
        if time_remaining and time_remaining < 30:
            options = {'maxiter': max_iter//2, 'ftol': 1e-10, 'gtol': 1e-10}
        elif time_remaining and time_remaining < 60:
            options = {'maxiter': max_iter, 'ftol': 1e-11, 'gtol': 1e-11}
        else:
            options = {'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}

        try:
            result = minimize(
                objective_function,
                x0,
                method=method,
                bounds=bounds,
                options=options
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                return final_points
        except:
            pass

        return points

    def multi_start_optimization(initial_points_list, time_limit=150):
        """Run optimization with multiple starting points and methods."""
        start_time = time.time()
        best_ratio = -np.inf
        best_points = None
        methods = ['L-BFGS-B', 'SLSQP', 'Nelder-Mead']

        # Try each initial configuration with different optimization methods
        for i, initial_points in enumerate(initial_points_list):
            if (time.time() - start_time) > time_limit * 0.95:
                break

            for method in methods:
                if (time.time() - start_time) > time_limit * 0.95:
                    break

                try:
                    # Use different optimization parameters for each method
                    if method == 'Nelder-Mead':
                        adapted_points = adaptive_optimize(
                            initial_points,
                            method=method,
                            max_iter=300,
                            time_remaining=time_limit - (time.time() - start_time)
                        )
                    else:
                        adapted_points = adaptive_optimize(
                            initial_points,
                            method=method,
                            max_iter=500,
                            time_remaining=time_limit - (time.time() - start_time)
                        )

                    ratio, _, _ = compute_distance_ratios(adapted_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = adapted_points.copy()

                except Exception as e:
                    continue

        return best_points if best_points is not None else initial_points_list[0]

    def hierarchical_evolution(time_limit=150):
        """Hierarchical optimization with progressive refinement."""
        start_time = time.time()
        best_ratio = 0
        best_points = None

        # Create diverse initial patterns
        initial_patterns = [
            create_hexagonal_lattice(),
            create_golden_ratio_pattern(),
            create_grid_pattern(),
            create_refined_spiral_pattern(),
            create_symmetry_breaking_pattern()
        ]

        # Add some random initialization for diversity
        np.random.seed(42)
        for _ in range(3):
            if (time.time() - start_time) > time_limit * 0.95:
                break
            random_points = np.random.rand(16, 2)
            initial_patterns.append(random_points)

        # Multi-start optimization across all initial patterns
        try:
            refined_points = multi_start_optimization(initial_patterns, time_limit)
            if refined_points is not None:
                ratio, _, _ = compute_distance_ratios(refined_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
        except:
            pass

        # Additional refinement if needed
        if best_points is not None and (time.time() - start_time) < time_limit * 0.8:
            try:
                # Apply final high-precision optimization
                final_points = adaptive_optimize(
                    best_points,
                    method='SLSQP',
                    max_iter=800,
                    time_remaining=time_limit - (time.time() - start_time)
                )
                final_ratio, _, _ = compute_distance_ratios(final_points)
                if final_ratio > best_ratio:
                    best_points = final_points
            except:
                pass

        return best_points if best_points is not None else create_grid_pattern()

    # Main execution with time management
    try:
        result = hierarchical_evolution(time_limit=150)
        return result
    except Exception as e:
        # Fallback to simple grid pattern
        return create_grid_pattern()

# EVOLVE-BLOCK-END