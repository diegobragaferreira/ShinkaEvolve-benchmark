# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances with numerical stability
        distances = squareform(pdist(points))

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        if len(distances[distances != np.inf]) == 0:
            return -1.0  # Worst possible case

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0 or np.isinf(min_dist):
            return -1.0

        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are within [0,1] x [0,1]
        points = x.reshape(-1, 2)

        # Check that all points are within bounds
        violations = []

        # x coordinates in [0,1]
        violations.append(np.min(points[:, 0]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 0]))  # Should be >= 0

        # y coordinates in [0,1]
        violations.append(np.min(points[:, 1]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 1]))  # Should be >= 0

        return np.array(violations)

    def bounded_objective(x):
        # Boundary clamping
        points = np.clip(x.reshape(-1, 2), 0, 1).flatten()
        return objective(points)

    def create_spherical_initialization(n_points=16):
        """Create initial points using spherical geometry projection for better distribution"""
        # Generate points on a sphere using Fibonacci method
        points_sphere = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points_sphere.append([x, y, z])

        # Project 3D sphere points to 2D using stereographic projection
        points_2d = []
        for x, y, z in points_sphere:
            # Stereographic projection from south pole
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])

        # Normalize to unit square
        points_2d = np.array(points_2d)

        # Scale and center the points
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)

        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05

        return points_2d

    def create_hexagonal_initialization(n_points=16):
        """Create hexagonal grid initialization"""
        # Create a proper hexagonal arrangement
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25

                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)

                points.append([x, y])

        points = np.array(points[:n_points])

        # Normalize to [0,1] bounds
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Scale and shift to fit nicely in unit square
        points[:, 0] *= 0.95
        points[:, 1] *= 0.95
        points[:, 0] += 0.025
        points[:, 1] += 0.025

        return np.clip(points, 0, 1)

    # Create multiple initial configurations for diversity
    np.random.seed(42)

    # Try multiple initialization strategies
    initial_configs = [
        create_spherical_initialization(16),
        create_hexagonal_initialization(16),
        np.random.rand(16, 2)
    ]

    best_points = None
    best_ratio = -np.inf

    # Try each initial configuration
    for i, initial_points in enumerate(initial_configs):
        # Flatten for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]

        # First use differential evolution for global search
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    bounded_objective,
                    bounds,
                    seed=42,
                    maxiter=50,
                    popsize=15,
                    tol=1e-6,
                    mutation=(0.5, 1),
                    recombination=0.7
                )

            # If DE succeeded, use that result
            if de_result.success:
                x0 = de_result.x
        except Exception:
            pass

        # Local optimization with L-BFGS-B for fine-tuning
        try:
            local_result = minimize(
                bounded_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if local_result.success:
                optimized_points = local_result.x.reshape(-1, 2)
                # Ensure all points are within bounds
                optimized_points = np.clip(optimized_points, 0, 1)

                # Calculate final ratio
                final_ratio = -objective(local_result.x)

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue

    # If we still don't have a solution, use the best configuration
    if best_points is None:
        # Fallback to a modified hexagonal grid
        best_points = create_hexagonal_initialization(16)

    return best_points


# EVOLVE-BLOCK-END