# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import random


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    def initialize_points():
        """Initialize points using a combination of spherical and structured approach"""
        # Start with points on a sphere (normalized to unit sphere)
        # Use Fibonacci spiral for good distribution on sphere
        points_sphere = []
        n = 14
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = np.arctan2(y, radius)  # angle from y-axis

            # Place points along a spiral
            theta = i * 2 * np.pi / phi

            x = radius * np.cos(theta)
            z = radius * np.sin(theta)

            points_sphere.append([x, y, z])

        # Normalize to unit sphere and scale appropriately
        points_sphere = np.array(points_sphere)
        points_sphere /= np.linalg.norm(points_sphere[0])  # Normalize first point to unit distance

        # Add some perturbation to improve distribution
        np.random.seed(42)
        points_sphere += np.random.normal(0, 0.05, points_sphere.shape)

        # Normalize again to keep them near unit sphere
        for i in range(len(points_sphere)):
            norm = np.linalg.norm(points_sphere[i])
            if norm > 0:
                points_sphere[i] = points_sphere[i] / norm

        # Scale to appropriate range in [0,1]^3
        # First center around origin and scale to [-0.5, 0.5]^3
        points_sphere -= np.mean(points_sphere, axis=0)
        points_sphere *= 0.5

        # Then shift to [0,1]^3
        points_sphere += 0.5

        # Ensure they're within bounds
        points_sphere = np.clip(points_sphere, 0, 1)

        return points_sphere.flatten()

    # Initialize with better starting points
    initial_points = initialize_points()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run optimization with reasonable settings
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500,
        popsize=20,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        init=[initial_points]  # Use our custom initialization
    )

    # Extract the best solution
    points = result.x.reshape(-1, 3)

    # Apply local refinement using L-BFGS-B for finer optimization
    def refined_objective(x):
        points_refined = x.reshape(-1, 3)
        distances = pdist(points_refined)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return -np.inf
        return -(d_min / d_max)

    # Local optimization with L-BFGS-B
    result_local = minimize(
        refined_objective,
        result.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    # Extract final solution
    points_final = result_local.x.reshape(-1, 3)

    return points_final


# EVOLVE-BLOCK-END