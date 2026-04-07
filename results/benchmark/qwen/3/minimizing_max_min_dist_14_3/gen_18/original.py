# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 3)

        # Calculate distance matrix
        distances = cdist(points, points)

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we minimize in scipy)
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return 0

    def spherical_fibonacci_points(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n):
            # Latitude
            phi = np.arccos(1 - 2*i/(n-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def normalize_to_cube(points):
        """Normalize points to fit in [0,1]^3"""
        # Scale to [-1,1]^3 first
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)

        # Avoid division by zero
        scale = np.maximum(max_vals - min_vals, 1e-10)

        # Normalize to [-1,1]^3
        normalized = (points - min_vals) / scale * 2 - 1

        # Then scale to [0,1]^3
        return (normalized + 1) / 2

    # Generate initial points using Fibonacci spiral on sphere, then normalize
    np.random.seed(42)
    init_points = spherical_fibonacci_points(14)
    init_points = normalize_to_cube(init_points)

    # Flatten initial points for optimization
    x0 = init_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1)] * 42  # 14 points * 3 coordinates each

    # Run differential evolution optimization
    start_time = time.time()

    # Use a reasonable number of iterations for our time budget
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-8,
        mutation=(0.5, 1),
        recombination=0.7,
        callback=None
    )

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Clip points to ensure they're within bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Final ratio: {-result.fun:.6f}")

    return optimized_points


# EVOLVE-BLOCK-END