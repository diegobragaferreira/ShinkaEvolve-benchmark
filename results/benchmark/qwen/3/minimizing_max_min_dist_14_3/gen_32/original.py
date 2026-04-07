# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective_function(x):
        # Reshape flat array back to 14x3 points
        points = x.reshape((14, 3))

        # Compute pairwise distances efficiently
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def fibonacci_sphere_sampling(n):
        """Generate points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def initialize_points():
        """Initialize points using Fibonacci sphere sampling with perturbations"""
        # Generate points on a unit sphere
        points = fibonacci_sphere_sampling(14)

        # Add small random perturbations to escape local minima
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.02, points.shape)
        points += perturbations

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        # Scale appropriately to have reasonable distances
        points *= 0.8

        # Transform to [0,1]^3 space
        # Map from [-1,1]^3 to [0,1]^3
        points = (points + 1) / 2

        return points.flatten()

    def optimize_points(initial_points):
        """Optimize points using differential evolution"""
        # Bounds for each coordinate [0, 1]
        bounds = [(0, 1)] * 14 * 3

        # Use differential evolution with tuned parameters
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42,
                maxiter=500,
                popsize=10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-8,
                callback=None
            )

            # Extract final points
            final_points = result.x.reshape((14, 3))

            # Ensure all points are within bounds
            final_points = np.clip(final_points, 0, 1)

            return final_points

        except Exception as e:
            # Return initial points if optimization fails
            return initial_points.reshape((14, 3))

    # Initialize with better spherical configuration
    initial_points = initialize_points()

    # Optimize the configuration
    final_points = optimize_points(initial_points)

    return final_points

# EVOLVE-BLOCK-END