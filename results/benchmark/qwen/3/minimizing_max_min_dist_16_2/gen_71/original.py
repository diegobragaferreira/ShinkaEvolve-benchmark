# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective_function(params):
        """Objective function to minimize (negative of min/max ratio)."""
        # Reshape parameters back to points array
        points = params.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize the ratio
        return -ratio

    # Generate initial points using a better structured approach for improved starting configuration
    np.random.seed(42)

    # Create a more optimized initial configuration
    # Use a combination of hexagonal and perturbed grid arrangement
    points = []

    # Arrange in a 4x4 grid (16 points total) with appropriate spacing
    for i in range(4):
        for j in range(4):
            x = j * 0.25 + (i % 2) * 0.125  # Offset every other row for hexagonal packing
            y = i * 0.25
            points.append([x, y])

    # Convert to numpy array and add small random noise
    points = np.array(points) + np.random.normal(0, 0.005, (16, 2))

    # Ensure all points are within [0,1] bounds and normalize properly
    points = np.clip(points, 0, 1)

    # Flatten for optimization
    initial_params = points.flatten()

    # Set up bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each

    # Optimize using L-BFGS-B which is well-suited for this type of problem
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=None
        )

        # Extract optimized points
        optimized_points = result.x.reshape(-1, 2)

        # Make sure they're within bounds
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    except Exception as e:
        # Fallback to initial points if optimization fails
        print(f"Optimization failed: {e}")
        return points

# EVOLVE-BLOCK-END