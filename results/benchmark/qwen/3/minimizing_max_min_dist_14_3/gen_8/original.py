# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape to points
        points = x.reshape(-1, 3)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Zero out diagonal
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (min/max)
        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    def constraint_sphere(x):
        # Keep points on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    def constraint_bounds(x):
        # Keep points within [-1, 1]^3
        points = x.reshape(-1, 3)
        return np.concatenate([
            1.0 - np.abs(points).max(axis=0),
            np.abs(points).max(axis=0) - (-1.0)
        ])

    # Start with a good initial configuration
    # Using a spherical arrangement as starting point
    np.random.seed(42)

    # Generate points uniformly distributed on sphere using Fibonacci
    # This provides a good initial distribution
    n = 14
    points = []

    # Fibonacci sphere sampling
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = np.arccos(y)  # angle from z-axis
        phi = (i * 2 * np.pi) / golden_ratio  # azimuthal angle

        x = radius * np.cos(phi)
        z = radius * np.sin(phi)

        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere
    points = points / np.linalg.norm(points[0])

    # Flatten for optimization
    x0 = points.flatten()

    # Define constraints
    cons = [
        {'type': 'eq', 'fun': constraint_sphere},
        # Bounds for each coordinate
        {'type': 'ineq', 'fun': lambda x: 1.0 - np.abs(x.reshape(-1, 3)).max(axis=0)},
        {'type': 'ineq', 'fun': lambda x: np.abs(x.reshape(-1, 3)).max(axis=0) - (-1.0)}
    ]

    # Use L-BFGS-B method which handles bounds well
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Ensure they're normalized to unit sphere (should already be done by constraint)
    for i in range(len(optimized_points)):
        norm = np.linalg.norm(optimized_points[i])
        if norm > 0:
            optimized_points[i] = optimized_points[i] / norm

    return optimized_points

# EVOLVE-BLOCK-END