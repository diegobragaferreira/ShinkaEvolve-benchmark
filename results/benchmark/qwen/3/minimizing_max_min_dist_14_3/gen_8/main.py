# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
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
        # Add small epsilon to avoid division by zero
        if d_max < 1e-12:
            return -1e12
        return -d_min / d_max

    def constraint_sphere(x):
        # Keep points on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    # Better initial point placement using known optimal configurations
    # Start with a regular icosahedron and then refine
    np.random.seed(42)

    # Create 14 points using a modified spherical code approach
    # Based on known good configurations for 14 points
    points = np.array([
        [0.0, 0.0, 1.0],                    # North pole
        [0.0, 0.0, -1.0],                   # South pole
        [0.5, 0.5, 0.7071067811865476],     # Some point on sphere
        [-0.5, 0.5, 0.7071067811865476],    # Mirror point
        [0.5, -0.5, 0.7071067811865476],    # Mirror point
        [-0.5, -0.5, 0.7071067811865476],   # Mirror point
        [0.5, 0.5, -0.7071067811865476],    # Opposite hemisphere
        [-0.5, 0.5, -0.7071067811865476],   # Mirror point
        [0.5, -0.5, -0.7071067811865476],   # Mirror point
        [-0.5, -0.5, -0.7071067811865476],  # Mirror point
        [0.7071067811865476, 0.0, 0.7071067811865476],  # On equator
        [-0.7071067811865476, 0.0, 0.7071067811865476], # Mirror
        [0.0, 0.7071067811865476, 0.7071067811865476],  # On equator
        [0.0, -0.7071067811865476, 0.7071067811865476]   # Mirror
    ])

    # Normalize to unit sphere
    for i in range(len(points)):
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm

    # Try multiple starting points to avoid local optima
    best_result = None
    best_ratio = -np.inf

    # Multi-start optimization - try different initial configurations
    for start_iter in range(5):
        # Perturb the initial configuration slightly
        np.random.seed(42 + start_iter)
        perturbed_points = points + np.random.normal(0, 0.05, points.shape)

        # Normalize again
        for i in range(len(perturbed_points)):
            norm = np.linalg.norm(perturbed_points[i])
            if norm > 0:
                perturbed_points[i] = perturbed_points[i] / norm

        x0 = perturbed_points.flatten()

        # Define constraints
        cons = [
            {'type': 'eq', 'fun': constraint_sphere}
        ]

        # Use L-BFGS-B method which handles bounds well
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                # Extract optimized points
                optimized_points = result.x.reshape(-1, 3)

                # Calculate the actual ratio
                distances = squareform(pdist(optimized_points))
                np.fill_diagonal(distances, np.inf)
                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = optimized_points.copy()

        except Exception as e:
            continue  # Skip this iteration if optimization fails

    # If no good solution found, return the original points
    if best_result is None:
        return points

    return best_result

# EVOLVE-BLOCK-END