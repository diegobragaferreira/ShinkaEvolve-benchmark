# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def constraint_function(x_flat):
        """Constraint to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Return negative values for constraint violations (we want >= 0)
        violations = np.concatenate([
            np.minimum(points[:, 0], 0),
            np.minimum(points[:, 1], 0),
            np.maximum(points[:, 0] - 1, 0),
            np.maximum(points[:, 1] - 1, 0)
        ])
        return violations

    # Generate initial points using a spherical code-inspired approach
    # Project points from Fibonacci spiral on sphere to 2D
    n_points = 16
    points = np.zeros((n_points, 2))

    # Use Fibonacci spiral approach for good initial distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        z = 1 - (i / (n_points - 1)) * 2  # z coordinate from -1 to 1
        radius = np.sqrt(1 - z*z)
        theta = np.arccos(z)
        phi = (i * golden_ratio) % (2 * np.pi)

        # Convert to Cartesian coordinates on unit sphere
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        # Map from sphere to square [0,1] x [0,1] using stereographic projection
        # This avoids edge effects and gives good distribution
        x_norm = (x + 1) / 2
        y_norm = (y + 1) / 2

        points[i] = [x_norm, y_norm]

    # Add some small perturbations to break symmetries
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)

    # Clip to valid bounds
    points = np.clip(points, 0, 1)

    # Flatten for optimization
    x0 = points.flatten()

    # Set up constraints
    bounds = [(0, 1) for _ in range(2 * n_points)]
    constraints = {'type': 'ineq', 'fun': constraint_function}

    # Local optimization with bounds and constraints
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-4}
    )

    # Extract final points
    final_points = result.x.reshape(-1, 2)

    # Final refinement with another optimization pass
    result2 = minimize(
        objective_function,
        final_points.flatten(),
        method='L-BFGS-B',
        bounds=[(0, 1) for _ in range(2 * n_points)],
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    final_points = result2.x.reshape(-1, 2)

    # Ensure points are clipped to valid domain
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END