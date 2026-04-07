# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def energy_function(points: np.ndarray) -> float:
        """Energy function that penalizes small distances."""
        distances = pdist(points)
        # Use inverse squared distances as penalty (higher penalty for smaller distances)
        return -np.sum(1.0 / (distances ** 2 + 1e-10))

    def objective_function(x_flat: np.ndarray) -> float:
        """Objective function for optimization (negative of min-max ratio)."""
        points = x_flat.reshape(-1, 3)
        return -compute_min_max_ratio(points)

    # Start with vertices of a regular icosahedron scaled to unit sphere
    # Icosahedron vertices (normalized to unit sphere)
    phi = (1 + math.sqrt(5)) / 2  # golden ratio

    # Generate icosahedron vertices
    vertices = []
    # Add vertices at (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
    for i in [1, -1]:
        for j in [1, -1]:
            vertices.append([i, j * phi, 0])
            vertices.append([0, i, j * phi])
            vertices.append([i * phi, 0, j])

    # Normalize to unit sphere
    vertices = np.array(vertices)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Take first 14 points (actually 12, so we'll add some points to make 14)
    if len(vertices) >= 14:
        points = vertices[:14]
    else:
        # If we have fewer vertices, duplicate and slightly perturb
        points = vertices.copy()
        while len(points) < 14:
            points = np.vstack([points, points[:(14-len(points))]])

        # Add some small random noise to make them distinct
        points = points[:14] + 0.01 * np.random.randn(14, 3)

    # Normalize to unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Apply optimization using L-BFGS for local refinement
    initial_points = points.copy()

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Use L-BFGS for local optimization
    result = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Normalize to unit sphere again
    optimized_points = optimized_points / np.linalg.norm(optimized_points, axis=1, keepdims=True)

    # Final improvement using a more robust approach
    best_points = optimized_points.copy()
    best_ratio = compute_min_max_ratio(best_points)

    # Try several random restarts with different initializations
    for _ in range(10):
        # Perturb the solution slightly
        perturbed = optimized_points + 0.05 * np.random.randn(*optimized_points.shape)
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)

        # Optimize again
        x0_pert = perturbed.flatten()
        try:
            result_pert = minimize(
                objective_function,
                x0_pert,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            optimized_pert = result_pert.x.reshape(-1, 3)
            optimized_pert = optimized_pert / np.linalg.norm(optimized_pert, axis=1, keepdims=True)

            ratio = compute_min_max_ratio(optimized_pert)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_pert
        except:
            continue

    return best_points

# EVOLVE-BLOCK-END