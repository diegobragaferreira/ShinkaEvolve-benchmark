# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import time
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    # Phase 1: Generate initial configuration using Fibonacci sphere with adaptation
    n = 14

    # Generate points on a sphere using Fibonacci method
    points_sphere = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points_sphere.append([x, y, z])

    # Convert to numpy array and normalize to unit sphere
    points = np.array(points_sphere)

    # Scale to approximate optimal distribution in unit cube
    # First, scale to unit cube approximately
    points = (points + 1) / 2  # Now in [0,1]^3

    # Add slight random perturbation to break degeneracies
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)

    # Clip to ensure bounds
    points = np.clip(points, 0, 1)

    # Phase 2: Local optimization using custom energy-based approach
    def compute_min_max_ratio(points_flat):
        """Compute negative of min/max distance ratio for optimization"""
        points = points_flat.reshape(-1, 3)
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return -np.inf

        return -min_dist / max_dist

    def local_refinement_step(points):
        """Apply local refinement using a constrained gradient-like method"""
        # Create a modified objective that penalizes violations of bounds
        def objective(params):
            # Reshape params
            points_temp = params.reshape(-1, 3)

            # Ensure bounds
            points_temp = np.clip(points_temp, 0, 1)

            # Compute ratio
            distances = cdist(points_temp, points_temp, 'euclidean')
            np.fill_diagonal(distances, np.inf)

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist == 0:
                return 1e10

            ratio = min_dist / max_dist
            return -ratio  # Negative because we want to maximize

        # Use L-BFGS-B with bounds
        bounds = [(0, 1) for _ in range(3 * n)]

        # Simple gradient-free local search using Nelder-Mead for robustness
        try:
            result = minimize(objective, points.flatten(), method='Nelder-Mead',
                            options={'maxiter': 500, 'disp': False})
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass

        return points

    # Perform local refinement
    points_refined = local_refinement_step(points)

    # Final optimization using a hybrid approach
    # Try several restarts with different seeds to escape local minima
    best_points = points_refined.copy()
    best_ratio = compute_min_max_ratio(best_points.flatten())

    # Run additional local optimizations with different starting points
    for attempt in range(5):
        np.random.seed(attempt * 100 + 42)

        # Perturb existing points slightly
        perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
        perturbed = np.clip(perturbed, 0, 1)

        refined = local_refinement_step(perturbed)
        current_ratio = compute_min_max_ratio(refined.flatten())

        if current_ratio < best_ratio:  # Better ratio (more negative means better)
            best_ratio = current_ratio
            best_points = refined.copy()

    # Final check and return
    final_points = best_points

    # Ensure the point set meets requirements
    final_points = np.clip(final_points, 0, 1)

    # Verify we have 14 points in 3D
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"

    return final_points


# EVOLVE-BLOCK-END