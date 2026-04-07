# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def fibonacci_spiral_on_sphere(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms

    def objective_function(points_flat):
        """Objective function to maximize min/max distance ratio"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return -np.inf
        return min_dist / max_dist

    # Generate initial points using Fibonacci spiral on sphere
    np.random.seed(42)
    initial_points = fibonacci_spiral_on_sphere(14)

    # Stage 1: Global optimization using Differential Evolution
    initial_flat = initial_points.flatten()
    bounds = [(-1, 1) for _ in range(42)]

    # Run differential evolution with constraints
    de_result = differential_evolution(
        lambda x: -objective_function(x),  # Minimize negative to maximize
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )

    # Stage 2: Local refinement using L-BFGS-B
    # Extract points from DE result and refine
    refined_points = de_result.x.reshape(-1, 3)

    # Flatten for optimization
    refined_flat = refined_points.flatten()

    # Refinement with L-BFGS-B
    lbfgs_bounds = [(-1, 1) for _ in range(42)]

    lbfgs_result = minimize(
        lambda x: -objective_function(x),
        refined_flat,
        method='L-BFGS-B',
        bounds=lbfgs_bounds,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
        tol=1e-12
    )

    # Extract final optimized points
    final_points = lbfgs_result.x.reshape(-1, 3)

    # Final normalization to unit sphere
    final_points = normalize_to_unit_sphere(final_points)

    return final_points


# EVOLVE-BLOCK-END