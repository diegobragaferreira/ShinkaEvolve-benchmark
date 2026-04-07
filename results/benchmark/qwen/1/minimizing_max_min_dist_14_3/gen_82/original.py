# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.
    """

    n = 14

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value to avoid self-distance issues
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    # Generate initial points using Fibonacci spiral on sphere
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n):
        phi = np.arccos(1 - 2 * i / (n - 1))
        theta = 2 * np.pi * i / golden_ratio

        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append([x, y, z])

    initial_points = np.array(points)

    # Normalize to unit sphere
    initial_points = initial_points / np.linalg.norm(initial_points[0]) if np.linalg.norm(initial_points[0]) > 0 else initial_points

    # Multiple restarts to find better solution
    best_ratio = -np.inf
    best_points = initial_points.copy()

    # Try multiple random perturbations of the initial solution
    for restart in range(10):
        # Add small noise to break symmetry
        np.random.seed(restart)
        noisy_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)

        # Normalize again
        noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1, keepdims=True)

        # Flatten for optimization
        x0 = noisy_points.flatten()

        # Define constraints
        cons = {'type': 'eq', 'fun': constraint_func}

        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', constraints=cons,
                            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000})

            if result.success:
                optimized_points = result.x.reshape(-1, 3)

                # Calculate final ratio
                distances = squareform(pdist(optimized_points))
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

        except Exception:
            continue

    return best_points

# EVOLVE-BLOCK-END