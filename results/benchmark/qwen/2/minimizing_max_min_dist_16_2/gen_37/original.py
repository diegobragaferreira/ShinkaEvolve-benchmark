# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points_flat):
        """Compute the min/max distance ratio for given flattened point coordinates."""
        # Reshape flat array back to (16, 2)
        points = points_flat.reshape(-1, 2)

        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Compute min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return ratio (avoid division by zero)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    def objective(points_flat):
        """Minimize negative of min/max ratio (equivalent to maximizing the ratio)."""
        return -compute_min_max_ratio(points_flat)

    def constraint_func(points_flat):
        """Ensure points stay within [0,1] x [0,1]."""
        points = points_flat.reshape(-1, 2)
        # Each point coordinate should be between 0 and 1
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])

    # Grid-based initialization - better than random
    np.random.seed(42)
    grid_x = np.linspace(0.1, 0.9, 4)  # Avoid edges
    grid_y = np.linspace(0.1, 0.9, 4)
    x_grid, y_grid = np.meshgrid(grid_x, grid_y)
    initial_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Add slight random perturbation to break perfect symmetry
    initial_points += np.random.normal(0, 0.02, initial_points.shape)

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Define constraints for bounds: 0 <= x_i <= 1, 0 <= y_i <= 1
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Define constraint: all coordinates must be in [0,1]
    cons = {'type': 'ineq', 'fun': constraint_func}

    # Optimize using SLSQP method
    start_time = time.time()

    result = minimize(
        objective,
        initial_flat,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    end_time = time.time()

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 2)

    # Ensure all points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    # Compute final metrics
    final_ratio = compute_min_max_ratio(result.x)
    benchmark_ratio = final_ratio / 0.2786  # AlphaEvolve benchmark
    eval_time = end_time - start_time

    # Print metrics for debugging
    print(f"Final min/max ratio: {final_ratio:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Evaluation time: {eval_time:.6f}s")

    return optimized_points


# EVOLVE-BLOCK-END