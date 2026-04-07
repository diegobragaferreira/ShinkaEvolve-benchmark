# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = cdist(points, points)

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective(points_flat):
        """Minimize negative of min/max ratio (equivalent to maximizing the ratio)."""
        points = points_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def constraint_func(points_flat):
        """Ensure points stay within [0,1] x [0,1]."""
        points = points_flat.reshape(-1, 2)
        # Each point coordinate should be between 0 and 1
        return np.concatenate([
            points[:, 0],     # x coordinates
            points[:, 1],     # y coordinates
            1 - points[:, 0], # 1 - x coordinates
            1 - points[:, 1]  # 1 - y coordinates
        ])

    def create_hexagonal_initialization():
        """Create points arranged in a hexagonal pattern to start with better spacing."""
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Hexagonal pattern with alternating rows
                x = j * 0.3 + (i % 2) * 0.15
                y = i * 0.3
                points.append([x, y])

        points = np.clip(points, 0, 1)
        return np.array(points[:16])

    def create_improved_grid_initialization():
        """Create an improved grid initialization with adaptive perturbations."""
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Even spacing with slight adjustments for better distribution
                x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
                y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5

                # Apply adaptive perturbation based on position
                if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                    # Corner points - smaller perturbation
                    perturbation = 0.01
                elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    # Edge points - medium perturbation
                    perturbation = 0.02
                else:
                    # Interior points - larger perturbation
                    perturbation = 0.03

                x += np.random.normal(0, perturbation)
                y += np.random.normal(0, perturbation)
                points.append([x, y])

        points = np.clip(points, 0, 1)
        return np.array(points[:16])

    def create_random_initialization():
        """Create random initialization with intentional spread."""
        np.random.seed(42)
        points = np.random.rand(16, 2)

        # Apply some basic spacing to prevent clustering
        for i in range(16):
            # Move points away from center slightly
            center_vec = points[i] - [0.5, 0.5]
            center_distance = np.linalg.norm(center_vec)
            if center_distance > 0:
                points[i] += center_vec * 0.1 / center_distance

        # Clip to ensure within bounds
        points = np.clip(points, 0, 1)
        return points

    # Multi-start approach to try different initial configurations
    best_ratio = -np.inf
    best_points = None

    # Different initialization strategies
    initial_strategies = [
        ("grid_perturbed", lambda: create_improved_grid_initialization()),
        ("hexagonal", lambda: create_hexagonal_initialization()),
        ("random_spread", lambda: create_random_initialization())
    ]

    # Try each initialization strategy multiple times with different random seeds
    for strategy_name, init_func in initial_strategies:
        for restart in range(3):  # 3 restarts per strategy
            np.random.seed(42 + hash(strategy_name) + restart)

            # Get initial points
            initial_points = init_func()

            # Flatten for optimization
            initial_flat = initial_points.flatten()

            # Define bounds for all coordinates
            bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

            # Define constraint: all coordinates must be in [0,1]
            cons = {'type': 'ineq', 'fun': constraint_func}

            # Try multiple optimization methods
            methods_to_try = ['SLSQP', 'L-BFGS-B']
            method_results = []

            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
                    )

                    if result.success:
                        # Extract points and compute ratio
                        optimized_points = result.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = compute_min_max_ratio(optimized_points)
                        method_results.append((ratio, optimized_points, method))

                except Exception as e:
                    continue  # Skip if this optimization fails

            # Select best result from this initialization strategy
            if method_results:
                best_method_result = max(method_results, key=lambda x: x[0])
                ratio, optimized_points, method_used = best_method_result

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

    # If no optimization was successful, return a deterministic configuration
    if best_points is None:
        # Fallback to a known good configuration based on prior knowledge
        best_points = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.25, 0.75], [0.75, 0.75],
            [0.1, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.9, 0.9],
            [0.3, 0.5], [0.7, 0.5],
            [0.5, 0.3], [0.5, 0.7],
            [0.4, 0.4], [0.6, 0.6],
            [0.4, 0.6], [0.6, 0.4]
        ])

    # Final check to ensure we have a valid result
    if best_points is None or len(best_points) != 16:
        # Last resort: return the default grid points
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        x_grid, y_grid = np.meshgrid(grid_x, grid_y)
        best_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])[:16]

    print(f"Final min/max ratio: {best_ratio:.6f}")
    print(f"Benchmark ratio: {best_ratio / 0.2786:.6f}")

    return best_points


# EVOLVE-BLOCK-END