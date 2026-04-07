# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
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

    # Multi-start optimization with different initialization strategies
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Hexagonal grid with perturbation
    def hexagonal_grid_init():
        points = np.zeros((16, 2))
        # Arrange in hexagonal pattern (roughly)
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)

        for i in range(rows):
            for j in range(cols):
                if i * cols + j < 16:  # Only take first 16 points
                    x = j * spacing_x
                    y = i * spacing_y
                    # Add slight offset to break symmetry
                    if i % 2 == 1:
                        x += spacing_x * 0.25
                    points[i * cols + j] = [x, y]
        return points

    # Strategy 2: Random with clustering avoidance
    def random_init():
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points

    # Strategy 3: Perturbed hexagonal grid (more structured)
    def perturbed_hexagonal_init():
        points = hexagonal_grid_init()
        np.random.seed(42)
        # Add small random perturbations
        points += np.random.normal(0, 0.02, points.shape)
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points

    initial_strategies = [
        hexagonal_grid_init,
        random_init,
        perturbed_hexagonal_init
    ]

    # Try multiple random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        # Select initialization strategy
        init_func = initial_strategies[restart % len(initial_strategies)]
        points = init_func()

        # Flatten for optimization
        x0 = points.flatten()

        # Set up constraints
        bounds = [(0, 1) for _ in range(32)]
        constraints = {'type': 'ineq', 'fun': constraint_function}

        # First optimization with SLSQP for global search
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()

        except Exception as e:
            continue

    # If none worked, fallback to simple approach
    if best_points is None:
        # Fallback to the original approach with better settings
        points = hexagonal_grid_init()
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        x0 = points.flatten()

        bounds = [(0, 1) for _ in range(32)]
        constraints = {'type': 'ineq', 'fun': constraint_function}

        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback to random points
                np.random.seed(42)
                best_points = np.random.rand(16, 2)
        except:
            # Final fallback
            np.random.seed(42)
            best_points = np.random.rand(16, 2)

    return best_points

# EVOLVE-BLOCK-END