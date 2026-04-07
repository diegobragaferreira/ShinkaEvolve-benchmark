# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def constraint_bounds(x):
        """Constraint function for bounds checking"""
        points = x.reshape(-1, 2)
        # Check bounds: each coordinate should be in [0,1]
        violations = []
        for coord in [points[:, 0], points[:, 1]]:
            violations.extend(np.maximum(0 - coord, 0))   # lower bound
            violations.extend(np.maximum(coord - 1, 0))   # upper bound
        return np.array(violations)

    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)

    def optimize_with_restarts():
        """Run optimization with multiple random restarts"""
        best_ratio = -np.inf
        best_points = None
        best_result = None

        # Try multiple different initial configurations
        initial_configs = []

        # 1. Golden spiral pattern
        spiral_points = golden_spiral_2d(16)
        # Scale and center the spiral
        spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
            np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        initial_configs.append(spiral_points.copy())

        # 2. Perturbed grid
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        grid_points += np.random.normal(0, 0.05, (16, 2))
        grid_points = np.clip(grid_points, 0, 1)
        initial_configs.append(grid_points)

        # 3. Random uniform points with fixed seed
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_configs.append(random_points)

        # Run optimization for each configuration
        for i, init_points in enumerate(initial_configs):
            try:
                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]

                # Optimize
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-9, 'eps': 1e-6},
                    callback=None
                )

                if result.success:
                    # Extract final points
                    final_points = result.x.reshape(-1, 2)

                    # Compute actual ratio
                    distances = pdist(final_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            best_result = result

            except Exception as e:
                warnings.warn(f"Optimization failed for initial config {i}: {e}")
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            return initial_configs[0]

        return best_points

    # Try optimized approach first
    try:
        final_points = optimize_with_restarts()
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to simple approach if something fails
        np.random.seed(42)
        final_points = np.random.rand(16, 2)

    return final_points


# EVOLVE-BLOCK-END