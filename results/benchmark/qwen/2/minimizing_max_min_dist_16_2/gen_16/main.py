# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero and penalize out-of-bounds points
        if d_max <= 0:
            return -np.inf

        ratio = d_min / d_max

        # Add penalty for points outside [0,1] bounds
        penalty = 0
        for i in range(16):
            if points[i, 0] < 0 or points[i, 0] > 1 or points[i, 1] < 0 or points[i, 1] > 1:
                penalty += 1000

        # Return negative ratio to minimize (since we want to maximize the ratio)
        return -(ratio - penalty / 10000)

    def objective_with_gradients(x):
        # Simple gradient approximation using finite differences
        eps = 1e-8
        grad = np.zeros_like(x)

        # Compute gradient using finite differences
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)

        return objective(x), grad

    # Create bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Initialize with a better starting configuration (hexagonal packing pattern)
    np.random.seed(42)

    # Create a hexagonal-like pattern that's more evenly distributed
    initial_points = np.zeros((16, 2))

    # Arrange points in a hexagonal pattern with some randomization
    hex_positions = []
    # Generate points in a hexagonal lattice pattern
    for i in range(4):
        for j in range(4):
            # Offset every other row
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + offset) / 3.0
            y = i / 3.0

            # Add slight perturbation to avoid perfect grid
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)

            # Clamp to bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            hex_positions.append([x, y])

    # Ensure we have exactly 16 points
    initial_points = np.array(hex_positions[:16])

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    try:
        # Use differential evolution with fewer iterations for speed
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        # Refine with local optimization
        refined_result = minimize(
            objective,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9}
        )

        final_result = refined_result.x

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to the hexagonal configuration
        final_result = initial_flat

    # Convert back to points array
    points = final_result.reshape(-1, 2)

    # Ensure all points are within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)

    return points


# EVOLVE-BLOCK-END