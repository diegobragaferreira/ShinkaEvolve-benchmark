# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import itertools

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into 16 points
        points = x.reshape(-1, 2)
        # Calculate pairwise distances
        distances = pdist(points)
        # Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0  # Avoid division by zero
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points stay within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        return np.concatenate([points.flatten() - 1, -points.flatten()])

    # Start with a structured grid initialization
    # Create a 4x4 grid pattern within [0,1] x [0,1]
    grid_points = []
    for i in range(4):
        for j in range(4):
            grid_points.append([(i+0.5)/4, (j+0.5)/4])

    initial_points = np.array(grid_points)

    # Add small random perturbations to avoid degenerate cases
    np.random.seed(42)
    initial_points += np.random.normal(0, 0.01, initial_points.shape)

    # Clip points to valid range
    initial_points = np.clip(initial_points, 0, 1)

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define bounds for each coordinate (between 0 and 1)
    bounds = [(0, 1) for _ in range(32)]

    # Define constraints for boundary
    cons = [{'type': 'ineq', 'fun': lambda x: 1 - x[::2]},  # x <= 1
            {'type': 'ineq', 'fun': lambda x: x[::2]},     # x >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[1::2]}, # y <= 1
            {'type': 'ineq', 'fun': lambda x: x[1::2]}]    # y >= 0

    # Try multiple local optimizations from different starting points
    best_ratio = -np.inf
    best_points = None

    # Grid search over 5x5 uniform grid for better starting points
    test_grid = np.linspace(0.05, 0.95, 5)
    for i in range(5):
        for j in range(5):
            # Create a perturbed version of the grid point
            base_x = test_grid[i]
            base_y = test_grid[j]

            # Create a random perturbation around this grid point
            perturbation = np.random.normal(0, 0.05, (16, 2))
            perturbed_points = np.array([[base_x, base_y]] * 16) + perturbation
            perturbed_points = np.clip(perturbed_points, 0, 1)
            x_start = perturbed_points.flatten()

            try:
                result = minimize(objective, x_start, method='SLSQP', bounds=bounds,
                                constraints=cons, options={'maxiter': 100})
                if result.success:
                    # Calculate the actual ratio
                    final_points = result.x.reshape(-1, 2)
                    distances = pdist(final_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
            except:
                continue

    # Final optimization with the best starting point
    if best_points is not None:
        x_final = best_points.flatten()
        try:
            result = minimize(objective, x_final, method='SLSQP', bounds=bounds,
                            constraints=cons, options={'maxiter': 200})
            if result.success:
                final_points = result.x.reshape(-1, 2)
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0 and min_dist / max_dist > best_ratio:
                    best_points = final_points

        except:
            pass

    # Return the best solution found
    if best_points is None:
        # Fallback to the grid initialization if nothing worked
        best_points = initial_points

    return best_points

# EVOLVE-BLOCK-END