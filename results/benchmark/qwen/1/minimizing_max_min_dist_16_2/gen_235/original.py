# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max

    # Create an initial good configuration based on hexagonal packing principles
    # Arrange points in a roughly hexagonal pattern within the unit square
    np.random.seed(42)

    # Generate a structured initial configuration that's close to optimal
    # Using a modified hexagonal grid pattern
    n_points = 16
    points = np.zeros((n_points, 2))

    # Create a hexagonal-like arrangement
    rows = 4
    cols = 4

    # Hexagonal grid with slight randomness to avoid perfect symmetry
    row_spacing = 1.0 / (rows - 1) if rows > 1 else 1.0
    col_spacing = 1.0 / (cols - 1) if cols > 1 else 1.0

    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= n_points:
                break
            x = j * col_spacing + (i % 2) * col_spacing * 0.5  # Offset every other row
            y = i * row_spacing
            points[i * cols + j] = [x + np.random.normal(0, 0.01), y + np.random.normal(0, 0.01)]

    # Ensure points stay within bounds
    points = np.clip(points, 0, 1)

    # Flatten for optimization
    x0 = points.flatten()

    # Define bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Use differential evolution for global optimization with better parameters
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,  # Increased iterations for better convergence
        popsize=20,   # Larger population size
        tol=1e-8,     # Tighter tolerance
        recombination=0.8,  # Higher recombination rate
        mutation=(0.5, 1.0),
        disp=False
    )

    # Return the optimized points
    points = result.x.reshape(-1, 2)

    return points


# EVOLVE-BLOCK-END