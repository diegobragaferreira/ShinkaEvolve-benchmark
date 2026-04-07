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
        # Reshape x into points
        points = x.reshape(-1, 2)

        # Compute pairwise distances using squareform for better numerical stability
        distances = squareform(pdist(points))

        # Zero out diagonal elements (distance to self)
        np.fill_diagonal(distances, np.inf)

        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max

    # Create a better initial configuration using a structured approach
    # Start with a hexagonal grid pattern which tends to give good spread
    np.random.seed(42)

    # Generate points in a hexagonal pattern
    points = []
    rows, cols = 4, 4
    sqrt3 = np.sqrt(3)
    spacing = 0.8  # Adjust spacing to fit well in [0,1] square

    # Create hexagonal grid
    for i in range(rows):
        for j in range(cols):
            if len(points) >= 16:
                break
            # Offset every other row for hexagonal packing
            x = j * spacing + (i % 2) * spacing * 0.5
            y = i * spacing * sqrt3 / 2

            # Scale to fit within unit square [0.05, 0.95] to avoid boundary issues
            x_scaled = 0.05 + (x / (spacing * cols)) * 0.9
            y_scaled = 0.05 + (y / (spacing * rows * sqrt3 / 2)) * 0.9

            # Add small random perturbation to avoid perfect symmetries
            x_scaled += np.random.normal(0, 0.01)
            y_scaled += np.random.normal(0, 0.01)

            points.append([x_scaled, y_scaled])

    points = np.array(points[:16])

    # Ensure points stay within bounds to avoid numerical issues at boundaries
    points = np.clip(points, 0.05, 0.95)

    # Flatten for optimization
    x0 = points.flatten()

    # Define bounds for each coordinate (0.05 to 0.95 to provide padding from boundaries)
    bounds = [(0.05, 0.95) for _ in range(32)]  # 16 points * 2 coordinates each

    # First stage: Use differential evolution for global optimization
    de_result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,  # Increased iterations for better convergence
        popsize=25,   # Larger population size for better exploration
        tol=1e-8,     # Tighter tolerance for global search
        recombination=0.9,  # Higher recombination rate
        mutation=(0.8, 1.0),  # Larger mutation range
        disp=False
    )

    # Second stage: Local refinement with L-BFGS-B
    lbfgs_result = minimize(
        objective,
        de_result.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14},
        callback=None
    )

    # Return the refined points from the local optimization
    points = lbfgs_result.x.reshape(-1, 2)

    return points


# EVOLVE-BLOCK-END