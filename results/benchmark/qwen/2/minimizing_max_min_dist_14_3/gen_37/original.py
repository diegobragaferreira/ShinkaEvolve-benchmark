# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape the flattened array back to 14x3 points
        points = x.reshape((14, 3))

        # Calculate pairwise distances
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        # The ratio is d_min / d_max
        return -(d_min / d_max)

    # Set up bounds for each coordinate (0 to 1 for unit cube)
    bounds = [(0, 1) for _ in range(14 * 3)]

    # Use a more intelligent initial guess
    # Start with a regular icosahedron-like structure projected onto a sphere
    np.random.seed(42)

    # Generate initial points using a more structured approach
    # This helps the optimizer converge faster
    initial_points = []
    for i in range(14):
        # Distribute points somewhat evenly
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        initial_points.append([x, y, z])

    # Scale and shift to fit in unit cube
    initial_points = np.array(initial_points)
    initial_points = (initial_points - np.min(initial_points, axis=0)) / (np.max(initial_points, axis=0) - np.min(initial_points, axis=0))
    initial_points = initial_points * 0.9 + 0.05  # Shift to [0.05, 0.95] range

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Run optimization with limited time and iterations
    start_time = time.time()
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False,
        tol=1e-6
    )

    # Extract final points
    final_points = result.x.reshape((14, 3))

    # Ensure points stay within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points


# EVOLVE-BLOCK-END