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

    # Define multiple initial configurations to increase chance of finding better solution
    initial_configs = []

    # Configuration 1: Modified hexagonal grid with padding
    np.random.seed(42)
    points1 = np.zeros((16, 2))
    rows, cols = 4, 4
    row_spacing = 1.0 / (rows - 1) if rows > 1 else 1.0
    col_spacing = 1.0 / (cols - 1) if cols > 1 else 1.0

    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= 16:
                break
            x = j * col_spacing + (i % 2) * col_spacing * 0.5  # Offset every other row
            y = i * row_spacing
            points1[i * cols + j] = [x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005)]

    # Apply boundary padding to avoid hitting edges
    points1 = np.clip(points1, 0.01, 0.99)
    initial_configs.append(points1.flatten())

    # Configuration 2: Circle arrangement with some noise
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radii = 0.4 + 0.1 * np.sin(np.arange(16) * np.pi / 8)  # Varying radii for diversity
    center = np.array([0.5, 0.5])
    points2 = np.column_stack([center[0] + radii * np.cos(angles),
                              center[1] + radii * np.sin(angles)])
    points2 += np.random.normal(0, 0.01, points2.shape)
    points2 = np.clip(points2, 0.01, 0.99)
    initial_configs.append(points2.flatten())

    # Configuration 3: Grid with jittering
    points3 = np.random.rand(16, 2) * 0.8 + 0.1  # Spread across [0.1, 0.9] range
    points3 = np.clip(points3, 0.01, 0.99)
    initial_configs.append(points3.flatten())

    # Define bounds for each coordinate (with small padding to avoid boundary issues)
    bounds = [(0.01, 0.99) for _ in range(32)]  # 16 points * 2 coordinates each

    best_result = None
    best_value = float('inf')

    # Try multiple initial configurations
    for i, x0 in enumerate(initial_configs):
        # First stage: Use differential evolution for global optimization
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42+i,  # Different seed for each run
                maxiter=200,  # Increased iterations for better convergence
                popsize=25,   # Larger population size for better exploration
                tol=1e-8,     # Tighter tolerance for global search
                recombination=0.9,
                mutation=(0.8, 1.0),
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

            # Keep track of the best result
            if lbfgs_result.fun < best_value:
                best_value = lbfgs_result.fun
                best_result = lbfgs_result

        except Exception as e:
            print(f"Error in optimization attempt {i}: {e}")
            continue

    # If we found a valid result, return it; otherwise use default
    if best_result is not None:
        points = best_result.x.reshape(-1, 2)
    else:
        # Fallback to first configuration if all attempts failed
        points = initial_configs[0].reshape(-1, 2)

    return points


# EVOLVE-BLOCK-END