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
        # Reshape into points
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -1e10

        # Return negative ratio to minimize (we want to maximize ratio)
        return -d_min / d_max

    def initialize_points():
        """Initialize points using a structured approach for better starting configuration."""
        # Create a grid-like pattern with some randomness
        np.random.seed(42)

        # Create a 4x4 grid pattern (16 points)
        grid_size = 4
        x = np.linspace(0.1, 0.9, grid_size)
        y = np.linspace(0.1, 0.9, grid_size)

        # Generate grid points
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Add small random perturbations to break symmetry
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise

        # Clip to ensure points stay within bounds
        points = np.clip(points, 0, 1)

        return points

    # Set up bounds (0 to 1 for each coordinate)
    bounds = [(0, 1)] * 32

    # Try multiple optimization approaches
    best_ratio = -np.inf
    best_points = None

    # Method 1: Differential Evolution with adaptive parameters
    try:
        # Global optimization with differential evolution
        result_de = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            atol=1e-8,
            rtol=1e-8,
            mutation=(0.7, 1.0),
            recombination=0.7
        )

        if result_de.success:
            # Local refinement with L-BFGS-B
            refined = minimize(
                objective,
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if refined.success:
                final_points = refined.x.reshape(-1, 2)
                distances = squareform(pdist(final_points))
                np.fill_diagonal(distances, np.inf)
                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

    except Exception:
        pass

    # Method 2: Direct optimization from structured initialization
    if best_points is None:
        try:
            # Start with structured initialization
            x0 = initialize_points().flatten()

            # Global optimization
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                atol=1e-8,
                rtol=1e-8
            )

            if result.success:
                # Local refinement
                refined = minimize(
                    objective,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10}
                )

                if refined.success:
                    final_points = refined.x.reshape(-1, 2)
                    distances = squareform(pdist(final_points))
                    np.fill_diagonal(distances, np.inf)
                    d_min = np.min(distances)
                    d_max = np.max(distances)

                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()

        except Exception:
            pass

    # Fallback to initialization if optimization fails
    if best_points is None:
        points = initialize_points()
    else:
        points = best_points

    # Ensure points are within valid bounds
    points = np.clip(points, 0, 1)

    return points


# EVOLVE-BLOCK-END