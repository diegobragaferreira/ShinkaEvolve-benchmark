# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time

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

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance with explicit numerical stability."""
        if len(points) < 2:
            return 0.0

        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    # Initialize with a more informed structured approach based on known good configurations
    def initialize_better_points():
        """Create a better initial configuration using known good patterns for 16 points."""
        np.random.seed(42)

        # Use a 4x4 grid pattern as base
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)

        # Create grid points
        X, Y = np.meshgrid(grid_x, grid_y)
        points = np.column_stack([X.ravel(), Y.ravel()])

        # Add small random perturbations to break symmetry and improve chances of better solutions
        noise_magnitude = 0.01
        noise = np.random.normal(0, noise_magnitude, points.shape)
        points += noise

        # Clip to ensure bounds
        points = np.clip(points, 0, 1)

        # If we somehow have too many points, truncate
        if len(points) > 16:
            points = points[:16]
        elif len(points) < 16:
            # Fill with additional random points if needed
            additional = np.random.rand(16 - len(points), 2)
            points = np.vstack([points, additional])

        return points

    # Set up bounds (0 to 1 for each coordinate)
    bounds = [(0, 1)] * 32

    # Two-phase optimization approach
    best_points = None
    best_ratio = -np.inf

    # Phase 1: More aggressive global search
    try:
        # Use a more extensive global search with better parameters
        result_global = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=300,  # Increased iterations
            popsize=25,   # Larger population size
            atol=1e-10,   # Tighter absolute tolerance
            rtol=1e-10,   # Tighter relative tolerance
            mutation=(0.8, 1.0),
            recombination=0.9
        )

        if result_global.success:
            # Phase 2: Multiple refinement strategies
            refinement_strategies = [
                {
                    'method': 'L-BFGS-B',
                    'options': {'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                },
                {
                    'method': 'SLSQP',
                    'options': {'maxiter': 300, 'ftol': 1e-12}
                }
            ]

            for strategy in refinement_strategies:
                try:
                    refined = minimize(
                        objective,
                        result_global.x,
                        method=strategy['method'],
                        bounds=bounds,
                        options=strategy['options']
                    )

                    if refined.success:
                        final_points = refined.x.reshape(-1, 2)
                        ratio = calculate_min_max_ratio(final_points)

                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()

                except Exception:
                    continue

    except Exception:
        pass

    # If we haven't found a good solution yet, try direct optimization from good initialization
    if best_points is None:
        try:
            # Try optimization from the structured initialization directly
            x0 = initialize_better_points().flatten()

            # Global optimization with different parameters
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                atol=1e-12,
                rtol=1e-12
            )

            if result.success:
                # Final refinement with multiple local methods
                refined = minimize(
                    objective,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-15}
                )

                if refined.success:
                    final_points = refined.x.reshape(-1, 2)
                    ratio = calculate_min_max_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

        except Exception:
            pass

    # Final fallback to initialization if everything failed
    if best_points is None:
        best_points = initialize_better_points()

    # Ensure points are within valid bounds
    best_points = np.clip(best_points, 0, 1)

    return best_points

# EVOLVE-BLOCK-END