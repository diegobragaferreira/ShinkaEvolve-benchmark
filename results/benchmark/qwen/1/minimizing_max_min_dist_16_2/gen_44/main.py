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

    # Create a better initial configuration using a known good starting point
    # Based on known optimal configurations for 16 points in 2D
    np.random.seed(42)

    # Start with a more sophisticated configuration inspired by circle packing and
    # known solutions to point dispersion problems
    points = np.array([
        [0.5, 0.5],      # Center point
        [0.2, 0.2],      # Corner points
        [0.8, 0.2],
        [0.8, 0.8],
        [0.2, 0.8],
        [0.1, 0.5],      # Edge points
        [0.9, 0.5],
        [0.5, 0.1],
        [0.5, 0.9],
        [0.3, 0.3],      # Diagonal points
        [0.7, 0.3],
        [0.7, 0.7],
        [0.3, 0.7],
        [0.15, 0.35],    # Additional strategic points
        [0.85, 0.65],
        [0.65, 0.15]
    ])

    # Add small random perturbations to avoid degenerate cases
    points += np.random.normal(0, 0.01, points.shape)

    # Ensure points stay within bounds
    points = np.clip(points, 0, 1)

    # Flatten for optimization
    x0 = points.flatten()

    # Define bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # First stage: Use differential evolution for global optimization with enhanced parameters
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

    # Second stage: Multiple local refinement approaches
    # Try several local optimization methods to ensure we get a good solution

    # Method 1: L-BFGS-B with stricter tolerances
    lbfgs_result = minimize(
        objective,
        de_result.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14},
        callback=None
    )

    # Method 2: SLSQP as a backup
    slsqp_result = minimize(
        objective,
        de_result.x,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
        callback=None
    )

    # Choose the best result from all local optimizations
    results = [lbfgs_result, slsqp_result]
    best_result = min(results, key=lambda r: r.fun)

    # Return the refined points from the best local optimization
    points = best_result.x.reshape(-1, 2)

    return points


# EVOLVE-BLOCK-END