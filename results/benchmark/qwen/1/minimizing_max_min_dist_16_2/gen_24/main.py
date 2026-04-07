# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Initialize points using hexagonal grid pattern for better starting configuration
    n = 16
    points = np.zeros((n, 2))

    # Create hexagonal grid pattern
    rows = 4
    cols = 4
    spacing = 0.25

    row_offset = 0.0
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n:
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1

    # Adjust points to fit within [0,1]x[0,1] and add some randomness
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

    # Phase 2: Hybrid optimization - Global search with differential evolution, then local refinement

    # Define bounds for optimization (each point has x and y coordinates in [0,1])
    bounds = [(0, 1) for _ in range(2*n)]

    # Objective function for optimization - minimizing negative ratio (to maximize ratio)
    def objective_function-flat(params):
        # Reshape params into points
        points = params.reshape(-1, 2)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            ratio = 0
        else:
            ratio = min_dist / max_dist

        # Negative ratio since we're minimizing
        return -ratio

    # Phase 2a: Global optimization using differential evolution
    print("Starting differential evolution...")
    de_result = differential_evolution(
        objective_function-flat,
        bounds,
        maxiter=1000,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )

    # Extract best solution from DE
    best_points = de_result.x.reshape(-1, 2)

    # Phase 2b: Local refinement using L-BFGS-B
    print("Starting local refinement with L-BFGS-B...")

    # Create bounds for L-BFGS-B (same as before but as tuple of tuples)
    lb = [0.0] * (2*n)
    ub = [1.0] * (2*n)

    # Optimize using L-BFGS-B for final refinement
    result = minimize(
        objective_function-flat,
        best_points.flatten(),
        method='L-BFGS-B',
        bounds=list(zip(lb, ub)),
        options={'maxiter': 5000, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    # Final best points
    final_points = result.x.reshape(-1, 2)

    # Ensure final points are within bounds (due to numerical errors)
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END