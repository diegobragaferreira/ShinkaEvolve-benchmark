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

    # Phase 2: Hybrid Optimization - Differential Evolution + Local Refinement

    # Objective function for optimization
    def objective_function(points_vec):
        # Reshape vector back to points array
        points = points_vec.reshape(-1, 2)

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

        # Return negative ratio (since we want to maximize ratio)
        return -ratio

    # Constraint function for bounds
    def constraint_func(points_vec):
        points = points_vec.reshape(-1, 2)
        # Check if any point is outside [0,1]x[0,1]
        violations = []
        for pt in points:
            if pt[0] < 0 or pt[0] > 1 or pt[1] < 0 or pt[1] > 1:
                violations.append(0)  # Violation
            else:
                violations.append(1)  # Valid
        return np.array(violations).mean()  # Should be 1 for valid solutions

    # Define bounds [0,1] for each coordinate
    bounds = [(0, 1) for _ in range(2*n)]

    # Use Differential Evolution for global search
    print("Starting Differential Evolution optimization...")
    de_result = differential_evolution(
        objective_function,
        bounds,
        maxiter=1000,  # Reduced iterations since we'll do local refinement
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )

    # Extract best solution from DE
    best_points_vec = de_result.x
    best_points = best_points_vec.reshape(-1, 2)

    # Apply local refinement with SLSQP
    print("Starting local SLSQP refinement...")

    # Define constraint dictionary for SLSQP
    constraints = {'type': 'ineq', 'fun': lambda x: constraint_func(x)}

    # Local optimization with SLSQP
    result = minimize(
        objective_function,
        best_points_vec,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10},
        tol=1e-10
    )

    # Final points after local refinement
    final_points = result.x.reshape(-1, 2)

    return final_points

# EVOLVE-BLOCK-END