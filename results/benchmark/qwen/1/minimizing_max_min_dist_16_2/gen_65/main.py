# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Better initialization using a more evenly distributed pattern
    n = 16

    # Create a more uniform distribution by using a modified grid pattern
    # Arrange points in a 4x4 grid but offset to avoid clustering
    points = np.zeros((n, 2))
    row_size = 4
    col_size = 4
    spacing_x = 1.0 / (col_size - 1) if col_size > 1 else 1.0
    spacing_y = 1.0 / (row_size - 1) if row_size > 1 else 1.0

    idx = 0
    for row in range(row_size):
        for col in range(col_size):
            if idx < n:
                x = col * spacing_x
                y = row * spacing_y
                # Add slight jitter to break perfect symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                points[idx] = [x, y]
                idx += 1

    # Clip to ensure points stay within bounds
    points = np.clip(points, 0, 1)

    # Phase 2: Hybrid Optimization with improved stability and fallback mechanisms

    # Objective function for optimization - improved numerical stability
    def objective_function(points_vec):
        # Reshape vector back to points array
        points = points_vec.reshape(-1, 2)

        # Ensure points are within bounds
        points = np.clip(points, 0, 1)

        # Compute pairwise distances using squareform for better numerical stability
        try:
            distances = squareform(pdist(points))

            # Mask diagonal elements (distance to self is 0)
            np.fill_diagonal(distances, np.inf)

            # Get min and max distances
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # Avoid division by zero or near-zero values
            if max_dist < 1e-12:
                ratio = 0
            else:
                ratio = min_dist / max_dist

            # Return negative ratio (since we want to maximize ratio)
            return -ratio

        except Exception:
            return 0

    # Improved constraint function for bounds with proper epsilon padding
    def constraint_func(points_vec):
        points = points_vec.reshape(-1, 2)
        epsilon = 1e-8

        # Check if any point is outside [0+eps, 1-eps] range
        valid_count = 0
        for pt in points:
            if 0 + epsilon <= pt[0] <= 1 - epsilon and 0 + epsilon <= pt[1] <= 1 - epsilon:
                valid_count += 1

        return valid_count / len(points)  # Should be 1 for valid solutions

    # Define bounds [0,1] for each coordinate with epsilon padding
    bounds = [(0, 1) for _ in range(2*n)]

    # Use Differential Evolution for global search with improved parameters
    print("Starting Differential Evolution optimization...")
    de_start_time = time.time()

    de_result = differential_evolution(
        objective_function,
        bounds,
        maxiter=1000,  # Reduced iterations for faster processing
        popsize=20,     # Standard population size
        mutation=(0.5, 1.0),  # Standard mutation range
        recombination=0.7,    # Standard recombination rate
        seed=42,
        disp=True,
        strategy='best1bin'
    )

    de_end_time = time.time()
    print(f"Differential Evolution completed in {de_end_time - de_start_time:.2f} seconds")

    # Extract best solution from DE
    best_points_vec = de_result.x
    best_points = best_points_vec.reshape(-1, 2)

    # Apply local refinement with SLSQP - improved with early stopping
    print("Starting local SLSQP refinement...")
    slsqp_start_time = time.time()

    # Monitor convergence for early stopping
    prev_obj_value = float('inf')
    patience_counter = 0
    max_patience = 50

    # Define constraint dictionary for SLSQP
    constraints = {'type': 'ineq', 'fun': lambda x: constraint_func(x)}

    # Use L-BFGS-B as fallback if SLSQP fails or gets stuck
    try:
        # First try SLSQP
        result = minimize(
            objective_function,
            best_points_vec,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10},
            tol=1e-10
        )

        # Check if result is acceptable
        if result.success and constraint_func(result.x) > 0.9:
            final_points = result.x.reshape(-1, 2)
        else:
            raise ValueError("SLSQP did not converge properly")

    except Exception as e:
        print(f"SLSQP failed, attempting L-BFGS-B fallback: {e}")
        try:
            # Fallback to L-BFGS-B
            result = minimize(
                objective_function,
                best_points_vec,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            final_points = result.x.reshape(-1, 2)
        except Exception as lbfgs_error:
            print(f"L-BFGS-B also failed: {lbfgs_error}")
            # Last resort: return the DE result if everything else fails
            final_points = best_points

    slsqp_end_time = time.time()
    print(f"SLSQP/L-BFGS-B refinement completed in {slsqp_end_time - slsqp_start_time:.2f} seconds")

    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END