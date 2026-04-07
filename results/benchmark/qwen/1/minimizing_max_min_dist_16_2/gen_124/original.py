# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        if len(distances) == 0 or np.allclose(distances, 0):
            return -1.0  # Worst possible case

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -1.0

        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are within [0+eps,1-eps] x [0+eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [eps, 1-eps]
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps

        # y coordinates in [eps, 1-eps]
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps

        return np.array(constraints)

    def bounded_objective(x):
        # Boundary checking with clamping to safe bounds
        eps = 1e-8
        points = np.clip(x.reshape(-1, 2), eps, 1-eps).flatten()
        return objective(points)

    # Create a sophisticated initial configuration inspired by hexagonal packing
    np.random.seed(42)

    # Generate a hexagonal-like structure that approximates optimal distribution
    # Arrange points in a pattern that balances density and spread
    initial_points = np.zeros((16, 2))

    # Create a modified hexagonal grid pattern
    rows = 4
    cols = 4

    # Hexagonal offset pattern
    x_spacing = 0.9 / (cols - 1) if cols > 1 else 0.5
    y_spacing = 0.9 / (rows - 1) if rows > 1 else 0.5

    offset_x = 0.05
    offset_y = 0.05

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < 16:
                # Apply hexagonal offset every other row
                x = offset_x + j * x_spacing
                y = offset_y + i * y_spacing
                if i % 2 == 1:  # Offset odd rows
                    x += x_spacing * 0.5
                initial_points[idx] = [x, y]
                idx += 1

    # Add slight randomization to break symmetry
    initial_points += np.random.normal(0, 0.005, (16, 2))

    # Clamp to valid bounds to ensure all points are inside [0,1]^2
    initial_points = np.clip(initial_points, 0, 1)

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define tighter bounds to avoid boundary issues
    bounds = [(1e-8, 1-1e-8) for _ in range(32)]

    # Use a hybrid optimization approach
    # Phase 1: Differential Evolution for global search
    try:
        de_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de_result = differential_evolution(
                bounded_objective,
                bounds,
                seed=42,
                maxiter=50,  # Reduced for faster execution
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
        de_time = time.time() - de_start

        if de_result.success and -de_result.fun > 0.15:
            x0 = de_result.x
        else:
            # If DE doesn't find good solution, use the structured start
            pass

    except Exception as e:
        print(f"Differential evolution failed: {e}")
        # Fall back to just using the structured initialization

    # Phase 2: Local optimization with adaptive iteration limits
    # More aggressive optimization with SLSQP for final refinement
    try:
        # Set up adaptive iteration count
        adaptive_iter = min(1000, max(100, int(1000 * (1 - (-objective(x0) / 0.5)))))

        result = minimize(
            bounded_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={
                'maxiter': adaptive_iter,
                'ftol': 1e-9,
                'gtol': 1e-9
            },
            callback=None
        )

        if result.success:
            x0 = result.x
        else:
            # If SLSQP fails, use L-BFGS-B as fallback
            lbfgs_result = minimize(
                bounded_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300}
            )
            if lbfgs_result.success:
                x0 = lbfgs_result.x

    except Exception as e:
        print(f"Local optimization failed: {e}")
        # Fallback to the structured initialization

    # Final refinement step with additional local optimization
    try:
        # Try SLSQP again with refined starting point
        final_result = minimize(
            bounded_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )

        if final_result.success:
            x0 = final_result.x
    except Exception as e:
        print(f"Final optimization failed: {e}")

    # Extract optimized points and ensure they are within bounds
    optimized_points = np.clip(x0.reshape(-1, 2), 1e-8, 1-1e-8)

    # Double-check constraint satisfaction
    if np.any(optimized_points < 1e-8) or np.any(optimized_points > 1-1e-8):
        print("Warning: Some points out of bounds after optimization")
        optimized_points = np.clip(optimized_points, 1e-8, 1-1e-8)

    return optimized_points


# EVOLVE-BLOCK-END