# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Calculate pairwise distances using squareform for stability
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -1.0  # Avoid division by zero or invalid distances
        return -d_min / d_max

    def constraint_bounds(x_flat):
        # Ensure all points are within [0,1] x [0,1]
        points = x_flat.reshape(-1, 2)
        # Return constraints: lower bound (-points) and upper bound (points - 1)
        lower_bound = -points.flatten()
        upper_bound = points.flatten() - 1
        return np.concatenate([lower_bound, upper_bound])

    # Create initial configuration using improved hexagonal-like arrangement
    np.random.seed(42)

    # Generate points in a more optimized hexagonal pattern with better spacing
    points = []
    rows = 4
    cols = 4

    # Create a grid with better spacing and perturbations
    for i in range(rows):
        for j in range(cols):
            # Hexagonal grid with proper spacing
            x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
            y = i / (rows - 1) if rows > 1 else 0.5

            # Add more substantial but controlled random perturbation
            x += (np.random.rand() - 0.5) * 0.15
            y += (np.random.rand() - 0.5) * 0.15

            # Ensure points stay within boundaries with epsilon padding
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)

            points.append([x, y])

    points = np.array(points[:16])  # Ensure exactly 16 points

    # Flatten the points for optimization
    x0 = points.flatten()

    # Define bounds for each coordinate [0, 1] with small epsilon padding
    bounds = [(1e-6, 1-1e-6) for _ in range(32)]

    # Adaptive optimization parameters
    max_iter_lbfgs = 1000
    ftol = 1e-10
    gtol = 1e-10

    def adaptive_minimize(obj_func, x0, bounds, maxiter, ftol, gtol):
        """Minimize with adaptive stopping criteria"""
        previous_obj_val = float('inf')
        consecutive_no_improvement = 0
        max_no_improvement = 20

        # Use a callback to track progress
        def callback(xk):
            nonlocal previous_obj_val, consecutive_no_improvement
            obj_val = obj_func(xk)
            if abs(previous_obj_val - obj_val) < 1e-12:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            previous_obj_val = obj_val

        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                callback=callback if consecutive_no_improvement < max_no_improvement else None
            )
            return result
        except Exception:
            return None

    final_points = None
    best_objective_value = float('inf')

    # Phase 1: L-BFGS-B optimization with adaptive stopping
    try:
        result = adaptive_minimize(objective, x0, bounds, max_iter_lbfgs, ftol, gtol)

        if result and result.success:
            current_obj_value = -objective(result.x)  # Convert back to positive ratio
            if current_obj_value < best_objective_value:
                best_objective_value = current_obj_value
                final_points = result.x.reshape(-1, 2)
        else:
            raise RuntimeError("L-BFGS-B failed")

    except Exception as e:
        print(f"L-BFGS-B failed: {e}")

    # Phase 2: Differential Evolution for global search (only if needed)
    if final_points is None:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # Increased iterations for better global search
                popsize=20,   # Larger population size
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )

            current_obj_value = -objective(de_result.x)
            if current_obj_value < best_objective_value:
                best_objective_value = current_obj_value
                final_points = de_result.x.reshape(-1, 2)

        except Exception as e:
            print(f"Differential Evolution failed: {e}")

    # Phase 3: Final local refinement with L-BFGS-B if no good solution found yet
    if final_points is None:
        try:
            final_points = x0.reshape(-1, 2)
        except Exception:
            # Last resort: return initial configuration
            final_points = points

    # Ensure final points respect bounds
    if final_points is not None:
        final_points = np.clip(final_points, 1e-6, 1-1e-6)

    return final_points if final_points is not None else points

# EVOLVE-BLOCK-END