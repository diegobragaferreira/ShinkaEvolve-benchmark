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

    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)

        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio since we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -np.inf

    def create_better_initial_configuration():
        """Create a better initial configuration using a more structured approach"""
        np.random.seed(42)

        # Start with a more strategic distribution
        # Use a pattern that avoids clustering and ensures better coverage
        points = []

        # Create points in a hexagonal-like pattern for better spacing
        # Main grid points
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                grid_points.append([x, y])

        # Apply small random perturbations to make it non-symmetric
        for i, point in enumerate(grid_points):
            # Use different random seeds for each point to ensure diversity
            np.random.seed(42 + i)
            perturbation_magnitude = 0.03
            point[0] += (np.random.random() - 0.5) * perturbation_magnitude
            point[1] += (np.random.random() - 0.5) * perturbation_magnitude
            points.append(point)

        # Convert to numpy array
        initial_points = np.array(points)

        # Force the four corners to be fixed to break symmetry and avoid degeneracy
        initial_points[0] = [0.0, 0.0]      # Bottom-left
        initial_points[3] = [1.0, 0.0]      # Bottom-right
        initial_points[12] = [0.0, 1.0]     # Top-left
        initial_points[15] = [1.0, 1.0]     # Top-right

        # Ensure all points are within bounds
        initial_points = np.clip(initial_points, 0, 1)

        return initial_points

    # Create better initial guess
    initial_points = create_better_initial_configuration()

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Define bounds (0 to 1 for each coordinate)
    bounds = [(0, 1) for _ in range(32)]

    # Multi-stage optimization approach
    best_result = None
    best_ratio = -np.inf

    # Strategy 1: Multiple differential evolution runs with different parameters
    for run in range(3):
        # Use different random seeds for variety
        de_seed = 42 + run

        # Try with different population sizes and mutation rates
        if run == 0:
            popsize = 15
            mutation = (0.5, 1)
            recombination = 0.7
            maxiter = 75
        elif run == 1:
            popsize = 20
            mutation = (0.7, 1)
            recombination = 0.9
            maxiter = 50
        else:  # run == 2
            popsize = 10
            mutation = (0.3, 1)
            recombination = 0.5
            maxiter = 100

        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-7,
                mutation=mutation,
                recombination=recombination,
                seed=de_seed,
                disp=False
            )

            # Evaluate final result
            final_obj = objective_function(result.x)
            if final_obj > best_ratio:
                best_ratio = final_obj
                best_result = result
        except Exception as e:
            # If DE fails, continue with other strategies
            continue

    # Strategy 2: Local optimization if DE didn't work well enough
    if best_result is None or best_ratio < -0.2:  # If ratio is still very poor
        # Try local optimization with the better initial configuration
        try:
            result = minimize(
                objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            final_obj = objective_function(result.x)
            if final_obj > best_ratio:
                best_ratio = final_obj
                best_result = result
        except Exception as e:
            # Fall back to initial configuration if optimization fails
            pass

    # Strategy 3: Additional local refinement with multiple restarts
    if best_result is not None:
        # Refine the best result found so far
        try:
            refined_result = minimize(
                objective_function,
                best_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )

            final_obj = objective_function(refined_result.x)
            if final_obj > best_ratio:
                best_ratio = final_obj
                best_result = refined_result
        except Exception as e:
            pass

    # Extract the best solution found
    if best_result is not None:
        best_points = best_result.x.reshape(-1, 2)
        # Ensure they're still in bounds
        best_points = np.clip(best_points, 0, 1)
    else:
        # If no optimization was successful, return the initial configuration
        best_points = initial_points

    return best_points

# EVOLVE-BLOCK-END