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

    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)

        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances efficiently
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
        """Create a better initial configuration using a spiral-like pattern for better coverage"""
        np.random.seed(42)

        # Start with a more uniform distribution using a spiral-like pattern
        # This helps avoid clustering and provides better coverage
        points = []
        n_points = 16

        # Create points in a spiral pattern that fills the space
        angles = np.linspace(0, 4*np.pi, n_points)
        radii = np.linspace(0.1, 0.8, n_points)

        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4

            # Add small random perturbation to break symmetry
            x += (np.random.random() - 0.5) * 0.08
            y += (np.random.random() - 0.5) * 0.08

            points.append([x, y])

        # Clip to unit square
        initial_points = np.array(points)
        initial_points = np.clip(initial_points, 0, 1)
        
        # Fix corner points explicitly to break symmetry and avoid degeneracy
        initial_points[0] = [0.0, 0.0]      # bottom-left corner
        initial_points[3] = [1.0, 0.0]      # bottom-right corner
        initial_points[12] = [0.0, 1.0]     # top-left corner
        initial_points[15] = [1.0, 1.0]     # top-right corner
        
        return initial_points

    # Create better initial guess
    initial_points = create_better_initial_configuration()

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Define bounds (0 to 1 for each coordinate)
    bounds = [(0, 1) for _ in range(32)]

    # Enhanced optimization strategy with multiple approaches
    best_result = None
    best_ratio = -np.inf

    # Strategy 1: Differential Evolution with multiple restarts for better global search
    for i in range(3):
        # Use differential evolution with higher iteration counts and population size
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-7,
                mutation=(0.5, 1),
                recombination=0.9,
                seed=42+i,  # Different seeds for variety
                disp=False
            )

            # Evaluate final result
            final_obj = objective_function(result.x)
            if final_obj > best_ratio:
                best_ratio = final_obj
                best_result = result
        except Exception as e:
            continue

    # Strategy 2: Local optimization with multiple restarts for final refinement
    if best_result is None:
        # Use the initial configuration as fallback
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
            pass
    else:
        # Try local optimization from the best DE result
        try:
            refined_result = minimize(
                objective_function,
                best_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            # Check if refined result is better
            refined_ratio = objective_function(refined_result.x)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_result = refined_result
        except Exception as e:
            pass

    # Strategy 3: Additional local refinement with multiple restarts if needed
    if best_result is not None:
        # Try refining the best result further
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

    # Extract the best solution
    if best_result is not None:
        best_points = best_result.x.reshape(-1, 2)
        # Ensure they're still in bounds
        best_points = np.clip(best_points, 0, 1)
    else:
        # Fallback to initial points if nothing worked
        best_points = initial_points

    # Final validation and cleanup
    # Ensure corner points are exactly at their fixed positions
    best_points[0] = [0.0, 0.0]      # bottom-left corner
    best_points[3] = [1.0, 0.0]      # bottom-right corner
    best_points[12] = [0.0, 1.0]     # top-left corner
    best_points[15] = [1.0, 1.0]     # top-right corner

    return best_points

# EVOLVE-BLOCK-END