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
        # Reshape flat array back to 16x2 points
        points = x.reshape(-1, 2)

        # Compute pairwise distances using squareform for better numerical stability
        distances = pdist(points)

        # Avoid division by zero
        if len(distances) == 0 or len(distances) < 2:
            return -np.inf

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Handle case where all points are coincident or nearly coincident
        if d_max <= 1e-10:
            return -np.inf

        # Return negative ratio since we're minimizing
        return -d_min / d_max

    def create_initial_guesses():
        """Create multiple diverse initial guesses"""
        initial_guesses = []

        # 1. Random initialization
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_guesses.append(random_points.flatten())

        # 2. Grid initialization
        grid_points = np.array([[i/5, j/5] for i in range(5) for j in range(5) if i*5+j < 16]).reshape(-1, 2)
        # Pad if needed
        if len(grid_points) < 16:
            extra_points = np.random.rand(16 - len(grid_points), 2)
            grid_points = np.vstack([grid_points, extra_points])
        initial_guesses.append(grid_points.flatten())

        # 3. Spiral initialization
        angles = np.linspace(0, 4*np.pi, 16)
        radii = np.linspace(0, 0.4, 16)
        spiral_x = 0.5 + radii * np.cos(angles)
        spiral_y = 0.5 + radii * np.sin(angles)
        spiral_points = np.column_stack([spiral_x, spiral_y])
        initial_guesses.append(spiral_points.flatten())

        # 4. Another random initialization with different seed
        np.random.seed(123)
        random_points_2 = np.random.rand(16, 2)
        initial_guesses.append(random_points_2.flatten())

        return initial_guesses

    # Define bounds for all coordinates: [0, 1] for each coordinate
    bounds = [(0, 1) for _ in range(32)]

    # Create diverse initial guesses
    initial_guesses = create_initial_guesses()

    best_result = None
    best_ratio = -np.inf

    # Multi-start approach with different initializations
    for i, initial_guess in enumerate(initial_guesses):
        try:
            # Use differential evolution for global search
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42+i,
                maxiter=200,      # Increased iterations
                popsize=30,       # Larger population
                tol=1e-8,         # Tighter tolerance
                recombination=0.9, # Higher recombination rate
                mutation=(0.8, 1.0), # Different mutation strategy
                init=[initial_guess]  # Start with our custom guess
            )

            # Local refinement with L-BFGS-B
            refined_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500}
            )

            # If that doesn't work, try SLSQP as backup
            if not refined_result.success:
                refined_result = minimize(
                    objective,
                    de_result.x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500}
                )

            # Check if this is better
            if -refined_result.fun > best_ratio:
                best_ratio = -refined_result.fun
                best_result = refined_result.x

        except Exception as e:
            # Skip failed optimizations and continue with others
            continue

    # Final refinement with enhanced settings
    if best_result is not None:
        final_result = minimize(
            objective,
            best_result,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
        )

        # If L-BFGS fails, try SLSQP as final fallback
        if not final_result.success:
            final_result = minimize(
                objective,
                best_result,
                method='SLSQP',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
            )

        points = final_result.x.reshape(-1, 2)
    else:
        # Fallback to best initial guess if everything else failed
        points = initial_guesses[0].reshape(-1, 2)

    # Ensure all points are within [0,1]^2 (handle any boundary issues)
    points = np.clip(points, 0, 1)

    return points


# EVOLVE-BLOCK-END