# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.
    """

    n = 14

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value to avoid self-distance issues
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    # Generate improved initial points using a better distribution strategy
    # Use a combination of Fibonacci spiral and perturbation for better spread
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    # Generate Fibonacci points but with more careful spacing
    for i in range(n):
        # Better distribution using golden spiral with modified parameterization
        phi = np.arccos(1 - 2 * i / (n - 1))
        theta = 2 * np.pi * i / golden_ratio

        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append([x, y, z])

    initial_points = np.array(points)

    # Apply a more sophisticated optimization approach:
    # 1. First, use a global optimization approach to get close to good solutions
    # 2. Then apply local refinement with different methods

    # Multiple restarts with different strategies
    best_ratio = -np.inf
    best_points = initial_points.copy()

    # Strategy 1: Standard Fibonacci with random perturbations
    for restart in range(5):
        # Add small noise to break symmetry
        np.random.seed(restart)
        noisy_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)

        # Normalize again
        noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1, keepdims=True)

        # Flatten for optimization
        x0 = noisy_points.flatten()

        # Define constraints
        cons = {'type': 'eq', 'fun': constraint_func}

        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', constraints=cons,
                            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500})

            if result.success:
                optimized_points = result.x.reshape(-1, 3)

                # Calculate final ratio
                distances = squareform(pdist(optimized_points))
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

        except Exception:
            continue

    # Strategy 2: Use SLSQP method which might be better for this constrained problem
    for restart in range(3):
        # Start with a small perturbation of the best current solution
        np.random.seed(10 + restart)
        if restart == 0:
            # Use the already found best solution as starting point
            noisy_points = best_points + np.random.normal(0, 0.005, best_points.shape)
        else:
            # Start from a different random perturbation
            noisy_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)

        # Normalize again
        noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1, keepdims=True)

        # Flatten for optimization
        x0 = noisy_points.flatten()

        # Define constraints
        cons = {'type': 'eq', 'fun': constraint_func}

        # Optimize using SLSQP which is often better for constrained problems
        try:
            result = minimize(objective, x0, method='SLSQP', constraints=cons,
                            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500})

            if result.success:
                optimized_points = result.x.reshape(-1, 3)

                # Calculate final ratio
                distances = squareform(pdist(optimized_points))
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

        except Exception:
            continue

    # Final polishing with additional refinement if needed
    # If we haven't found a significantly better solution, let's try a more aggressive approach
    if best_ratio < 0.1:  # If the improvement is minimal, try harder
        # Use a more robust initialization and more refined optimization
        try:
            # Try using a small subset of points to create better starting configuration
            # and then optimize the full set

            # For demonstration, we'll just do one final optimization
            # with a better tolerance and more iterations
            final_x0 = best_points.flatten()
            cons = {'type': 'eq', 'fun': constraint_func}

            result = minimize(objective, final_x0, method='L-BFGS-B', constraints=cons,
                            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000})

            if result.success:
                final_points = result.x.reshape(-1, 3)

                # Check if this improves our solution
                distances = squareform(pdist(final_points))
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_points = final_points

        except Exception as e:
            warnings.warn(f"Final refinement failed: {e}")

    return best_points

# EVOLVE-BLOCK-END