# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Generate initial points using Sobol sequence for better space filling
    def sobol_init(samples=14, seed=42):
        # Using a simple low-discrepancy sequence approach
        # For 3D, we can generate points using a modified Fibonacci-like approach
        # but with better distribution properties
        points = []
        # Generate points in a way that distributes them more evenly
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        # Use a different approach for better distribution in 3D
        for i in range(samples):
            # Modified approach: use a more uniform distribution
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            # Use a more structured approach
            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        # Add some random perturbation to avoid symmetries
        points = np.array(points)
        # Add controlled noise
        noise = np.random.normal(0, 0.02, points.shape)
        points = points + noise

        return points

    # Alternative: create Sobol-like distribution manually
    def sobol_like_distribution(samples=14):
        # Generate points using a modified approach that spreads them well
        points = []
        # Generate points distributed on sphere and then project to cube
        for i in range(samples):
            # Distribute points along latitude bands
            lat = np.arccos(1 - 2*(i/(samples-1)))  # Uniform on sphere
            lon = np.sqrt(samples) * lat  # Adjust longitude to spread well

            x = np.sin(lat) * np.cos(lon)
            y = np.sin(lat) * np.sin(lon)
            z = np.cos(lat)
            points.append([x, y, z])
        return np.array(points)

    # Try multiple initialization strategies
    init_strategies = [
        sobol_like_distribution,
        lambda s: sobol_init(s),
        lambda s: np.random.rand(s, 3)  # Random initialization as fallback
    ]

    best_ratio = -np.inf
    best_points = None

    # Multi-start optimization with different initializations
    num_starts = 10
    for start_idx in range(num_starts):
        # Select initialization strategy
        if start_idx < len(init_strategies):
            init_func = init_strategies[start_idx]
        else:
            init_func = lambda s: np.random.rand(s, 3)  # fallback

        # Get initial points
        initial_points = init_func(n)

        # Normalize to unit cube [0,1]^3
        # First center around origin and scale appropriately
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        # Then shift to [0,1]^3
        initial_points = initial_points + 0.5

        # Add slight random perturbation to break symmetry
        if start_idx > 0:
            perturbation = np.random.normal(0, 0.01, initial_points.shape)
            initial_points += perturbation
            # Clip to stay within bounds
            initial_points = np.clip(initial_points, 0, 1)

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        def objective(x_flat):
            # Reshape back to points
            points = x_flat.reshape((n, d))

            # Calculate pairwise distances
            distances = pdist(points)

            # Filter out zero distances to avoid numerical issues
            distances = distances[distances > 1e-12]

            if len(distances) == 0:
                return -np.inf

            # Compute min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Avoid division by zero
            if d_max == 0:
                return -np.inf

            # Minimize negative of ratio (since we want to maximize ratio)
            ratio = d_min / d_max

            # Return negative because we're minimizing
            return -ratio

        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(n * d)]

        # Perform optimization
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
            )
        except Exception:
            # Fallback to initial points if optimization fails
            continue

        # Extract optimized points
        optimized_points = result.x.reshape((n, d))

        # Ensure all points are within [0,1]^3
        optimized_points = np.clip(optimized_points, 0, 1)

        # Calculate the actual ratio for this optimization run
        final_distances = pdist(optimized_points)
        final_distances = final_distances[final_distances > 1e-12]

        if len(final_distances) > 0:
            d_min = np.min(final_distances)
            d_max = np.max(final_distances)
            if d_max > 0:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

    # If we didn't find a good solution, return the initial points from Sobol
    if best_points is None:
        initial_points = sobol_like_distribution(n)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        initial_points = initial_points + 0.5
        return initial_points

    # Apply final refinement with L-BFGS-B on the best solution found
    final_flat = best_points.flatten()

    # Refine with a more aggressive optimization
    try:
        refined_result = minimize(
            objective,
            final_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        refined_points = refined_result.x.reshape((n, d))
        refined_points = np.clip(refined_points, 0, 1)

        # Final check to make sure we have a valid solution
        final_distances = pdist(refined_points)
        final_distances = final_distances[final_distances > 1e-12]

        if len(final_distances) > 0 and np.min(final_distances) > 1e-12:
            return refined_points
        else:
            return best_points
    except Exception:
        # If refinement fails, return the best solution so far
        return best_points

# EVOLVE-BLOCK-END