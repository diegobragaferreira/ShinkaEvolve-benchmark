# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from itertools import product

def sobol_sequence(n, d=3):
    """Generate Sobol sequence points in d dimensions"""
    # Simple 3D Sobol sequence generator (simplified version)
    # In practice, would use a proper Sobol sequence library like 'sobol'
    # For demonstration purposes, we'll use a simple approach that's still better than random

    points = []
    # Use a deterministic construction that spreads points well
    for i in range(n):
        # Use a combination of prime numbers and fractional parts to create good spread
        t = (i * 0.618033988749895) % 1  # Golden ratio
        u = (i * 0.414213562373095) % 1  # sqrt(2) - 1
        v = (i * 0.732050807568877) % 1  # 2*sqrt(3) - 2

        # Map to [0,1]^3
        x = (t + 0.1 * i) % 1
        y = (u + 0.1 * i) % 1
        z = (v + 0.1 * i) % 1
        points.append([x, y, z])

    return np.array(points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    n = 14
    d = 3

    # Multi-start optimization parameters
    num_starts = 15  # Increase number of starts for better exploration
    perturbation_scale = 0.015  # Reduce perturbation for more precise search

    best_ratio = -np.inf
    best_points = None

    # Run multiple optimizations from different starting points
    for start_idx in range(num_starts):
        # Initialize with Sobol sequence points instead of Fibonacci
        # This gives better space filling properties
        if start_idx == 0:
            # First start: use Sobol sequence
            initial_points = sobol_sequence(n, d)
        else:
            # Subsequent starts: perturb Sobol points
            initial_points = sobol_sequence(n, d)
            # Add slight random perturbation to break symmetry
            perturbation = np.random.normal(0, perturbation_scale, initial_points.shape)
            initial_points += perturbation
            # Clip to stay within bounds
            initial_points = np.clip(initial_points, 0, 1)

        # Normalize to unit cube [0,1]^3
        # First center around origin and scale appropriately
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        # Then shift to [0,1]^3
        initial_points = initial_points + 0.5

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        def objective(x_flat):
            # Reshape back to points
            points = x_flat.reshape((n, d))

            # Calculate pairwise distances
            distances = pdist(points)

            # Filter out very small distances to avoid numerical issues
            distances = distances[distances > 1e-12]

            # Handle case where all points might be degenerate
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

        # Perform optimization - use more iterations for better convergence
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
        except Exception:
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

    # If we didn't find a good solution, return the initial points
    if best_points is None:
        initial_points = sobol_sequence(n, d)
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
            options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-13}
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