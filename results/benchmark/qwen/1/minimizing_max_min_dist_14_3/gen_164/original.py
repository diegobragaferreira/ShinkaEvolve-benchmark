# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

    # Generate initial points using improved Fibonacci spiral on unit sphere
    def fibonacci_sphere(samples=14):
        points = []
        # Golden angle for even distribution
        phi = np.pi * (3. - np.sqrt(5.))

        # Generate points with better spacing control
        for i in range(samples):
            # Parameterize along the sphere surface
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            # Use golden angle for even angular distribution
            theta = phi * i

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    # More advanced initialization with improved distribution
    def enhanced_fibonacci_init(samples=14):
        # Generate points using improved Fibonacci approach with jitter
        points = []
        phi = np.pi * (3. - np.sqrt(5.))

        for i in range(samples):
            # Distribute points more evenly
            y = 1 - (i / float(samples - 1)) * 2
            radius = np.sqrt(1 - y * y)

            # Add controlled jitter to improve distribution
            theta = phi * i + np.random.normal(0, 0.05)

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    # Multi-start optimization with adaptive refinement
    best_ratio = -np.inf
    best_points = None

    # Enhanced optimization parameters
    num_starts = 25  # Increased for better exploration
    base_perturbation = 0.025
    perturbation_decay = 0.93  # Slightly faster decay for better convergence
    min_perturbation = 0.001

    # Initialization strategies
    init_strategies = [
        enhanced_fibonacci_init,
        lambda s: fibonacci_sphere(s),
        lambda s: np.random.rand(s, 3)  # Random fallback
    ]

    start_time = time.time()

    for start_idx in range(num_starts):
        # Check time limit
        if time.time() - start_time > 350:
            break

        # Adaptive perturbation scaling
        current_perturbation = max(base_perturbation * (perturbation_decay ** start_idx),
                                 min_perturbation)

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

        # Add controlled perturbation to break symmetry (only for non-first starts)
        if start_idx > 0:
            perturbation = np.random.normal(0, current_perturbation, initial_points.shape)
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

        # Progressive optimization with fallback strategies
        current_bounds = bounds
        optimization_options = {
            'maxiter': 800,
            'ftol': 1e-10,  # Tighter tolerance than baseline
            'gtol': 1e-10
        }

        result = None
        try:
            # Try L-BFGS-B first (fast convergence)
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=current_bounds,
                options=optimization_options
            )

            # If L-BFGS-B fails, try SLSQP for better constraint handling
            if not result.success:
                result = minimize(
                    objective,
                    initial_flat,
                    method='SLSQP',
                    bounds=current_bounds,
                    options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-10}
                )

            # If still failing, try Nelder-Mead as fallback
            if not result.success:
                result = minimize(
                    objective,
                    initial_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'fatol': 1e-10, 'xatol': 1e-10}
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

    # If we didn't find a good solution, return the best initialization
    if best_points is None:
        initial_points = enhanced_fibonacci_init(n)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        initial_points = initial_points + 0.5
        return initial_points

    # Apply final multi-stage refinement
    refined_points = best_points.copy()

    # Stage 1: High precision L-BFGS-B (revised)
    try:
        final_flat = refined_points.flatten()
        refined_result = minimize(
            objective,
            final_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)

            # Validate improvement
            final_distances_new = pdist(candidate_points)
            final_distances_new = final_distances_new[final_distances_new > 1e-12]

            if len(final_distances_new) > 0:
                new_min = np.min(final_distances_new)
                new_max = np.max(final_distances_new)
                if new_max > 0:
                    new_ratio = new_min / new_max
                    if new_ratio > best_ratio:
                        refined_points = candidate_points
    except Exception:
        pass

    # Stage 2: Additional SLSQP refinement
    try:
        final_flat = refined_points.flatten()
        refined_result = minimize(
            objective,
            final_flat,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)

            # Validate improvement
            final_distances_new = pdist(candidate_points)
            final_distances_new = final_distances_new[final_distances_new > 1e-12]

            if len(final_distances_new) > 0:
                new_min = np.min(final_distances_new)
                new_max = np.max(final_distances_new)
                if new_max > 0:
                    new_ratio = new_min / new_max
                    if new_ratio > best_ratio:
                        refined_points = candidate_points
    except Exception:
        pass

    # Final validation to make sure we have a valid solution
    final_distances = pdist(refined_points)
    final_distances = final_distances[final_distances > 1e-12]

    if len(final_distances) > 0 and np.min(final_distances) > 1e-12:
        return refined_points
    else:
        return best_points

# EVOLVE-BLOCK-END