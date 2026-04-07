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

    # Enhanced Fibonacci spiral initialization with jitter
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Add jitter to improve distribution
            theta = phi * i + np.random.normal(0, 0.05)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)

    # Sobol-like distribution for better space-filling
    def sobol_like_distribution(samples=14):
        points = []
        for i in range(samples):
            # Use a pattern inspired by low-discrepancy sequences
            y = 1 - (i / float(samples - 1)) * 2
            radius = np.sqrt(1 - y * y)
            
            # Apply modified golden angle with additional perturbation
            phi = np.pi * (3. - np.sqrt(5.))
            theta = phi * i + np.sin(i * 0.5) * 0.1 + np.random.uniform(-0.1, 0.1)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)

    # Hybrid initialization combining multiple strategies
    def hybrid_initialization(samples=14):
        points = []
        
        # Strategy 1: Fibonacci with jitter
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(samples // 2):
            y = 1 - (i / float(samples // 2 - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i + np.random.uniform(-0.1, 0.1)
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
            
        # Strategy 2: Random points with sphere projection
        for i in range(samples // 2):
            r = np.random.random()
            theta = np.random.uniform(0, 2*np.pi)
            phi_ang = np.arccos(2*r - 1)
            
            x = np.sin(phi_ang) * np.cos(theta)
            y = np.sin(phi_ang) * np.sin(theta)
            z = np.cos(phi_ang)
            points.append([x, y, z])
            
        return np.array(points)

    # Normalize points to unit cube [0,1]^3
    def normalize_to_cube(points):
        centered = points - np.mean(points, axis=0)
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / max_coord * 0.5
        else:
            scaled = centered
        normalized = scaled + 0.5
        return normalized

    # Objective function with distance variance regularization
    def objective(x_flat):
        points = x_flat.reshape((n, d))
        distances = pdist(points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]

        if len(distances) == 0:
            return -np.inf

        d_min = np.min(distances)
        d_max = np.max(distances)
        d_mean = np.mean(distances)
        d_var = np.var(distances)

        if d_max == 0:
            return -np.inf

        # Regularized objective: maximize ratio while minimizing distance variance
        ratio = d_min / d_max
        variance_penalty = 0.05 * d_var / (d_mean * d_mean + 1e-12)

        # Return negative because we're minimizing
        return -(ratio - variance_penalty)

    # Multi-start optimization with progressive refinement
    best_ratio = -np.inf
    best_points = None

    # Multiple initialization strategies
    init_strategies = [
        sobol_like_distribution,
        fibonacci_sphere,
        hybrid_initialization,
        lambda s: np.random.rand(s, 3),  # Random fallback
    ]

    # Number of optimization runs with adaptive parameters
    num_runs = 30
    base_perturbation = 0.03
    perturbation_decay = 0.92
    min_perturbation = 0.0005

    start_time = time.time()
    
    for run_idx in range(num_runs):
        # Check time limit
        if time.time() - start_time > 350:
            break
            
        # Adaptive perturbation scaling
        current_perturbation = max(base_perturbation * (perturbation_decay ** run_idx),
                                 min_perturbation)

        # Select initialization strategy
        if run_idx < len(init_strategies):
            init_func = init_strategies[run_idx]
        else:
            init_func = lambda s: np.random.rand(s, 3)

        # Get initial points
        initial_points = init_func(n)

        # Normalize to unit cube [0,1]^3
        initial_points = normalize_to_cube(initial_points)

        # Add controlled perturbation to break symmetry
        if run_idx > 0:
            perturbation = np.random.normal(0, current_perturbation, initial_points.shape)
            initial_points += perturbation
            # Clip to stay within bounds
            initial_points = np.clip(initial_points, 0, 1)

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(n * d)]

        # Progressive optimization with fallback strategies
        try:
            # Try L-BFGS-B first (fast convergence)
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            # If L-BFGS-B fails, try SLSQP for better constraint handling
            if not result.success:
                result = minimize(
                    objective,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
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

        except Exception as e:
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
        initial_points = hybrid_initialization(n)
        initial_points = normalize_to_cube(initial_points)
        return initial_points

    # Apply final multi-stage refinement
    refined_points = best_points.copy()

    # Stage 1: High precision L-BFGS-B
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