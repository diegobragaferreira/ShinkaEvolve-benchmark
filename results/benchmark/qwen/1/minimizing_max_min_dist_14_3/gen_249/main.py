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

    # Sobol-like initialization for better space-filling properties
    def sobol_like_init(samples=14):
        # Generate points using a low-discrepancy sequence approach
        points = []
        # Use scrambled Halton-like sequence for better distribution
        for i in range(samples):
            # Generate coordinates using irrational numbers for good distribution
            x = (i * 0.618033988749895) % 1.0  # golden ratio
            y = (i * 0.414213562373095) % 1.0  # sqrt(2) - 1
            z = (i * 0.732050807568877) % 1.0  # 2*sqrt(3) - 2

            # Convert to sphere coordinates with some jitter
            r = np.cbrt(x)  # cubic root for better distribution
            theta = y * 2 * np.pi
            phi = np.arccos(2*z - 1)

            px = r * np.sin(phi) * np.cos(theta)
            py = r * np.sin(phi) * np.sin(theta)
            pz = r * np.cos(phi)

            points.append([px, py, pz])

        return np.array(points)

    # Enhanced initialization with multiple strategies
    def enhanced_initialization(samples=14):
        points = []

        # Strategy 1: Fibonacci spiral (primary)
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(samples // 2):
            y = 1 - (i / float(samples // 2 - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i + np.random.uniform(-0.1, 0.1)  # Add jitter
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])

        # Strategy 2: Random points for diversity
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

    # Calculate ratio with proper validation
    def calculate_ratio(points):
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0

        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        if len(distances) == 0:
            return 0.0, 0.0, 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max > 0:
            ratio = d_min / d_max
        else:
            ratio = 0.0

        return ratio, d_min, d_max

    # Objective function for optimization with distance variance regularization
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
        # The variance term helps promote more uniform distribution
        ratio = d_min / d_max
        variance_penalty = 0.05 * d_var / (d_mean * d_mean + 1e-12)

        # Return negative because we're minimizing
        return -(ratio - variance_penalty)

    # Multi-start optimization with progressive refinement
    best_ratio = -np.inf
    best_points = None

    # Multiple initialization strategies
    init_strategies = [
        enhanced_fibonacci_init,
        sobol_like_init,
        fibonacci_sphere,
        enhanced_initialization,
        lambda s: np.random.rand(s, 3),  # Random fallback
    ]

    # Number of optimization runs with adaptive parameters
    num_runs = 30
    base_perturbation = 0.03
    perturbation_decay = 0.92
    min_perturbation = 0.0005

    for run_idx in range(num_runs):
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

        # Progressive optimization with adaptive parameters
        try:
            # Try L-BFGS-B first (fast convergence)
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            # If L-BFGS-B fails, try SLSQP for better constraint handling
            if not result.success:
                result = minimize(
                    objective,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
                )

            # If still failing, try Nelder-Mead as fallback
            if not result.success:
                result = minimize(
                    objective,
                    initial_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12}
                )

        except Exception as e:
            # Fallback to initial points if optimization fails
            continue

        # Extract optimized points
        optimized_points = result.x.reshape((n, d))

        # Ensure all points are within [0,1]^3
        optimized_points = np.clip(optimized_points, 0, 1)

        # Calculate the actual ratio for this optimization run
        ratio, _, _ = calculate_ratio(optimized_points)

        # Update best solution
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # If we didn't find a good solution, return the best initialization
    if best_points is None:
        initial_points = enhanced_fibonacci_init(n)
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
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
        )

        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)

            # Validate improvement
            _, old_min, old_max = calculate_ratio(refined_points)
            _, new_min, new_max = calculate_ratio(candidate_points)

            if new_min > old_min and new_max <= old_max:
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
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
        )

        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)

            # Validate improvement
            _, old_min, old_max = calculate_ratio(refined_points)
            _, new_min, new_max = calculate_ratio(candidate_points)

            if new_min > old_min and new_max <= old_max:
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