# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hybrid approach combining spherical initialization with advanced optimization techniques.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    n = 14
    d = 3

    def generate_fibonacci_sphere_points():
        """Generate initial points on unit sphere using Fibonacci method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def project_to_cube(points):
        """Project points from unit sphere to unit cube"""
        # Normalize to unit sphere first
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points_normalized = points / np.maximum(norms, 1e-10)

        # Project to cube: map to [0,1]^3
        points_cube = (points_normalized + 1) / 2
        return np.clip(points_cube, 0, 1)

    def compute_min_max_ratio(points):
        """Compute negative of min/max distance ratio for optimization"""
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return -np.inf

        return -min_dist / max_dist

    def objective_with_penalty(x):
        """Objective function with penalty for boundary violations"""
        points = x.reshape(n, d)

        # Apply soft penalty for boundary violations
        penalty = 0
        for i in range(n):
            for j in range(d):
                if points[i,j] < 0:
                    penalty += 1e6 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1e6 * (points[i,j] - 1)**2

        # Calculate distance matrix
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf + penalty

        # Return negative ratio plus penalty (to minimize)
        return -(min_dist / max_dist) + penalty

    def local_refinement(points):
        """Refine using multiple local optimization approaches"""
        def obj(x):
            points_temp = x.reshape(n, d)
            distances = cdist(points_temp, points_temp, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for maximization

        # Strategy 1: L-BFGS-B with bounds
        bounds = [(0, 1) for _ in range(n * d)]
        try:
            res = minimize(obj, points.flatten(), method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
            if res.success:
                return res.x.reshape(n, d)
        except:
            pass

        # Strategy 2: Nelder-Mead as fallback
        try:
            res = minimize(obj, points.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'disp': False})
            if res.success:
                return res.x.reshape(n, d)
        except:
            pass

        return points

    # Phase 1: Generate initial configuration using spherical approach
    np.random.seed(42)

    # Generate points on sphere
    sphere_points = generate_fibonacci_sphere_points()

    # Project to cube
    cube_points = project_to_cube(sphere_points)

    # Add structured perturbation to avoid symmetric solutions
    cube_points += np.random.normal(0, 0.01, cube_points.shape)
    cube_points = np.clip(cube_points, 0, 1)

    # Phase 2: Global optimization with differential evolution
    bounds = [(0, 1) for _ in range(n * d)]
    max_time = 280  # Leave some time for final refinement
    start_time = time.time()

    # Use differential evolution with proper timeout management
    result = differential_evolution(
        objective_with_penalty,
        bounds,
        seed=42,
        maxiter=500,
        popsize=20,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        tol=1e-6,
        callback=lambda x, convergence: time.time() - start_time > max_time
    )

    # Extract best solution
    best_points = result.x.reshape(n, d)

    # Phase 3: Local refinement with multiple strategies
    refined_points = local_refinement(best_points)

    # Phase 4: Multiple restarts for better exploration
    best_final_points = refined_points.copy()
    best_final_ratio = compute_min_max_ratio(refined_points)

    # Try several random restarts
    for restart in range(3):
        np.random.seed(restart * 1000 + 42)

        # Create slightly perturbed starting point
        perturbed = refined_points + np.random.normal(0, 0.005, refined_points.shape)
        perturbed = np.clip(perturbed, 0, 1)

        # Refine the perturbed point
        restarted_points = local_refinement(perturbed)
        restarted_ratio = compute_min_max_ratio(restarted_points)

        if restarted_ratio < best_final_ratio:  # Better ratio
            best_final_ratio = restarted_ratio
            best_final_points = restarted_points.copy()

    # Final verification
    final_points = best_final_points

    # Ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)

    # Verify we have correct shape
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"

    return final_points


# EVOLVE-BLOCK-END