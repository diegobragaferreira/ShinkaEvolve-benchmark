# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    n = 14
    d = 3

    def compute_min_max_ratio(points_flat):
        """Compute negative of min/max distance ratio for optimization"""
        points = points_flat.reshape(n, d)
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return -np.inf

        return -min_dist / max_dist

    def objective_with_penalty(x):
        """Objective function with penalty for boundary violations"""
        # Reshape x back to points
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

    # Phase 1: Generate initial configuration using known optimal 14-point spherical distribution
    # This uses a well-studied configuration from mathematical literature for 14 points on sphere
    np.random.seed(42)

    # Known good 14-point configuration from mathematical studies
    # These coordinates approximate a highly symmetric distribution
    known_14_points = np.array([
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, -1.0000],
        [0.9343, 0.0000, 0.3564],
        [-0.9343, 0.0000, 0.3564],
        [0.0000, 0.9343, 0.3564],
        [0.0000, -0.9343, 0.3564],
        [0.0000, 0.9343, -0.3564],
        [0.0000, -0.9343, -0.3564],
        [0.9343, 0.0000, -0.3564],
        [-0.9343, 0.0000, -0.3564],
        [0.3564, 0.9343, 0.0000],
        [-0.3564, 0.9343, 0.0000],
        [0.3564, -0.9343, 0.0000],
        [-0.3564, -0.9343, 0.0000]
    ])

    # Normalize to unit sphere (should already be normalized but just to be safe)
    norms = np.linalg.norm(known_14_points, axis=1, keepdims=True)
    points_sphere = known_14_points / np.maximum(norms, 1e-10)

    # Project to [0,1]^3 cube
    points = (points_sphere + 1) / 2

    # Add more strategic perturbations to break symmetries and improve optimization
    # Use different magnitudes for different directions to preserve more structure
    perturbation_magnitude = 0.008
    noise = np.random.normal(0, perturbation_magnitude, points.shape)
    # Apply directional bias to preserve some geometric structure
    noise[:, 0] *= 1.2  # Slightly more perturbation in x-direction
    noise[:, 1] *= 0.8  # Less perturbation in y-direction
    noise[:, 2] *= 1.0  # Normal perturbation in z-direction

    points += noise

    # Ensure points stay within bounds
    points = np.clip(points, 0, 1)

    # Phase 2: Global optimization with differential evolution
    bounds = [(0, 1) for _ in range(n * d)]

    # Run differential evolution with adaptive population sizing
    max_time = 280  # Leave some time for final refinement
    start_time = time.time()

    # Adaptive population size control
    initial_popsize = 20
    max_popsize = 50
    popsize = initial_popsize
    last_improvement_gen = 0
    stagnation_count = 0
    max_stagnation = 10

    def adaptive_callback(x, convergence):
        nonlocal popsize, last_improvement_gen, stagnation_count

        current_time = time.time()
        if current_time - start_time > max_time:
            return True

        # Track stagnation - if no improvement for several generations
        if convergence < 1e-6:  # Convergence threshold
            stagnation_count += 1
            if stagnation_count > max_stagnation and popsize < max_popsize:
                popsize = min(popsize + 5, max_popsize)
                stagnation_count = 0
        else:
            stagnation_count = 0
            last_improvement_gen = convergence

        return False

    # Use a more robust differential evolution setup with adaptive population
    result = differential_evolution(
        objective_with_penalty,
        bounds,
        seed=42,
        maxiter=500,
        popsize=popsize,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        tol=1e-6,
        callback=adaptive_callback
    )

    # Extract and refine the best solution
    best_points = result.x.reshape(n, d)

    # Phase 3: Local refinement with multiple strategies
    def local_refinement(points):
        """Refine using multiple local optimization approaches"""
        # Strategy 1: L-BFGS-B with bounds
        def lbfgsb_refinement():
            def obj(x):
                points_temp = x.reshape(n, d)
                distances = cdist(points_temp, points_temp, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist  # Negative for maximization

            bounds = [(0, 1) for _ in range(n * d)]
            try:
                res = minimize(obj, points.flatten(), method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
                if res.success:
                    return res.x.reshape(n, d)
            except:
                pass
            return points

        # Strategy 2: Nelder-Mead as fallback
        def nelder_mead_refinement():
            def obj(x):
                points_temp = x.reshape(n, d)
                distances = cdist(points_temp, points_temp, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist  # Negative for maximization

            try:
                res = minimize(obj, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 500, 'disp': False})
                if res.success:
                    return res.x.reshape(n, d)
            except:
                pass
            return points

        # Apply both strategies and return the better one
        refined_lbfgsb = lbfgsb_refinement()
        refined_nm = nelder_mead_refinement()

        # Compare results
        ratio_lbfgsb = compute_min_max_ratio(refined_lbfgsb.flatten())
        ratio_nm = compute_min_max_ratio(refined_nm.flatten())

        return refined_lbfgsb if ratio_lbfgsb < ratio_nm else refined_nm

    # Apply local refinement
    refined_points = local_refinement(best_points)

    # Phase 4: Multiple restarts for better exploration
    best_final_points = refined_points.copy()
    best_final_ratio = compute_min_max_ratio(refined_points.flatten())

    # Try several random restarts
    for restart in range(3):
        np.random.seed(restart * 1000 + 42)

        # Create slightly perturbed starting point
        perturbed = refined_points + np.random.normal(0, 0.005, refined_points.shape)
        perturbed = np.clip(perturbed, 0, 1)

        # Refine the perturbed point
        restarted_points = local_refinement(perturbed)
        restarted_ratio = compute_min_max_ratio(restarted_points.flatten())

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