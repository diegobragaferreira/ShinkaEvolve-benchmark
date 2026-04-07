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

    def enforce_spherical_constraint(points):
        """Ensure points lie on the unit sphere by normalizing them"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Handle zero norm cases to avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def objective_with_spherical_constraint(x):
        """Objective function that respects spherical constraint"""
        # Reshape x back to points
        points = x.reshape(n, d)

        # Enforce spherical constraint
        points = enforce_spherical_constraint(points)

        # Calculate distance matrix
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative ratio (to minimize)
        return -min_dist / max_dist

    # Phase 1: Generate improved initial configuration using enhanced spherical approach
    np.random.seed(42)

    # Create a more uniform distribution on the sphere using iterative optimization approach
    # Start with Fibonacci-like distribution
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    # Generate points using improved Fibonacci-like method with better spacing
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y*y)  # radius at y

        # More careful angular distribution
        theta = i * 2 * np.pi / golden_ratio

        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-10)

    # Apply a small amount of jitter to break symmetries
    # Use a more controlled approach to avoid extreme distortions
    jitter_magnitude = 0.02
    points += np.random.normal(0, jitter_magnitude, points.shape) * 0.5

    # Normalize again after jitter
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-10)

    # Project to [0,1]^3 cube while preserving good distribution
    points = (points + 1) / 2

    # Apply more substantial perturbation to ensure good exploration
    # but keep within reasonable bounds
    perturbation_magnitude = 0.01
    points += np.random.normal(0, perturbation_magnitude, points.shape) * 0.7

    # Ensure points stay within bounds
    points = np.clip(points, 0, 1)

    # Phase 2: Global optimization with differential evolution
    bounds = [(0, 1) for _ in range(n * d)]

    # Run differential evolution with improved spherical constraint handling
    max_time = 280  # Leave some time for final refinement
    start_time = time.time()

    # Adaptive population size control for better exploration
    initial_popsize = 25
    max_popsize = 60
    popsize = initial_popsize
    last_improvement_gen = 0
    stagnation_count = 0
    max_stagnation = 15

    def adaptive_callback(x, convergence):
        nonlocal popsize, last_improvement_gen, stagnation_count

        current_time = time.time()
        if current_time - start_time > max_time:
            return True

        # Track stagnation and improve population size if needed
        # Use a more sensitive check for improvement
        if convergence < 1e-8:  # Tighter convergence threshold
            stagnation_count += 1
            if stagnation_count > max_stagnation and popsize < max_popsize:
                # Increase population size more aggressively
                popsize = min(popsize + 10, max_popsize)
                stagnation_count = 0
                print(f"Population size increased to {popsize} due to stagnation")
        else:
            stagnation_count = 0
            last_improvement_gen = convergence

        return False

    # Use a more robust differential evolution setup with adaptive population
    result = differential_evolution(
        objective_with_spherical_constraint,
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

    # Phase 3: Improved local refinement with spherical constraint handling
    def local_refinement(points):
        """Refine using improved local optimization approaches with spherical constraint"""

        def objective_spherical(x):
            """Objective function that respects spherical constraint"""
            points_temp = x.reshape(n, d)
            # Enforce spherical constraint
            points_temp = enforce_spherical_constraint(points_temp)
            distances = cdist(points_temp, points_temp, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for maximization

        # Strategy 1: L-BFGS-B with bounds and spherical constraint
        def lbfgsb_refinement():
            try:
                # Create bounds that keep points within unit cube, but we'll enforce spherical constraint
                bounds = [(0, 1) for _ in range(n * d)]
                res = minimize(objective_spherical, points.flatten(), method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
                if res.success:
                    refined_points = res.x.reshape(n, d)
                    # Ensure spherical constraint is maintained
                    refined_points = enforce_spherical_constraint(refined_points)
                    return refined_points
            except Exception as e:
                print(f"L-BFGS-B refinement failed: {e}")
            return points

        # Strategy 2: Nelder-Mead with spherical constraint
        def nelder_mead_refinement():
            try:
                res = minimize(objective_spherical, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 500, 'disp': False})
                if res.success:
                    refined_points = res.x.reshape(n, d)
                    # Ensure spherical constraint is maintained
                    refined_points = enforce_spherical_constraint(refined_points)
                    return refined_points
            except Exception as e:
                print(f"Nelder-Mead refinement failed: {e}")
            return points

        # Strategy 3: Simple gradient ascent approach with proper constraint handling
        def gradient_approach():
            # Simple gradient descent on the negative ratio (maximization)
            current_points = points.copy()
            learning_rate = 0.01
            max_steps = 500

            for step in range(max_steps):
                # Compute gradients numerically
                try:
                    # Compute distances
                    distances = cdist(current_points, current_points, 'euclidean')
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist == 0:
                        break

                    # Simple gradient approximation using finite differences
                    epsilon = 1e-6
                    grad_sum = np.zeros_like(current_points)

                    for i in range(n):
                        for j in range(d):
                            # Perturb point
                            perturbed = current_points.copy()
                            perturbed[i, j] += epsilon

                            # Enforce spherical constraint after perturbation
                            perturbed = enforce_spherical_constraint(perturbed)

                            distances_pert = cdist(perturbed, perturbed, 'euclidean')
                            np.fill_diagonal(distances_pert, np.inf)
                            min_dist_pert = np.min(distances_pert)
                            max_dist_pert = np.max(distances_pert)

                            if max_dist_pert > 0:
                                ratio_pert = min_dist_pert / max_dist_pert
                                grad_sum[i, j] = (ratio_pert - (-min_dist / max_dist)) / epsilon

                    # Update points (gradient ascent for maximization)
                    current_points += learning_rate * grad_sum

                    # Enforce spherical constraint
                    current_points = enforce_spherical_constraint(current_points)

                    # Ensure within bounds
                    current_points = np.clip(current_points, 0, 1)

                except Exception as e:
                    print(f"Gradient approach failed: {e}")
                    break

            return current_points

        # Apply refinements in order of preference
        refined_lbfgsb = lbfgsb_refinement()
        ratio_lbfgsb = compute_min_max_ratio(refined_lbfgsb.flatten())

        refined_nm = nelder_mead_refinement()
        ratio_nm = compute_min_max_ratio(refined_nm.flatten())

        refined_grad = gradient_approach()
        ratio_grad = compute_min_max_ratio(refined_grad.flatten())

        # Return the best result among all methods
        best_ratios = [ratio_lbfgsb, ratio_nm, ratio_grad]
        best_results = [refined_lbfgsb, refined_nm, refined_grad]

        best_idx = np.argmin(best_ratios)
        return best_results[best_idx]

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