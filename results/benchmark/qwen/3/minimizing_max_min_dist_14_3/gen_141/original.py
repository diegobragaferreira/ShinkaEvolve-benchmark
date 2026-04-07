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

    def compute_diversity_score(points):
        """Compute a diversity score based on inverse of minimum distance"""
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        return min_dist if min_dist > 0 else 1e-10

    def objective_with_soft_constraints(x):
        """Objective function with soft constraints for boundary violations"""
        points = x.reshape(n, d)
        
        # Soft constraint penalties for boundary violations
        penalty = 0
        for i in range(n):
            for j in range(d):
                if points[i,j] < 0:
                    penalty += 1e4 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1e4 * (points[i,j] - 1)**2

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

    def generate_cube_grid_points():
        """Generate points in a structured cube grid pattern"""
        # Create a roughly uniform grid in cube
        grid_size = int(np.ceil(n**(1/3)))
        grid = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size].reshape(3, -1).T
        grid = grid[:n]  # Take only required number of points
        grid = grid.astype(float) / (grid_size - 1) if grid_size > 1 else np.zeros((n, 3))
        return np.clip(grid, 0, 1)

    def generate_random_points():
        """Generate random points in unit cube"""
        np.random.seed(42)
        return np.random.rand(n, d)

    def adaptive_differential_evolution(objective_func, bounds, max_time, initial_popsize=20):
        """Adaptive differential evolution with dynamic population sizing"""
        start_time = time.time()
        popsize = initial_popsize
        last_improvement = float('-inf')
        improvement_stall_count = 0
        last_best_value = float('inf')

        def callback_with_adaptation(x, convergence):
            nonlocal popsize, last_improvement, improvement_stall_count, last_best_value

            current_time = time.time()
            if current_time - start_time > max_time:
                return True  # Stop optimization

            # Check for improvement
            current_value = objective_func(x)
            if current_value < last_best_value - 1e-8:  # Significant improvement
                last_best_value = current_value
                last_improvement = current_time
                improvement_stall_count = 0
            else:
                improvement_stall_count += 1

            # Adapt population size if stagnation detected
            if improvement_stall_count > 10 and popsize < 35:  # Only increase if not too large
                popsize = min(popsize + 5, 35)
                improvement_stall_count = 0

            return False  # Continue optimization

        # Run differential evolution with adaptive parameters
        result = differential_evolution(
            objective_func,
            bounds,
            seed=42,
            maxiter=1000,
            popsize=popsize,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            tol=1e-6,
            callback=callback_with_adaptation
        )

        return result

    def adaptive_local_refinement(points, iteration=0):
        """Refine using multiple local optimization approaches with adaptive tolerances"""
        def obj(x):
            points_temp = x.reshape(n, d)
            distances = cdist(points_temp, points_temp, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for maximization

        # Adaptive tolerances based on iteration
        ftol = 1e-9 if iteration > 3 else 1e-6
        gtol = 1e-9 if iteration > 3 else 1e-6

        # Strategy 1: L-BFGS-B with bounds
        bounds = [(0, 1) for _ in range(n * d)]
        try:
            res = minimize(obj, points.flatten(), method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': ftol, 'gtol': gtol})
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

    # Phase 1: Multi-initialization with diversity selection
    initial_strategies = [
        generate_fibonacci_sphere_points,
        generate_cube_grid_points,
        generate_random_points
    ]

    # Evaluate multiple initializations
    initial_scores = []
    initial_configurations = []

    for i, init_func in enumerate(initial_strategies):
        np.random.seed(42 + i)
        try:
            initial_points = init_func()
            diversity_score = compute_diversity_score(initial_points)
            initial_scores.append(diversity_score)
            initial_configurations.append(initial_points)
        except Exception as e:
            print(f"Initialization {i} failed: {e}")
            initial_scores.append(0)
            initial_configurations.append(np.random.rand(n, d))

    # Select best initialization based on diversity score
    best_init_idx = np.argmax(initial_scores)
    selected_initial = initial_configurations[best_init_idx]

    # Add structured perturbation to avoid symmetric solutions
    selected_initial += np.random.normal(0, 0.01, selected_initial.shape)
    selected_initial = np.clip(selected_initial, 0, 1)

    # Phase 2: Adaptive global optimization with differential evolution
    bounds = [(0, 1) for _ in range(n * d)]
    max_time = 280  # Leave some time for final refinement

    # Use adaptive differential evolution
    result = adaptive_differential_evolution(
        objective_with_soft_constraints,
        bounds,
        max_time,
        initial_popsize=20
    )

    # Extract best solution
    best_points = result.x.reshape(n, d)

    # Phase 3: Adaptive local refinement with early stopping
    current_points = best_points.copy()
    previous_ratio = compute_min_max_ratio(current_points.flatten())
    improvement_threshold = 1e-8
    patience_counter = 0
    max_patience = 5

    # Apply multiple rounds of adaptive refinement
    for iteration in range(8):
        refined_points = adaptive_local_refinement(current_points, iteration)
        new_ratio = compute_min_max_ratio(refined_points.flatten())

        # Check for improvement
        if new_ratio < previous_ratio - improvement_threshold:
            current_points = refined_points
            previous_ratio = new_ratio
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if no significant improvement
        if patience_counter >= max_patience:
            break

    # Phase 4: Multiple restarts with varying perturbation scales
    best_final_points = current_points.copy()
    best_final_ratio = compute_min_max_ratio(current_points.flatten())

    # Try several random restarts with different perturbation scales
    perturbation_scales = [0.005, 0.01, 0.02]
    
    for restart in range(5):
        np.random.seed(restart * 1000 + 42)
        
        # Select perturbation scale randomly from predefined scales
        scale = perturbation_scales[restart % len(perturbation_scales)]
        
        # Create slightly perturbed starting point
        perturbed = current_points + np.random.normal(0, scale, current_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Apply adaptive refinement to perturbed point
        restarted_points = adaptive_local_refinement(perturbed, restart)
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