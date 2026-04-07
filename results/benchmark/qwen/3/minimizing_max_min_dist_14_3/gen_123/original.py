# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
import time
import warnings


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Handle edge cases
        if len(distances) == 0:
            return -1.0

        # Remove any NaN or infinite values
        distances = distances[np.isfinite(distances)]

        if len(distances) == 0:
            return -1.0

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -1.0
        return -d_min / d_max

    def penalty_objective(x, penalty_weight=1e6):
        """Objective with penalty for constraint violations"""
        points = x.reshape(-1, 3)

        # Apply penalty for points outside unit cube
        penalty = 0
        for i in range(len(points)):
            for j in range(3):  # x, y, z coordinates
                if points[i,j] < 0:
                    penalty += penalty_weight * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += penalty_weight * (points[i,j] - 1)**2

        # Original objective
        obj_val = objective(x)

        return obj_val + penalty

    def fibonacci_sphere(samples=14):
        """Generate points on a sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def create_better_initializations():
        """Create multiple better initial configurations"""
        configs = []

        # Config 1: Standard Fibonacci sphere
        fib_points = fibonacci_sphere(14)
        # Scale to unit cube [0,1]^3
        fib_points = fib_points - np.mean(fib_points, axis=0)
        max_coord = np.max(np.abs(fib_points))
        if max_coord > 0:
            fib_points = fib_points / (2 * max_coord) + 0.5
        configs.append(("fibonacci", fib_points))

        # Config 2: Perturbed Fibonacci (better for escaping local optima)
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        # Clamp to [0,1]^3
        perturbed = np.clip(perturbed, 0, 1)
        configs.append(("perturbed_fibonacci", perturbed))

        # Config 3: Random uniform distribution
        random_points = np.random.rand(14, 3)
        configs.append(("random", random_points))

        # Config 4: Grid-based with jitter
        grid_coords = np.linspace(0.05, 0.95, 3)  # Avoid edges
        grid_points = []
        for x in grid_coords:
            for y in grid_coords:
                for z in grid_coords:
                    grid_points.append([x, y, z])
        # Take first 14 points and add jitter
        grid_array = np.array(grid_points[:14]) + np.random.normal(0, 0.02, (14, 3))
        # Clamp to [0,1]^3
        grid_array = np.clip(grid_array, 0, 1)
        configs.append(("grid", grid_array))

        # Config 5: Another random with different seed
        np.random.seed(2468)
        random_points2 = np.random.rand(14, 3)
        configs.append(("random2", random_points2))

        return configs

    def adaptive_de_optimize(objective_func, bounds, initial_points, time_limit):
        """Run differential evolution with adaptive population sizing"""
        start_time = time.time()
        best_solution = None
        best_value = float('inf')

        # Multi-start optimization with different initial configurations
        configs = create_better_initializations()
        config_count = len(configs)

        # Track convergence for adaptive population sizing
        prev_best = float('inf')
        stagnation_counter = 0
        max_stagnation = 5

        for i, (config_name, config) in enumerate(configs):
            if time.time() - start_time > time_limit - 15:  # Leave buffer time
                break

            x0 = config.flatten()

            # Adaptive population sizing based on iteration count
            # Start with larger population for better exploration
            popsize = 15 + i * 2  # Gradually increase popsize

            # Increase popsize if we detect stagnation
            if stagnation_counter > max_stagnation:
                popsize = min(popsize + 10, 50)  # Cap at 50

            maxiter = 80 if i < 2 else 60  # More iterations for initial configs

            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42 + i,
                    maxiter=maxiter,
                    popsize=popsize,
                    mutation=(0.8, 1),  # Higher mutation rate for more exploration
                    recombination=0.9,  # Higher recombination rate
                    atol=1e-7,
                    tol=1e-7,
                    callback=lambda x, convergence: time.time() - start_time > time_limit - 15,
                    disp=False,
                    polish=True
                )

                if result.success:
                    current_value = result.fun
                    if current_value < best_value:
                        best_value = current_value
                        best_solution = result.x
                        prev_best = best_value
                        stagnation_counter = 0  # Reset stagnation counter
                    else:
                        # Check for stagnation
                        if abs(current_value - prev_best) < 1e-8:
                            stagnation_counter += 1
                        else:
                            prev_best = current_value
                            stagnation_counter = 0  # Reset if improvement

            except Exception as e:
                warnings.warn(f"DE optimization failed for config {config_name}: {e}")
                continue

        if best_solution is None:
            # Fallback to initial configuration
            return initial_points.flatten()

        return best_solution

    def improved_local_refinement(objective_func, initial_points, time_limit):
        """Apply multiple local optimization refinement approaches"""
        start_time = time.time()

        # Try Basin-hopping first - very effective for this kind of problem
        try:
            # Basin-hopping with better parameters
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1) for _ in range(42)]}
            result_bh = basinhopping(
                objective_func,
                initial_points,
                niter=30,
                T=1.0,
                stepsize=0.1,
                minimizer_kwargs=minimizer_kwargs,
                seed=42,
                callback=lambda x, f, accepted: time.time() - start_time > time_limit - 10
            )

            if result_bh.success:
                return result_bh.x
        except Exception as e:
            warnings.warn(f"Basin-hopping optimization failed: {e}")

        # Fall back to L-BFGS-B with progressive tolerance tightening
        try:
            # Try with progressively tighter tolerances
            tolerance_levels = [
                {'ftol': 1e-6, 'gtol': 1e-6},  # Looser
                {'ftol': 1e-8, 'gtol': 1e-8},  # Medium
                {'ftol': 1e-9, 'gtol': 1e-9}   # Tighter
            ]

            current_solution = initial_points

            for i, tol_params in enumerate(tolerance_levels):
                if time.time() - start_time > time_limit - 10:
                    break

                result = minimize(
                    objective_func,
                    current_solution,
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(42)],
                    options={'maxiter': 100, **tol_params},
                    callback=lambda x: time.time() - start_time > time_limit - 10
                )

                if result.success:
                    current_solution = result.x
                else:
                    break

            return current_solution
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")

        return initial_points

    def calculate_ratio(points):
        """Calculate min/max distance ratio with robust error handling"""
        if len(points) < 2:
            return 0.0
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            # Filter out invalid distances
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            if d_max <= 0:
                return 0.0
            return d_min / d_max
        except:
            return 0.0

    # Main optimization loop
    start_time = time.time()
    time_limit = 345  # seconds (leave some buffer for final steps)

    # Generate initial configuration
    np.random.seed(42)

    # Start with spherical arrangement
    initial_points = fibonacci_sphere(14)

    # Scale to fit within unit cube [0,1]^3
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / (2 * max_coord) + 0.5

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(42)]

    # Run adaptive differential evolution optimization
    optimized_flat = adaptive_de_optimize(penalty_objective, bounds, initial_points, time_limit)

    # Apply local refinement if time permits
    if time.time() - start_time < time_limit - 10:
        refined_flat = improved_local_refinement(penalty_objective, optimized_flat, time_limit)

        # Evaluate which solution is better
        points_before = optimized_flat.reshape(-1, 3)
        points_after = refined_flat.reshape(-1, 3)

        ratio_before = calculate_ratio(points_before)
        ratio_after = calculate_ratio(points_after)

        if ratio_after > ratio_before:
            optimized_flat = refined_flat

    # Extract optimized points
    optimized_points = optimized_flat.reshape(-1, 3)

    # Ensure all points are within [0,1]^3 (handle any edge cases)
    optimized_points = np.clip(optimized_points, 0, 1)

    # Final validation check
    if calculate_ratio(optimized_points) <= 0:
        # If something went wrong, return original good initialization
        return initial_points

    return optimized_points


# EVOLVE-BLOCK-END