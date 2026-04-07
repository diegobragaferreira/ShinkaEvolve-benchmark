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

    def spherical_fibonacci_points(n):
        """Generate n points on sphere using spherical Fibonacci method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def create_enhanced_initialization():
        """Create multiple enhanced initial configurations"""
        configs = []

        # Configuration 1: Enhanced spherical Fibonacci with jitter
        np.random.seed(42)
        fib_points = spherical_fibonacci_points(14)

        # Add small jitter to escape local minima
        jitter = np.random.normal(0, 0.02, fib_points.shape)
        config1 = fib_points + jitter
        # Normalize to unit sphere
        norms = np.linalg.norm(config1, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        config1 = config1 / norms

        # Scale to fit within unit cube [0,1]^3
        config1 = config1 - np.mean(config1, axis=0)
        max_coord = np.max(np.abs(config1))
        if max_coord > 0:
            config1 = config1 / (2 * max_coord) + 0.5
        configs.append(config1)

        # Configuration 2: Random uniform distribution
        np.random.seed(123)
        config2 = np.random.rand(14, 3)
        configs.append(config2)

        # Configuration 3: Another Fibonacci with different scaling
        np.random.seed(456)
        config3 = spherical_fibonacci_points(14)
        # Apply different transformation
        config3 = config3 * 0.8 + 0.1  # Scale and shift
        configs.append(config3)

        # Configuration 4: Grid-based initialization
        np.random.seed(789)
        # Create a 3x3x3 grid and take 14 points
        grid = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j].reshape(3, -1).T
        # Take first 14 points and add jitter
        config4 = grid[:14] + np.random.normal(0, 0.03, (14, 3))
        # Clamp to [0,1]^3
        config4 = np.clip(config4, 0, 1)
        configs.append(config4)

        return configs

    def adaptive_de_optimize(objective_func, bounds, initial_points, time_limit):
        """Run differential evolution with adaptive population sizing and better parameters"""
        start_time = time.time()
        best_solution = None
        best_value = float('inf')

        # Multi-start optimization with different initial configurations
        configs = create_enhanced_initialization()
        config_count = len(configs)

        for i, config in enumerate(configs):
            if time.time() - start_time > time_limit - 15:  # Leave buffer time
                break

            x0 = config.flatten()

            # Adaptive population sizing based on iteration count
            # Start with larger population for better exploration
            popsize = 12 + i * 3  # Gradually increase popsize
            maxiter = 60 if i < 2 else 40  # More iterations for initial configs

            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42 + i,
                    maxiter=maxiter,
                    popsize=popsize,
                    mutation=(0.7, 1),  # Higher mutation rate for more exploration
                    recombination=0.8,  # Higher recombination rate
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

            except Exception as e:
                warnings.warn(f"DE optimization failed for config {i}: {e}")
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
                niter=20,
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

        # Fall back to L-BFGS-B with adaptive tolerance tightening
        try:
            # Track improvement to adaptively tighten tolerances
            last_obj_value = float('inf')
            improvement_count = 0
            max_improvement_streak = 5

            # Start with looser tolerances for faster initial convergence
            ftol = 1e-6
            gtol = 1e-6

            # Try with initial loose tolerances
            result = minimize(
                objective_func,
                initial_points,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'maxiter': 150, 'ftol': ftol, 'gtol': gtol},
                callback=lambda x: time.time() - start_time > time_limit - 10
            )

            # If we have room and significant improvement, try tighter tolerances
            if time.time() - start_time < time_limit - 10 and result.success:
                # Check if we're still making progress
                current_obj = result.fun
                if abs(current_obj - last_obj_value) > 1e-8:  # If there's significant improvement
                    improvement_count += 1
                    if improvement_count >= 2:  # After seeing consistent improvement
                        # Tighten tolerances for more precise final result
                        tightened_result = minimize(
                            objective_func,
                            result.x,
                            method='L-BFGS-B',
                            bounds=[(0, 1) for _ in range(42)],
                            options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9},
                            callback=lambda x: time.time() - start_time > time_limit - 10
                        )
                        if tightened_result.success:
                            return tightened_result.x
                else:
                    improvement_count = 0  # Reset if no improvement

            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"L-BFGS optimization failed: {e}")

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

    # Generate initial configuration - use enhanced initialization
    np.random.seed(42)

    # Start with spherical arrangement
    initial_points = spherical_fibonacci_points(14)

    # Scale to fit within unit cube [0,1]^3
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / (2 * max_coord) + 0.5

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(42)]

    # Run adaptive differential evolution optimization
    optimized_flat = adaptive_de_optimize(objective, bounds, initial_points, time_limit)

    # Apply local refinement if time permits
    if time.time() - start_time < time_limit - 10:
        refined_flat = improved_local_refinement(objective, optimized_flat, time_limit)

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