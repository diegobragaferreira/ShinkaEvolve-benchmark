# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def project_to_sphere(points):
        """Project points onto unit sphere while maintaining relative positions."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis] * 0.99

    def adaptive_perturb(points, point_idx, distances):
        """Apply adaptive perturbation based on local point relationships."""
        # Get distances to all other points
        dist_to_others = distances[point_idx]
        dist_to_others[point_idx] = np.inf  # Exclude self-distance

        # Find nearest and furthest points
        nearest_idx = np.argmin(dist_to_others)
        furthest_idx = np.argmax(dist_to_others)

        # Calculate adaptive perturbation scale based on local density
        mean_dist = np.mean(dist_to_others)
        perturbation_scale = 0.005 * (1.0 + 0.2 * mean_dist)  # Scale with local density

        # Create perturbation vector
        delta = np.random.normal(0, perturbation_scale, 3)

        # Adjust perturbation direction to improve min/max ratio
        if dist_to_others[nearest_idx] < dist_to_others[furthest_idx]:
            # Move closer to nearest point to increase min distance
            direction = points[nearest_idx] - points[point_idx]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                delta += 0.5 * direction * perturbation_scale

        # Move away from furthest point to decrease max distance
        direction = points[point_idx] - points[furthest_idx]
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            delta -= 0.3 * direction * perturbation_scale

        return delta

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_offset(n, offsets):
        """Generate n points distributed approximately uniformly on a sphere with phase offsets."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            # Apply offset to phase
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi + offsets[i]),
                              np.cos(i * 2 * np.pi / phi + offsets[i]))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_golden(n):
        """Generate n points using golden ratio with slight variation."""
        points = []
        # Use slightly adjusted golden ratio for variety
        phi = (1 + np.sqrt(5)) / 2 * 0.95  # Slightly modified
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_alternative(n):
        """Generate n points using alternative Fibonacci approach."""
        points = []
        # Use a different parameterization that sometimes works better
        phi = (3 + np.sqrt(5)) / 2  # Alternative golden ratio related constant
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_custom(n):
        """Generate n points with custom parameterization."""
        points = []
        # More experimental parameterization
        phi = np.pi * (3 - np.sqrt(5))  # Plastic number related
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = i * phi
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_physics(n):
        """Generate n points using physics-inspired distribution."""
        points = []
        # Using a variant that mimics particle distribution
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = i * phi + np.random.uniform(0, 0.5)  # Add some randomness
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def fibonacci_sphere_lucas(n):
        """Generate n points using Lucas sequence ratio."""
        points = []
        # Lucas numbers: 2, 1, 3, 4, 7, 11, 18, ...
        # Using ratio similar to Fibonacci but with Lucas base
        lucas_ratio = 2.414213562373095  # (3 + sqrt(5))/2
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = i * lucas_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def spherical_voronoi_init(n, num_attempts=5):
        """Generate diverse initial points using SphericalVoronoi for better distribution."""
        best_points = None
        best_ratio = 0.0

        for attempt in range(num_attempts):
            # Generate random points on sphere
            np.random.seed(attempt + 1000)
            points = np.random.randn(n, 3)
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]

            try:
                # Create SphericalVoronoi diagram
                sv = SphericalVoronoi(points, radius=1.0)

                # Get the centroids of the Voronoi cells as new candidate points
                voronoi_points = sv.vertices

                # If we got enough points, use them; otherwise fall back to original
                if len(voronoi_points) >= n:
                    # Take first n points, but make sure they're properly normalized
                    selected_points = voronoi_points[:n]
                    selected_points = selected_points / np.linalg.norm(selected_points, axis=1)[:, np.newaxis]
                    # Add small noise to break degeneracies
                    noise = np.random.normal(0, 0.001, selected_points.shape)
                    selected_points += noise
                    selected_points = selected_points / np.linalg.norm(selected_points, axis=1)[:, np.newaxis]

                    # Evaluate this configuration
                    ratio = compute_min_max_ratio(selected_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = selected_points.copy()
                else:
                    # If Voronoi didn't give us enough points, use a simple approach
                    # Add noise to original points and normalize
                    noise = np.random.normal(0, 0.005, points.shape)
                    noisy_points = points + noise
                    noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1)[:, np.newaxis]

                    ratio = compute_min_max_ratio(noisy_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = noisy_points.copy()

            except Exception:
                # If SphericalVoronoi fails, fall back to regular noise
                noise = np.random.normal(0, 0.005, points.shape)
                noisy_points = points + noise
                noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1)[:, np.newaxis]

                ratio = compute_min_max_ratio(noisy_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = noisy_points.copy()

        return best_points if best_points is not None else points

    def two_phase_optimization(initial_points):
        """Perform two-phase optimization: global search then local refinement."""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)

        # Phase 1: Global search with modified optimization
        current_points = initial_points.copy()

        # Run multiple iterations of optimization with different strategies
        for iter_num in range(3):
            # Flatten points for optimization
            x0 = current_points.flatten()

            def constraint_func(x):
                # Ensure points stay within unit sphere (for better conditioning)
                points = x.reshape(-1, 3)
                norms = np.linalg.norm(points, axis=1)
                # Return positive values where constraint is satisfied
                return 1.0 - norms  # Positive when norm <= 1

            try:
                # Try with different optimization settings
                result = minimize(lambda x: -compute_min_max_ratio(x.reshape(-1, 3)),
                                x0, method='SLSQP',
                                constraints={'type': 'ineq', 'fun': constraint_func},
                                options={'maxiter': 300, 'ftol': 1e-6},
                                bounds=[(-1, 1)] * 42)

                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    # Normalize to unit sphere
                    norms_final = np.linalg.norm(final_points, axis=1)
                    if np.max(norms_final) > 1:
                        final_points = final_points / np.max(norms_final) * 0.99

                    current_ratio = compute_min_max_ratio(final_points)
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = final_points.copy()

            except:
                pass

            # Apply small random perturbations for diversity
            if iter_num < 2:  # Don't perturb on the last iteration
                current_points = best_points.copy()
                # Perturb a few points
                for _ in range(2):
                    idx = np.random.randint(0, len(current_points))
                    delta = np.random.normal(0, 0.005, 3)
                    current_points[idx] += delta
                    # Project back to sphere
                    norms = np.linalg.norm(current_points[idx])
                    if norms > 0:
                        current_points[idx] = current_points[idx] / norms * 0.99

        # Phase 2: Local refinement with detailed optimization
        try:
            x0 = best_points.flatten()

            def constraint_func(x):
                # Ensure points stay within unit sphere (for better conditioning)
                points = x.reshape(-1, 3)
                norms = np.linalg.norm(points, axis=1)
                # Return positive values where constraint is satisfied
                return 1.0 - norms  # Positive when norm <= 1

            result = minimize(lambda x: -compute_min_max_ratio(x.reshape(-1, 3)),
                            x0, method='SLSQP',
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 500, 'ftol': 1e-8},
                            bounds=[(-1, 1)] * 42)

            if result.success:
                refined_points = result.x.reshape(-1, 3)
                # Normalize to unit sphere if needed
                norms_refined = np.linalg.norm(refined_points, axis=1)
                if np.max(norms_refined) > 1:
                    refined_points = refined_points / np.max(norms_refined) * 0.99

                refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_points = refined_points.copy()
                    best_ratio = refined_ratio

        except:
            pass

        return best_points, best_ratio

    def simulated_annealing_multiple_starts():
        """Run simulated annealing multiple times with different random seeds."""
        best_points = None
        best_ratio = 0.0
        best_seed = 0

        # Extended seeds for better exploration
        seeds = [42, 123, 456, 789, 999, 111, 222, 333, 555, 666, 777, 888, 999, 101, 202, 303]

        # Try multiple restarts with different seeds
        for seed in seeds:
            np.random.seed(seed)

            # Initialize with multiple Fibonacci sphere arrangements
            # Test different starting configurations for better diversity
            starting_configurations = [
                ("base", lambda: fibonacci_sphere(14)),
                ("noisy", lambda: fibonacci_sphere(14) + np.random.normal(0, 0.01, (14, 3))),
                ("lucas", lambda: fibonacci_sphere_lucas(14)),
                ("physics", lambda: fibonacci_sphere_physics(14)),
                ("golden", lambda: fibonacci_sphere_golden(14)),
                ("alternative", lambda: fibonacci_sphere_alternative(14))
            ]

            for config_name, init_func in starting_configurations:
                points = init_func()

                # Add controlled noise
                noise = np.random.normal(0, 0.01, points.shape)
                points += noise

                # Normalize to unit sphere
                norms = np.linalg.norm(points, axis=1)
                if np.max(norms) > 0:
                    points = points / np.max(norms) * 0.9

                # Project to unit sphere
                points = project_to_sphere(points)

                # Simulated Annealing parameters
                current_temp = 1.0
                min_temp = 1e-8
                base_cooling_rate = 0.9995
                max_iter = 70000  # Increased iterations
                accept_threshold = 0.01

                # Track recent improvements for adaptive cooling
                recent_improvements = []
                improvement_window = 1000
                max_improvement_window = 5000

                # Initialize best solution
                sa_best_points = points.copy()
                sa_best_ratio = compute_min_max_ratio(sa_best_points)

                start_time = time.time()
                last_improvement = 0

                # Checkpointing variables
                best_checkpoint_points = sa_best_points.copy()
                best_checkpoint_ratio = sa_best_ratio
                last_checkpoint = 0

                for iteration in range(max_iter):
                    # Adaptive cooling schedule with dynamic adjustment
                    cooling_rate = base_cooling_rate

                    # Monitor recent improvements to adjust cooling rate
                    if len(recent_improvements) >= improvement_window:
                        # Calculate improvement rate in recent iterations
                        recent_improvement_count = sum(recent_improvements[-improvement_window:])
                        improvement_rate = recent_improvement_count / improvement_window

                        # If improvement rate is low, increase cooling rate to escape local optima
                        if improvement_rate < 0.05:  # Less than 5% accepted improvements
                            cooling_rate = min(0.99995, cooling_rate * 1.05)  # Aggressive cooling
                        elif improvement_rate < 0.15:  # Between 5% and 15% improvement rate
                            cooling_rate = min(0.9998, cooling_rate * 1.02)  # Medium cooling
                        else:  # High improvement rate
                            cooling_rate = max(0.9994, cooling_rate * 0.98)  # Slower cooling

                    # More aggressive early cooling, slower later
                    if iteration < 10000:
                        cooling_rate = min(cooling_rate, 0.9998)
                    elif iteration < 20000:
                        cooling_rate = min(cooling_rate, 0.9997)
                    else:
                        cooling_rate = min(cooling_rate, 0.9996)

                    # Cool down temperature
                    current_temp *= cooling_rate

                    if current_temp < min_temp:
                        break

                    # Try to make a small random move
                    test_points = sa_best_points.copy()

                    # Select random point to perturb
                    point_idx = np.random.randint(0, 14)

                    # Make small random perturbation - adapt to current temp
                    delta = np.random.normal(0, 0.001 * current_temp, 3)
                    test_points[point_idx] += delta

                    # Project back to sphere
                    test_points = project_to_sphere(test_points)

                    # Compute new ratio
                    test_ratio = compute_min_max_ratio(test_points)

                    # Accept or reject based on Metropolis criterion
                    if test_ratio > sa_best_ratio:
                        sa_best_points = test_points.copy()
                        sa_best_ratio = test_ratio
                        last_improvement = iteration
                        # Record improvement
                        recent_improvements.append(1)
                    elif np.random.random() < np.exp((test_ratio - sa_best_ratio) / current_temp):
                        sa_best_points = test_points.copy()
                        sa_best_ratio = test_ratio
                        last_improvement = iteration
                        # Record improvement
                        recent_improvements.append(1)
                    else:
                        # Record no improvement
                        recent_improvements.append(0)

                    # Keep recent improvements window fixed
                    if len(recent_improvements) > max_improvement_window:
                        recent_improvements.pop(0)

                    # Checkpoint every 3000 iterations
                    if iteration > 0 and iteration % 3000 == 0:
                        if sa_best_ratio > best_checkpoint_ratio:
                            best_checkpoint_points = sa_best_points.copy()
                            best_checkpoint_ratio = sa_best_ratio
                            last_checkpoint = iteration

                    # Local refinement every 500 iterations (more frequent)
                    if iteration > 0 and iteration % 500 == 0:
                        # Apply local optimization to current best solution
                        def objective(x_flat):
                            points_test = x_flat.reshape(-1, 3)
                            distances = pdist(points_test)
                            min_dist = np.min(distances)
                            max_dist = np.max(distances)
                            if max_dist == 0:
                                return float('inf')
                            return -min_dist / max_dist

                        def constraint_func(x_flat):
                            points_test = x_flat.reshape(-1, 3)
                            norms = np.linalg.norm(points_test, axis=1)
                            return 1.0 - norms

                        x0 = sa_best_points.flatten()
                        cons = {'type': 'ineq', 'fun': constraint_func}

                        try:
                            result = minimize(objective, x0, method='SLSQP',
                                             constraints=cons,
                                             options={'maxiter': 100, 'ftol': 1e-6})

                            if result.success:
                                refined_points = result.x.reshape(-1, 3)
                                refined_norms = np.linalg.norm(refined_points, axis=1)
                                if np.max(refined_norms) > 1:
                                    refined_points = refined_points / np.max(refined_norms) * 0.99

                                refined_ratio = compute_min_max_ratio(refined_points)
                                if refined_ratio > sa_best_ratio:
                                    sa_best_points = refined_points.copy()
                                    sa_best_ratio = refined_ratio
                                    last_improvement = iteration
                        except:
                            pass

                    # Early stopping if no improvement for long time
                    if iteration - last_improvement > 15000:
                        break

                    # Restart from best checkpoint if no improvement in a while
                    if iteration - last_checkpoint > 20000 and last_checkpoint > 0:
                        sa_best_points = best_checkpoint_points.copy()
                        sa_best_ratio = best_checkpoint_ratio
                        last_improvement = iteration

                if sa_best_ratio > best_ratio:
                    best_ratio = sa_best_ratio
                    best_points = sa_best_points.copy()
                    best_seed = seed

        return best_points, best_ratio

    # Multiple restart strategy with different initialization methods
    best_points = None
    best_ratio = 0.0

    # Strategy 1: Enhanced Fibonacci sphere initialization with multiple seeds and variations
    # Generate multiple Fibonacci configurations with different parameterizations
    fib_configurations = []

    # Original Fibonacci sphere
    fib_configurations.append(("original", lambda: fibonacci_sphere(14)))

    # Modified Fibonacci with different offsets
    np.random.seed(1001)
    offset1 = np.random.uniform(0, 1, 14)
    fib_configurations.append(("offset1", lambda: fibonacci_sphere_offset(14, offset1)))

    np.random.seed(1002)
    offset2 = np.random.uniform(0, 1, 14)
    fib_configurations.append(("offset2", lambda: fibonacci_sphere_offset(14, offset2)))

    # Different parameterization approaches
    fib_configurations.append(("golden", lambda: fibonacci_sphere_golden(14)))
    fib_configurations.append(("alternative", lambda: fibonacci_sphere_alternative(14)))
    fib_configurations.append(("custom", lambda: fibonacci_sphere_custom(14)))
    fib_configurations.append(("physics", lambda: fibonacci_sphere_physics(14)))
    fib_configurations.append(("lucas", lambda: fibonacci_sphere_lucas(14)))

    # Test different Fibonacci configurations with more seeds
    for config_name, fib_func in fib_configurations:
        # Try multiple random seeds for each configuration type
        for seed in [42, 123, 456, 789, 999, 111, 222, 333, 555, 666, 777, 888]:  # More seeds for better exploration
            np.random.seed(seed)
            points = fib_func()
            # Add controlled noise
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1)
            if np.max(norms) > 0:
                points = points / np.max(norms) * 0.9
            points = project_to_sphere(points)
            points, ratio = two_phase_optimization(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()

    # Strategy 4: Spherical Voronoi initialization for better point distribution
    sv_points = spherical_voronoi_init(14)
    if sv_points is not None:
        points, ratio = two_phase_optimization(sv_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    # Strategy 2: Simulated Annealing with multiple restarts
    sa_points, sa_ratio = simulated_annealing_multiple_starts()
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()

    # Strategy 3: Perturbed Fibonacci sphere
    if best_points is None:
        np.random.seed(42)
        points = fibonacci_sphere(14)
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        norms = np.linalg.norm(points, axis=1)
        if np.max(norms) > 0:
            points = points / np.max(norms) * 0.9
        points = project_to_sphere(points)
        points, ratio = two_phase_optimization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    # Final fallback to ensure we have a valid result
    if best_points is None:
        # Fallback to original initialization
        np.random.seed(42)
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(14):
            theta = np.arccos(1 - 2 * (i / 13))
            phi = np.mod(i * golden_ratio, 1) * 2 * np.pi

            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            points.append([x + np.random.normal(0, 0.01),
                          y + np.random.normal(0, 0.01),
                          z + np.random.normal(0, 0.01)])

        points = np.array(points)
        points = project_to_sphere(points)
        best_points = points

    return best_points


# EVOLVE-BLOCK-END