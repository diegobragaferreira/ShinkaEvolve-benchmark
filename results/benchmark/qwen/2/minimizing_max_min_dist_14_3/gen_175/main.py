# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

    def spherical_voronoi_init(n, num_attempts=3):
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

        # Try multiple restarts with different seeds
        seeds = [42, 123, 456, 789, 999]
        for seed in seeds:
            np.random.seed(seed)

            # Initialize with Fibonacci sphere arrangement
            points = fibonacci_sphere(14)

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
            max_iter = 20000  # Reduced iterations for faster execution

            # Initialize best solution
            sa_best_points = points.copy()
            sa_best_ratio = compute_min_max_ratio(sa_best_points)

            last_improvement = 0

            for iteration in range(max_iter):
                # Adaptive cooling schedule
                cooling_rate = base_cooling_rate
                if iteration < 5000:
                    cooling_rate = min(cooling_rate, 0.9998)
                elif iteration < 15000:
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
                elif np.random.random() < np.exp((test_ratio - sa_best_ratio) / current_temp):
                    sa_best_points = test_points.copy()
                    sa_best_ratio = test_ratio
                    last_improvement = iteration

                # Early stopping if no improvement for long time
                if iteration - last_improvement > 5000:
                    break

                # Apply local refinement every 2000 iterations
                if iteration > 0 and iteration % 2000 == 0:
                    # Perform local optimization on current best points
                    try:
                        # Flatten points for optimization
                        x0 = sa_best_points.flatten()

                        def constraint_func(x):
                            # Ensure points stay within unit sphere
                            points = x.reshape(-1, 3)
                            norms = np.linalg.norm(points, axis=1)
                            return 1.0 - norms

                        # Use SLSQP for local refinement
                        result = minimize(
                            lambda x: -compute_min_max_ratio(x.reshape(-1, 3)),
                            x0,
                            method='SLSQP',
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 100, 'ftol': 1e-6},
                            bounds=[(-1, 1)] * 42
                        )

                        if result.success:
                            refined_points = result.x.reshape(-1, 3)
                            # Ensure points stay on unit sphere
                            norms_refined = np.linalg.norm(refined_points, axis=1)
                            if np.max(norms_refined) > 1:
                                refined_points = refined_points / np.max(norms_refined) * 0.99

                            refined_ratio = compute_min_max_ratio(refined_points)
                            if refined_ratio > sa_best_ratio:
                                sa_best_points = refined_points.copy()
                                sa_best_ratio = refined_ratio
                                last_improvement = iteration
                    except:
                        pass

            if sa_best_ratio > best_ratio:
                best_ratio = sa_best_ratio
                best_points = sa_best_points.copy()

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

    # Test different Fibonacci configurations
    for config_name, fib_func in fib_configurations:
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