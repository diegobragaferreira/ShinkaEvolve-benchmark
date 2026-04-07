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

    def create_voronoi_force_adjustment(points, alpha=0.01, beta=0.005):
        """
        Compute force-based adjustment using SphericalVoronoi.
        This creates a repulsion/attraction force field based on Voronoi cells.
        """
        if len(points) < 2:
            return points

        try:
            # Create SphericalVoronoi diagram
            sv = SphericalVoronoi(points, radius=1.0)

            # Get Voronoi vertices
            vertices = sv.vertices

            # For each point, compute adjustments based on Voronoi structure
            adjusted_points = points.copy()

            # Compute forces from Voronoi geometry
            for i in range(len(points)):
                # Find Voronoi cell vertices associated with this point
                # In practice, we'll compute simpler distance-based forces

                # Compute distance to all other points
                distances = np.linalg.norm(points[i] - points, axis=1)
                distances[i] = np.inf  # Ignore self-distance

                # Find nearest and furthest points
                nearest_idx = np.argmin(distances)
                furthest_idx = np.argmax(distances)

                # Apply force to balance distances
                # Move towards nearest to increase min distance
                if distances[nearest_idx] > 0:
                    direction = points[nearest_idx] - points[i]
                    force_magnitude = alpha * (1.0 / distances[nearest_idx])
                    adjusted_points[i] += force_magnitude * direction / np.linalg.norm(direction)

                # Move away from furthest to decrease max distance
                if distances[furthest_idx] > 0:
                    direction = points[i] - points[furthest_idx]
                    force_magnitude = beta * (1.0 / distances[furthest_idx])
                    adjusted_points[i] -= force_magnitude * direction / np.linalg.norm(direction)

            return adjusted_points

        except:
            # Fallback to simple distance-based adjustment
            return points

    def adaptive_local_optimization(points, max_iter=100):
        """Apply adaptive local optimization to improve the current configuration."""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(points)

        for iteration in range(max_iter):
            # Compute current ratio
            current_ratio = compute_min_max_ratio(current_points)

            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()

            # Apply Voronoi-based adjustment
            adjusted_points = create_voronoi_force_adjustment(current_points)

            # Apply some random perturbations to maintain diversity
            if iteration % 10 == 0:
                for i in range(len(adjusted_points)):
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.001, 3)
                    adjusted_points[i] += perturbation

            # Ensure points stay on sphere
            norms = np.linalg.norm(adjusted_points, axis=1)
            adjusted_points = adjusted_points / norms[:, np.newaxis] * 0.99

            # Check if we should accept this adjustment
            test_ratio = compute_min_max_ratio(adjusted_points)

            if test_ratio > current_ratio:
                current_points = adjusted_points.copy()
            elif np.random.random() < 0.1:  # Accept some bad moves occasionally
                current_points = adjusted_points.copy()

        return best_points, best_ratio

    def hybrid_optimization(initial_points, max_iterations=3000):
        """Combine multiple optimization strategies."""
        current_points = initial_points.copy()
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)

        # Track improvement history
        improvement_history = []
        patience = 0
        max_patience = 500

        for iteration in range(max_iterations):
            # Adaptive cooling schedule
            cooling_rate = max(0.99, 1.0 - iteration / (max_iterations * 2.0))

            # Sometimes do large jump for exploration
            if iteration % 100 == 0 and iteration > 0:
                # Random perturbation
                noise = np.random.normal(0, 0.01, current_points.shape)
                current_points += noise
                norms = np.linalg.norm(current_points, axis=1)
                current_points = current_points / norms[:, np.newaxis] * 0.99

            # Apply force-based adjustment
            adjusted_points = create_voronoi_force_adjustment(current_points)

            # Apply local optimization periodically
            if iteration % 50 == 0:
                optimized_points, optimized_ratio = adaptive_local_optimization(adjusted_points, 50)
                if optimized_ratio > compute_min_max_ratio(adjusted_points):
                    adjusted_points = optimized_points
                    current_points = adjusted_points.copy()

            # Accept or reject with probability based on improvement
            test_ratio = compute_min_max_ratio(adjusted_points)

            if test_ratio > best_ratio:
                best_ratio = test_ratio
                best_points = adjusted_points.copy()
                improvement_history.append(1)
                patience = 0
            elif np.random.random() < np.exp((test_ratio - best_ratio) * 0.5):
                current_points = adjusted_points.copy()
                improvement_history.append(1)
                patience = 0
            else:
                improvement_history.append(0)
                patience += 1

            # Early stopping if no improvement for too long
            if len(improvement_history) > 100:
                if np.sum(improvement_history[-100:]) < 5:
                    break

            # Reduce cooling rate after significant improvement
            if patience > 100:
                cooling_rate *= 0.99

        return best_points, best_ratio

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

        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789, 999]:
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
            max_iter = 50000  # Increased iterations
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

                # Local refinement every 1500 iterations (more frequent)
                if iteration > 0 and iteration % 1500 == 0:
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

    # Test different Fibonacci configurations with more seeds
    for config_name, fib_func in fib_configurations:
        # Try multiple random seeds for each configuration type
        for seed in [42, 123, 456, 789]:  # More seeds
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

    # Strategy 2: Hybrid optimization with Voronoi forces
    if best_points is None:
        try:
            np.random.seed(42)
            points = fibonacci_sphere(14)
            noise = np.random.normal(0, 0.02, points.shape)
            points += noise
            norms = np.linalg.norm(points, axis=1)
            if np.max(norms) > 0:
                points = points / np.max(norms) * 0.9
            points = project_to_sphere(points)

            # Use hybrid optimization approach from the better performing program
            optimized_points, ratio = hybrid_optimization(points, 2000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except:
            pass

    # Strategy 3: Simulated Annealing with multiple restarts
    sa_points, sa_ratio = simulated_annealing_multiple_starts()
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()

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