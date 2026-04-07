# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, falling back to pure Python")

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # JIT compiled functions for performance
    if NUMBA_AVAILABLE:
        @jit(nopython=True, parallel=True)
        def compute_distances_numba(points):
            """Compute pairwise distances using numba for speed"""
            n = points.shape[0]
            distances = np.zeros((n, n))
            for i in prange(n):
                for j in range(i+1, n):
                    dist = 0.0
                    for k in range(3):
                        diff = points[i, k] - points[j, k]
                        dist += diff * diff
                    dist = np.sqrt(dist)
                    distances[i, j] = dist
                    distances[j, i] = dist
            return distances

        @jit(nopython=True)
        def compute_min_max_ratio_numba(points):
            """Compute min/max ratio using numba"""
            n = points.shape[0]
            min_dist = np.inf
            max_dist = 0.0

            for i in range(n):
                for j in range(i+1, n):
                    dist = 0.0
                    for k in range(3):
                        diff = points[i, k] - points[j, k]
                        dist += diff * diff
                    dist = np.sqrt(dist)
                    if dist > 0 and dist < min_dist:
                        min_dist = dist
                    if dist > max_dist:
                        max_dist = dist

            if max_dist == 0:
                return 0.0
            return min_dist / max_dist if min_dist != np.inf else 0.0
    else:
        # Fallback to pure python versions
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

        def compute_distances(points):
            return pdist(points)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Use numba version if available
        if NUMBA_AVAILABLE:
            return compute_min_max_ratio_numba(points)
        else:
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

    def adaptive_perturb_smart(points, point_idx, distances):
        """Apply adaptive perturbations with smarter selection logic."""
        # Get distances to all other points
        dist_to_others = distances[point_idx]
        dist_to_others[point_idx] = np.inf  # Exclude self-distance

        # Find nearest and furthest points
        nearest_idx = np.argmin(dist_to_others)
        furthest_idx = np.argmax(dist_to_others)

        # Calculate adaptive perturbation scale based on local density
        mean_dist = np.mean(dist_to_others)
        perturbation_scale = max(0.001, 0.005 * (1.0 + 0.2 * mean_dist))  # Minimum scale of 0.001

        # Determine whether to move this point based on its local configuration
        nearest_dist = dist_to_others[nearest_idx]
        furthest_dist = dist_to_others[furthest_idx]
        
        # Create perturbation vector
        delta = np.random.normal(0, perturbation_scale, 3)

        # Smarter direction selection based on local geometry
        if nearest_dist < furthest_dist * 0.8:  # Point is relatively close to someone
            # Move towards its nearest neighbor to increase minimum distance
            direction = points[nearest_idx] - points[point_idx]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                delta += 0.7 * direction * perturbation_scale
        else:  # Point is relatively far from others
            # Move away from furthest neighbor to decrease maximum distance
            direction = points[point_idx] - points[furthest_idx]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                delta -= 0.5 * direction * perturbation_scale

        return delta

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
                                options={'maxiter': 200, 'ftol': 1e-6},
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
                # Perturb a few points with smarter selection
                for _ in range(3):
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
                            options={'maxiter': 400, 'ftol': 1e-8},
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

    def adaptive_simulated_annealing():
        """Run enhanced adaptive simulated annealing with better cooling and restarts."""
        best_points = None
        best_ratio = 0.0
        max_time = 340  # Reserve 10 seconds for cleanup
        
        # Try multiple restarts with different seeds
        seeds = [42, 123, 456, 789, 999, 246, 135]
        
        for seed in seeds:
            np.random.seed(seed)
            
            # Initialize with Fibonacci sphere arrangement
            points = fibonacci_sphere(14)
            
            # Add controlled noise
            noise_scale = 0.015
            points += np.random.normal(0, noise_scale, points.shape)
            
            # Project to unit sphere
            points = project_to_sphere(points)
            
            # Initialize tracking variables
            current_temp = 1.0
            min_temp = 1e-12
            max_iter = 80000  # Increased iterations for better exploration
            last_improvement = 0
            time_start = time.time()
            
            # Initialize best solution
            sa_best_points = points.copy()
            sa_best_ratio = compute_min_max_ratio(sa_best_points)
            
            # Checkpoint tracking
            checkpoint_points = sa_best_points.copy()
            checkpoint_ratio = sa_best_ratio
            last_checkpoint_time = time_start
            
            # Track recent improvements for adaptive cooling
            recent_improvements = []
            improvement_window = 1000
            
            # Main optimization loop
            for iteration in range(max_iter):
                if time.time() - time_start > max_time:
                    break
                    
                # Dynamic cooling schedule based on performance
                if iteration < 15000:
                    cooling_rate = 0.9996
                elif iteration < 40000:
                    cooling_rate = 0.9997
                else:
                    cooling_rate = 0.9998
                
                # Aggressive cooling when stuck
                if iteration - last_improvement > 15000:
                    cooling_rate = max(cooling_rate, 0.99992)
                    
                # Cool down temperature
                current_temp *= cooling_rate

                if current_temp < min_temp:
                    break

                # Smart perturbation: consider neighborhood information
                test_points = sa_best_points.copy()
                
                # Select random point to perturb with preference for problematic points
                if np.random.random() < 0.7:  # 70% chance to use smart selection
                    # Try to select a point that could benefit most from perturbation
                    distances = pdist(sa_best_points)
                    # Reshape to matrix form
                    dist_matrix = distances.reshape(len(sa_best_points), len(sa_best_points))
                    # Find points with extreme distances (either very close or very far)
                    min_distances = np.min(dist_matrix, axis=1)
                    max_distances = np.max(dist_matrix, axis=1)
                    # Prefer points with either very small min or very large max
                    scores = min_distances + (1.0 - max_distances/np.max(max_distances))
                    point_idx = np.argmax(scores)  # Select point with highest score
                else:
                    point_idx = np.random.randint(0, 14)
                
                # Get neighborhood info for smart perturbation
                distances = pdist(sa_best_points)
                # Reshape to matrix form
                dist_matrix = distances.reshape(len(sa_best_points), len(sa_best_points))
                _, nearest_distances = np.unique(dist_matrix[point_idx], return_index=True)[:5]
                
                # Adjust perturbation size based on local geometry
                if len(nearest_distances) > 0:
                    avg_nearest_dist = np.mean(nearest_distances)
                    # Use more conservative scaling
                    perturbation_scale = 0.0008 * (1.0 + 0.3 * avg_nearest_dist)
                else:
                    perturbation_scale = 0.0008
                
                # Make perturbation using smart function
                delta = adaptive_perturb_smart(sa_best_points, point_idx, dist_matrix)
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
                if len(recent_improvements) > improvement_window:
                    recent_improvements.pop(0)

                # Periodic checkpointing and refinement
                if iteration > 0 and iteration % 1500 == 0:
                    # Save checkpoint
                    if sa_best_ratio > checkpoint_ratio:
                        checkpoint_points = sa_best_points.copy()
                        checkpoint_ratio = sa_best_ratio
                    
                    # Local refinement every 1500 iterations
                    try:
                        # Only perform local refinement if sufficient time remains
                        if time.time() - time_start < max_time - 5:
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

                            result = minimize(objective, x0, method='SLSQP',
                                            constraints=cons,
                                            options={'maxiter': 40, 'ftol': 1e-6})

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
                if iteration - last_improvement > 10000:
                    break

            # Update global best if this run was better
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
    fib_configurations.append(("alternative", lambda: fibonacci_sphere_alternative(14)))
    fib_configurations.append(("custom", lambda: fibonacci_sphere_custom(14)))

    # Additional diverse initialization strategies
    fib_configurations.append(("random", lambda: np.random.uniform(-1, 1, (14, 3))))

    # Test different Fibonacci configurations with more diverse seeds
    for config_name, fib_func in fib_configurations:
        # Try more random seeds for better exploration
        for seed in [42, 123, 456, 789, 999, 135, 246]:  # More seeds for better coverage
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

    # Strategy 2: Enhanced adaptive simulated annealing
    sa_points, sa_ratio = adaptive_simulated_annealing()
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