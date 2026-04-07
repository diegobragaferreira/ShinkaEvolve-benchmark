# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
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

    def fibonacci_sphere_offset(n, offset):
        """Generate n points distributed approximately uniformly on a sphere with phase offset."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi + offset), 
                              np.cos(i * 2 * np.pi / phi + offset))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def adaptive_perturb_strategy(points, point_idx, distances):
        """Apply adaptive perturbation based on local point relationships."""
        # Get distances to all other points
        dist_to_others = distances[point_idx]
        dist_to_others[point_idx] = np.inf  # Exclude self-distance

        # Find nearest and furthest points
        nearest_idx = np.argmin(dist_to_others)
        furthest_idx = np.argmax(dist_to_others)

        # Calculate adaptive perturbation scale based on local density
        mean_dist = np.mean(dist_to_others)
        std_dist = np.std(dist_to_others)
        # Scale with local density and variance
        perturbation_scale = 0.008 * (1.0 + 0.3 * mean_dist + 0.1 * std_dist)  

        # Create perturbation vector
        delta = np.random.normal(0, perturbation_scale, 3)

        # Adjust perturbation direction to improve min/max ratio
        # Prioritize movement that increases min distance
        if dist_to_others[nearest_idx] < dist_to_others[furthest_idx]:
            # Move closer to nearest point to increase min distance
            direction = points[nearest_idx] - points[point_idx]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                delta += 0.7 * direction * perturbation_scale

        # Move away from furthest point to decrease max distance
        direction = points[point_idx] - points[furthest_idx]
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            delta -= 0.4 * direction * perturbation_scale

        return delta

    def two_phase_optimization(initial_points):
        """Perform two-phase optimization: global search then local refinement."""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)

        # Phase 1: Global search with modified optimization
        current_points = initial_points.copy()

        # Run multiple iterations of optimization with different strategies
        for iter_num in range(4):
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
                                options={'maxiter': 400, 'ftol': 1e-6},
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
            if iter_num < 3:  # Don't perturb on the last iteration
                current_points = best_points.copy()
                # Perturb a few points more systematically
                num_perturb = min(3, len(current_points)//2)
                for _ in range(num_perturb):
                    idx = np.random.randint(0, len(current_points))
                    delta = np.random.normal(0, 0.006, 3)
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
                            options={'maxiter': 800, 'ftol': 1e-8},
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

    def enhanced_simulated_annealing_multiple_starts():
        """Run enhanced simulated annealing multiple times with different random seeds."""
        best_points = None
        best_ratio = 0.0
        best_seed = 0
        
        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789, 999, 1001, 1002]:
            np.random.seed(seed)
            
            # Initialize with multiple Fibonacci sphere variants
            points_list = []
            
            # Original Fibonacci sphere
            points_list.append(fibonacci_sphere(14))
            
            # Offset Fibonacci spheres with different phases
            for offset in [0.1, 0.3, 0.5, 0.7]:
                points_list.append(fibonacci_sphere_offset(14, offset))
            
            # Select best initial configuration
            best_initial = None
            best_initial_ratio = 0.0
            
            for points in points_list:
                # Add controlled noise
                noise = np.random.normal(0, 0.01, points.shape)
                points += noise
                
                # Normalize to unit sphere
                norms = np.linalg.norm(points, axis=1)
                if np.max(norms) > 0:
                    points = points / np.max(norms) * 0.9

                # Project to unit sphere
                points = project_to_sphere(points)
                
                ratio = compute_min_max_ratio(points)
                if ratio > best_initial_ratio:
                    best_initial_ratio = ratio
                    best_initial = points.copy()
            
            points = best_initial
            
            # Simulated Annealing parameters (enhanced)
            current_temp = 1.0
            min_temp = 1e-10
            base_cooling_rate = 0.9995
            max_iter = 50000
            
            # Track recent improvements for adaptive cooling
            recent_improvements = []
            improvement_window = 500
            max_improvement_window = 3000

            # Initialize best solution
            sa_best_points = points.copy()
            sa_best_ratio = compute_min_max_ratio(sa_best_points)

            start_time = time.time()
            last_improvement = 0
            improvement_counter = 0

            for iteration in range(max_iter):
                # Dynamic cooling schedule based on improvement rate
                cooling_rate = base_cooling_rate
                
                # Adjust cooling based on recent improvements
                if len(recent_improvements) >= improvement_window:
                    recent_improvement_count = sum(recent_improvements[-improvement_window:])
                    improvement_rate = recent_improvement_count / improvement_window
                    
                    if improvement_rate < 0.03:  # Very slow improvement
                        cooling_rate = min(0.99998, cooling_rate * 1.02)
                    elif improvement_rate < 0.08:  # Slow improvement
                        cooling_rate = min(0.99995, cooling_rate * 1.01)
                    elif improvement_rate < 0.15:  # Moderate improvement
                        cooling_rate = min(0.9999, cooling_rate * 1.005)
                    else:  # Fast improvement
                        cooling_rate = max(0.9997, cooling_rate * 0.998)
                
                # More aggressive early cooling
                if iteration < 10000:
                    cooling_rate = min(cooling_rate, 0.9998)
                elif iteration < 25000:
                    cooling_rate = min(cooling_rate, 0.9997)
                else:
                    cooling_rate = min(cooling_rate, 0.9996)
                
                # Cool down temperature
                current_temp *= cooling_rate

                if current_temp < min_temp:
                    break

                # Try to make a small random move with adaptive perturbation
                test_points = sa_best_points.copy()

                # Smart point selection based on distance characteristics
                # Select points that are either very close or very far for more impactful moves
                distances = pdist(sa_best_points)
                distances_matrix = distances.reshape(len(sa_best_points), -1)
                
                # Find points with extreme local distances
                local_means = np.mean(distances_matrix, axis=1)
                local_stds = np.std(distances_matrix, axis=1)
                
                # Weighted selection - prefer points with larger variance in distances  
                # or those that are more isolated
                weights = local_stds + 0.1 * local_means
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Select point with probability proportional to weights
                point_idx = np.random.choice(len(sa_best_points), p=weights)
                
                # Make adaptive perturbation
                delta = adaptive_perturb_strategy(sa_best_points, point_idx, distances_matrix[point_idx])
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
                    improvement_counter += 1
                    # Record improvement
                    recent_improvements.append(1)
                elif np.random.random() < np.exp((test_ratio - sa_best_ratio) / current_temp):
                    sa_best_points = test_points.copy()
                    sa_best_ratio = test_ratio
                    last_improvement = iteration
                    improvement_counter += 1
                    # Record improvement
                    recent_improvements.append(1)
                else:
                    # Record no improvement
                    recent_improvements.append(0)

                # Keep recent improvements window fixed
                if len(recent_improvements) > max_improvement_window:
                    recent_improvements.pop(0)

                # Local refinement at specific intervals with better frequency
                if iteration > 0 and iteration % 2500 == 0:
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
                                         options={'maxiter': 150, 'ftol': 1e-6})

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
                                improvement_counter += 1
                    except:
                        pass

                # Early stopping if no significant improvement for long time
                if iteration - last_improvement > 12000:
                    break

            if sa_best_ratio > best_ratio:
                best_ratio = sa_best_ratio
                best_points = sa_best_points.copy()
                best_seed = seed

        return best_points, best_ratio

    # Multiple restart strategy with enhanced initialization methods
    best_points = None
    best_ratio = 0.0

    # Strategy 1: Enhanced Fibonacci sphere initialization with multiple seeds and variations
    seeds = [42, 123, 456, 789, 999, 1001, 1002]
    for seed in seeds:
        np.random.seed(seed)
        points = fibonacci_sphere(14)
        # Add more controlled noise with variable magnitude
        noise_magnitude = 0.01 + np.random.random() * 0.01
        noise = np.random.normal(0, noise_magnitude, points.shape)
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

    # Strategy 2: Enhanced Simulated Annealing with multiple restarts
    sa_points, sa_ratio = enhanced_simulated_annealing_multiple_starts()
    if sa_ratio > best_ratio:
        best_ratio = sa_ratio
        best_points = sa_points.copy()

    # Strategy 3: Perturbed Fibonacci sphere with enhanced approach
    if best_points is None:
        np.random.seed(42)
        points = fibonacci_sphere(14)
        # Use more significant noise for better exploration
        noise = np.random.normal(0, 0.025, points.shape)
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