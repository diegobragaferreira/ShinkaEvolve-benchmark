# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
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

        # Compute pairwise distances efficiently using cdist
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
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

    def get_neighbor_info(points, point_idx, k=5):
        """Get information about k nearest neighbors for a given point."""
        distances = cdist([points[point_idx]], points)[0]
        distances[point_idx] = np.inf  # Exclude self
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        return nearest_indices, nearest_distances

    def adaptive_simulated_annealing():
        """Run adaptive simulated annealing with multiple restarts."""
        best_points = None
        best_ratio = 0.0
        best_seed = 0
        max_time = 350  # Leave 10 seconds for cleanup
        
        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789, 999]:
            np.random.seed(seed)
            
            # Initialize with Fibonacci sphere arrangement
            points = fibonacci_sphere(14)
            
            # Add controlled noise
            noise_scale = 0.01
            points += np.random.normal(0, noise_scale, points.shape)
            
            # Project to unit sphere
            points = project_to_sphere(points)
            
            # Initialize tracking variables
            current_temp = 1.0
            min_temp = 1e-10
            max_iter = 100000
            last_improvement = 0
            time_start = time.time()
            
            # Initialize best solution
            sa_best_points = points.copy()
            sa_best_ratio = compute_min_max_ratio(sa_best_points)
            
            # Checkpoint tracking
            checkpoint_points = sa_best_points.copy()
            checkpoint_ratio = sa_best_ratio
            last_checkpoint_time = time_start
            
            # Main optimization loop
            for iteration in range(max_iter):
                if time.time() - time_start > max_time:
                    break
                    
                # Dynamic cooling schedule based on performance
                if iteration < 10000:
                    cooling_rate = 0.9995
                elif iteration < 50000:
                    cooling_rate = 0.9997
                else:
                    cooling_rate = 0.9999
                
                # More aggressive cooling when stuck
                if iteration - last_improvement > 20000:
                    cooling_rate = max(cooling_rate, 0.99995)
                    
                # Cool down temperature
                current_temp *= cooling_rate

                if current_temp < min_temp:
                    break

                # Smart perturbation: consider neighborhood information
                test_points = sa_best_points.copy()
                
                # Select random point to perturb
                point_idx = np.random.randint(0, 14)
                
                # Get neighborhood info for smart perturbation
                _, nearest_distances = get_neighbor_info(sa_best_points, point_idx)
                
                # Adjust perturbation size based on local geometry
                if len(nearest_distances) > 0:
                    avg_nearest_dist = np.mean(nearest_distances)
                    perturbation_scale = 0.001 * (1.0 + 0.5 * avg_nearest_dist)  # Scale based on local density
                else:
                    perturbation_scale = 0.001
                
                # Make small random perturbation
                delta = np.random.normal(0, perturbation_scale, 3)
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

                # Periodic checkpointing and refinement
                if iteration > 0 and iteration % 2000 == 0:
                    # Save checkpoint
                    if sa_best_ratio > checkpoint_ratio:
                        checkpoint_points = sa_best_points.copy()
                        checkpoint_ratio = sa_best_ratio
                    
                    # Local refinement every 2000 iterations
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
                                            options={'maxiter': 50, 'ftol': 1e-6})

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

            # Update global best if this run was better
            if sa_best_ratio > best_ratio:
                best_ratio = sa_best_ratio
                best_points = sa_best_points.copy()
                best_seed = seed

        return best_points, best_ratio

    # Run optimization with multiple restarts
    try:
        points, ratio = adaptive_simulated_annealing()
    except Exception as e:
        # Fallback to initial configuration if something goes wrong
        points = None
        ratio = 0.0

    # Final fallback to ensure we have a valid result
    if points is None or ratio == 0.0:
        # Fallback to initial configuration
        np.random.seed(42)
        points = fibonacci_sphere(14)
        points += np.random.normal(0, 0.01, points.shape)
        points = project_to_sphere(points)

    return points

# EVOLVE-BLOCK-END