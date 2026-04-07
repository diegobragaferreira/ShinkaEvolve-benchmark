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

    def simulated_annealing_multiple_starts():
        """Run simulated annealing multiple times with different random seeds."""
        best_points = None
        best_ratio = 0.0
        best_seed = 0

        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789, 999]:
            np.random.seed(seed)

            # Initialize with Fibonacci sphere arrangement
            golden_ratio = (1 + np.sqrt(5)) / 2
            points = []

            for i in range(14):
                theta = np.arccos(1 - 2 * (i / 13))
                phi = np.mod(i * golden_ratio, 1) * 2 * np.pi

                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                # Add controlled noise
                points.append([x + np.random.normal(0, 0.01),
                              y + np.random.normal(0, 0.01),
                              z + np.random.normal(0, 0.01)])

            points = np.array(points)

            # Project to unit sphere
            points = project_to_sphere(points)

            # Simulated Annealing parameters
            current_temp = 1.0
            min_temp = 1e-10
            max_iter = 100000
            accept_threshold = 0.01
            adaptation_window = 2000  # Window for adaptation

            # Initialize best solution
            sa_best_points = points.copy()
            sa_best_ratio = compute_min_max_ratio(sa_best_points)
            last_improvement = 0
            improvement_history = []  # Track recent improvements

            start_time = time.time()
            max_time = 350  # Leave 10 seconds for cleanup

            for iteration in range(max_iter):
                if time.time() - start_time > max_time:
                    break

                # Dynamic cooling schedule with more sophisticated adaptation
                if iteration < 10000:
                    # Initial aggressive cooling
                    cooling_rate = 0.9995
                elif iteration < 50000:
                    # Medium cooling rate
                    cooling_rate = 0.9997
                else:
                    # Fine-tuning phase with very slow cooling
                    cooling_rate = 0.99995

                # Adaptive cooling based on recent performance
                if len(improvement_history) >= 10:
                    recent_improvements = improvement_history[-10:]
                    if sum(recent_improvements) < 0.0001:  # Very slow improvement
                        cooling_rate = max(cooling_rate, 0.99998)
                    elif sum(recent_improvements) > 0.001:  # Fast improvement
                        cooling_rate = max(cooling_rate, 0.9996)

                # Cool down temperature
                current_temp *= cooling_rate

                if current_temp < min_temp:
                    break

                # Try to make a small random move
                test_points = sa_best_points.copy()

                # Select random point to perturb
                point_idx = np.random.randint(0, 14)

                # Smart perturbation based on neighborhood information
                # Get distances to all other points
                distances = pdist(np.vstack([sa_best_points[point_idx], sa_best_points]))[0]
                distances[point_idx] = np.inf  # Exclude self-distance
                nearest_idx = np.argmin(distances)
                furthest_idx = np.argmax(distances)
                avg_dist = np.mean(distances)

                # Adjust perturbation size based on local geometry
                if len(distances) > 0:
                    # Scale perturbation based on local point density
                    perturbation_scale = 0.0005 + 0.0005 * (avg_dist / 1.0)
                else:
                    perturbation_scale = 0.001

                # Make adaptive perturbation
                delta = np.random.normal(0, perturbation_scale, 3)

                # Directional bias to improve min/max ratio
                if distances[nearest_idx] < distances[furthest_idx]:
                    # Move closer to nearest point to increase min distance
                    direction = sa_best_points[nearest_idx] - sa_best_points[point_idx]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        delta += 0.3 * direction * perturbation_scale
                else:
                    # Move away from furthest point to decrease max distance
                    direction = sa_best_points[point_idx] - sa_best_points[furthest_idx]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        delta -= 0.3 * direction * perturbation_scale

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
                    improvement_history.append(test_ratio - sa_best_ratio)
                elif np.random.random() < np.exp((test_ratio - sa_best_ratio) / current_temp):
                    sa_best_points = test_points.copy()
                    sa_best_ratio = test_ratio
                    last_improvement = iteration
                    improvement_history.append(test_ratio - sa_best_ratio)
                else:
                    improvement_history.append(0.0)

                # Keep history size manageable
                if len(improvement_history) > 50:
                    improvement_history.pop(0)

                # Local refinement every 2000 iterations
                if iteration > 0 and iteration % 2000 == 0:
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

                # Aggressive early stopping based on multiple criteria
                if iteration - last_improvement > 15000:
                    # If no significant improvement for a long time, stop
                    if len(improvement_history) > 0 and sum(improvement_history[-5:]) < 1e-8:
                        break
                if iteration - last_improvement > 20000:
                    # Very strict early stopping
                    break

            if sa_best_ratio > best_ratio:
                best_ratio = sa_best_ratio
                best_points = sa_best_points.copy()
                best_seed = seed

        return best_points, best_ratio

    # Run optimization with multiple restarts
    points, ratio = simulated_annealing_multiple_starts()

    # Final fallback to ensure we have a valid result
    if points is None or ratio == 0.0:
        # Fallback to initial configuration
        np.random.seed(42)
        golden_ratio = (1 + np.sqrt(5)) / 2
        points = []
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

    return points


# EVOLVE-BLOCK-END