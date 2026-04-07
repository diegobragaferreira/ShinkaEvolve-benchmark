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

    class PointOptimizer:
        def __init__(self):
            self.best_solution = None
            self.best_ratio = 0.0
            self.start_time = time.time()
            self.max_time = 350  # Leave 10 seconds for cleanup
            
        def compute_min_max_ratio(self, points):
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

        def project_to_unit_sphere(self, points):
            """Project points onto unit sphere while maintaining relative positions."""
            norms = np.linalg.norm(points, axis=1)
            # Avoid division by zero
            norms = np.where(norms == 0, 1.0, norms)
            return points / norms[:, np.newaxis] * 0.99

        def fibonacci_sphere(self, n):
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

        def get_neighbor_info(self, points, point_idx, k=5):
            """Get information about k nearest neighbors for a given point."""
            distances = cdist([points[point_idx]], points)[0]
            distances[point_idx] = np.inf  # Exclude self
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            return nearest_indices, nearest_distances

        def adaptive_perturbation(self, points, point_idx, distances):
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

        def simulated_annealing(self, initial_points, seed):
            """Run simulated annealing optimization with adaptive cooling."""
            np.random.seed(seed)
            
            # Initialize tracking variables
            current_points = initial_points.copy()
            sa_best_points = initial_points.copy()
            sa_best_ratio = self.compute_min_max_ratio(sa_best_points)
            
            # Adaptive cooling schedule
            current_temp = 1.0
            min_temp = 1e-10
            last_improvement = 0
            
            # Main optimization loop
            iteration = 0
            while True:
                if time.time() - self.start_time > self.max_time:
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
                    
                # Compute current distances for efficient access
                distances = cdist(current_points, current_points, 'euclidean')
                
                # Try to make a small random move
                test_points = current_points.copy()

                # Select random point to perturb
                point_idx = np.random.randint(0, len(current_points))

                # Smart perturbation: consider neighborhood information
                _, nearest_distances = self.get_neighbor_info(current_points, point_idx)
                
                # Adjust perturbation size based on local geometry
                if len(nearest_distances) > 0:
                    avg_nearest_dist = np.mean(nearest_distances)
                    perturbation_scale = 0.001 * (1.0 + 0.5 * avg_nearest_dist)  # Scale based on local density
                else:
                    perturbation_scale = 0.001
                
                # Make adaptive perturbation
                delta = self.adaptive_perturbation(current_points, point_idx, distances)
                test_points[point_idx] += delta

                # Project back to sphere
                test_points = self.project_to_unit_sphere(test_points)

                # Compute new ratio
                test_ratio = self.compute_min_max_ratio(test_points)

                # Accept or reject based on Metropolis criterion
                if test_ratio > sa_best_ratio:
                    sa_best_points = test_points.copy()
                    sa_best_ratio = test_ratio
                    last_improvement = iteration
                elif np.random.random() < np.exp((test_ratio - sa_best_ratio) / current_temp):
                    sa_best_points = test_points.copy()
                    sa_best_ratio = test_ratio
                    last_improvement = iteration

                # Update current solution
                current_points = sa_best_points.copy()
                
                iteration += 1
                
                # Early stopping if no improvement for long time
                if iteration - last_improvement > 15000:
                    break

            return sa_best_points, sa_best_ratio

        def optimize_with_multiple_starts(self):
            """Run optimization with multiple restarts using different strategies."""
            # Strategy 1: Fibonacci sphere initialization
            fib_points = self.fibonacci_sphere(14)
            fib_points = self.project_to_unit_sphere(fib_points)
            # Add small noise
            fib_points += np.random.normal(0, 0.01, fib_points.shape)
            fib_points = self.project_to_unit_sphere(fib_points)
            
            points, ratio = self.simulated_annealing(fib_points, 42)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()

            # Strategy 2: Random initialization with normalization
            np.random.seed(123)
            rand_points = np.random.uniform(-1, 1, (14, 3))
            rand_points = self.project_to_unit_sphere(rand_points)
            points, ratio = self.simulated_annealing(rand_points, 123)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()

            # Strategy 3: Perturbed Fibonacci sphere
            np.random.seed(456)
            pert_fib_points = fib_points.copy()
            pert_fib_points += np.random.normal(0, 0.02, pert_fib_points.shape)
            pert_fib_points = self.project_to_unit_sphere(pert_fib_points)
            points, ratio = self.simulated_annealing(pert_fib_points, 456)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()

            # Strategy 4: Another Fibonacci variation
            np.random.seed(789)
            fib_points2 = self.fibonacci_sphere(14)
            fib_points2 = self.project_to_unit_sphere(fib_points2)
            fib_points2 += np.random.normal(0, 0.015, fib_points2.shape)
            fib_points2 = self.project_to_unit_sphere(fib_points2)
            points, ratio = self.simulated_annealing(fib_points2, 789)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()

        def finalize_result(self):
            """Finalize and return the best result."""
            # If no successful optimization, fall back to the best Fibonacci initialization
            if self.best_solution is None:
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
                points = self.project_to_unit_sphere(points)
                self.best_solution = points

            return self.best_solution

    # Create optimizer instance and run optimization
    optimizer = PointOptimizer()
    optimizer.optimize_with_multiple_starts()
    return optimizer.finalize_result()


# EVOLVE-BLOCK-END