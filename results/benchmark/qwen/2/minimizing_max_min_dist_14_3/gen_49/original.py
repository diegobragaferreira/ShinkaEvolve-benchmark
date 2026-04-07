# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

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

        # Compute pairwise distances
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

    def simulated_annealing():
        # Initialize with Fibonacci sphere arrangement
        np.random.seed(42)
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
        min_temp = 1e-8
        cooling_rate = 0.9995
        max_iter = 100000
        accept_threshold = 0.01

        # Initialize best solution
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)

        start_time = time.time()

        for iteration in range(max_iter):
            # Cool down temperature
            current_temp *= cooling_rate

            if current_temp < min_temp:
                break

            # Try to make a small random move
            test_points = best_points.copy()

            # Select random point to perturb
            point_idx = np.random.randint(0, 14)

            # Make small random perturbation
            delta = np.random.normal(0, 0.001, 3)
            test_points[point_idx] += delta

            # Project back to sphere
            test_points = project_to_sphere(test_points)

            # Compute new ratio
            test_ratio = compute_min_max_ratio(test_points)

            # Accept or reject based on Metropolis criterion
            if test_ratio > best_ratio:
                best_points = test_points.copy()
                best_ratio = test_ratio
            elif np.random.random() < np.exp((test_ratio - best_ratio) / current_temp):
                best_points = test_points.copy()
                best_ratio = test_ratio

            # Occasionally print progress
            if iteration % 10000 == 0:
                elapsed = time.time() - start_time
                #print(f"Iteration {iteration}, Ratio: {best_ratio:.6f}, Temp: {current_temp:.6f}, Time: {elapsed:.2f}s")

        return best_points, best_ratio

    def multi_start_optimization():
        """Run simulated annealing multiple times with different random seeds."""
        best_points = None
        best_ratio = 0.0
        best_seed = 0

        # Try multiple restarts with different seeds
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            points, ratio = simulated_annealing()

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
                best_seed = seed

        return best_points, best_ratio

    # Run optimization
    points, ratio = multi_start_optimization()

    # Final check and normalization
    if points is None:
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