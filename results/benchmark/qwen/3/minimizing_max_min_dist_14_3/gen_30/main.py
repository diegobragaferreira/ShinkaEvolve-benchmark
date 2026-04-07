# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import warnings
import itertools
from numba import jit

@jit(nopython=True)
def compute_distances_numba(points):
    """Faster distance computation using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape to points
        points = x.reshape(-1, 3)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Zero out diagonal
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (min/max)
        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    def objective_numba(x):
        # Reshape to points
        points = x.reshape(-1, 3)
        distances = compute_distances_numba(points)
        np.fill_diagonal(distances, np.inf)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    def normalize_points(points):
        """Normalize points to unit sphere"""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def fibonacci_sphere_sample(n):
        """Generate points on sphere using Fibonacci method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = np.arccos(y)  # angle from z-axis
            phi = (i * 2 * np.pi) / golden_ratio  # azimuthal angle

            x = radius * np.cos(phi)
            z = radius * np.sin(phi)

            points.append([x, y, z])
        return np.array(points)

    def generate_initial_config():
        """Generate multiple good initial configurations and pick best"""
        configs = []
        best_ratio = -np.inf

        # Method 1: Fibonacci sphere sampling
        points1 = fibonacci_sphere_sample(14)
        points1 = normalize_points(points1)
        configs.append(points1)

        # Method 2: Random points on sphere
        np.random.seed(42)
        points2 = np.random.randn(14, 3)
        points2 = normalize_points(points2)
        configs.append(points2)

        # Method 3: Perturbed Fibonacci points
        points3 = points1.copy()
        np.random.seed(123)
        points3 += np.random.normal(0, 0.05, points3.shape)
        points3 = normalize_points(points3)
        configs.append(points3)

        # Method 4: Another random configuration
        np.random.seed(456)
        points4 = np.random.rand(14, 3) * 2 - 1
        points4 = normalize_points(points4)
        configs.append(points4)

        # Method 5: Slightly different Fibonacci distribution
        np.random.seed(789)
        points5 = fibonacci_sphere_sample(14)
        points5 += np.random.normal(0, 0.02, points5.shape)
        points5 = normalize_points(points5)
        configs.append(points5)

        # Evaluate all initial configs
        best_config = None
        for config in configs:
            # Flatten for objective calculation
            flat_config = config.flatten()
            try:
                ratio = -objective(flat_config)  # Negate since objective returns negative
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_config = config.copy()
            except:
                continue

        return best_config if best_config is not None else configs[0]

    # Generate initial configuration
    points = generate_initial_config()

    # Flatten for optimization
    x0 = points.flatten()

    # Use differential evolution as primary optimization method
    # This is more robust for escaping local optima
    bounds = [(-1, 1) for _ in range(42)]  # 14 points * 3 coordinates each

    # First try differential evolution for global optimization
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            tol=1e-12
        )

        if de_result.success:
            # Refine with local optimization around DE result
            refined_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if refined_result.success:
                optimized_points = refined_result.x.reshape(-1, 3)
                optimized_points = normalize_points(optimized_points)
                final_points = optimized_points
            else:
                final_points = points
        else:
            final_points = points

    except Exception as e:
        # Fallback to local optimization only
        try:
            local_result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if local_result.success:
                optimized_points = local_result.x.reshape(-1, 3)
                optimized_points = normalize_points(optimized_points)
                final_points = optimized_points
            else:
                final_points = points
        except Exception as e2:
            final_points = points

    # Final check and cleanup
    final_points = normalize_points(final_points)

    return final_points

# EVOLVE-BLOCK-END