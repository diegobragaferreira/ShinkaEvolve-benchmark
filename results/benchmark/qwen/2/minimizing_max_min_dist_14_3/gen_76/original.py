# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x back to points
        points = x.reshape(-1, 3)

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize ratio (minimize negative)
        if max_dist == 0:
            return float('inf')
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points stay within unit sphere (for better conditioning)
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values where constraint is satisfied
        return 1.0 - norms  # Positive when norm <= 1

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

    def evaluate_solution(points):
        """Evaluate the quality of a point configuration."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    best_points = None
    best_ratio = 0.0

    # Multiple restart strategy with different Fibonacci seeds
    seeds = [42, 123, 456, 789, 999]

    for seed in seeds:
        np.random.seed(seed)

        # Start with Fibonacci sphere arrangement
        points = fibonacci_sphere(14)

        # Add controlled noise
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        if np.max(norms) > 0:
            points = points / np.max(norms) * 0.9

        # Run local optimization with adaptive perturbations
        x0 = points.flatten()
        cons = {'type': 'ineq', 'fun': constraint_func}

        try:
            # First run basic optimization
            result = minimize(objective, x0, method='SLSQP',
                             constraints=cons,
                             options={'maxiter': 500, 'ftol': 1e-8},
                             bounds=[(-1, 1)] * 42)

            if result.success:
                final_points = result.x.reshape(-1, 3)
                norms_final = np.linalg.norm(final_points, axis=1)
                if np.max(norms_final) > 1:
                    final_points = final_points / np.max(norms_final) * 0.99
                current_ratio = evaluate_solution(final_points)

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = final_points.copy()

        except Exception as e:
            # Continue to next seed if this one fails
            continue

    # If no successful optimization, fall back to the best Fibonacci initialization
    if best_points is None:
        np.random.seed(42)
        points = fibonacci_sphere(14)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        norms = np.linalg.norm(points, axis=1)
        if np.max(norms) > 0:
            points = points / np.max(norms) * 0.9
        best_points = points

    return best_points


# EVOLVE-BLOCK-END