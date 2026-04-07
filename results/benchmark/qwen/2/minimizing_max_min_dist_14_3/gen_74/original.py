# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')
import time

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

    def project_to_unit_sphere(points):
        """Project points to unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

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

    def two_phase_optimization(initial_points):
        """Perform two-phase optimization: global search then local refinement."""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)

        # Phase 1: Global search with modified optimization
        current_points = initial_points.copy()

        # Run multiple iterations of optimization with different strategies
        for iter_num in range(5):
            x0 = current_points.flatten()
            cons = {'type': 'ineq', 'fun': constraint_func}

            try:
                # Try with different optimization settings
                result = minimize(objective, x0, method='SLSQP',
                                 constraints=cons,
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
            if iter_num < 4:  # Don't perturb on the last iteration
                current_points = best_points.copy()
                # Perturb a few points
                for _ in range(3):
                    idx = np.random.randint(0, len(current_points))
                    delta = np.random.normal(0, 0.005, 3)
                    current_points[idx] += delta
                    current_points[idx] = project_to_unit_sphere(current_points[idx].reshape(1, 3)).flatten()

        # Phase 2: Local refinement with detailed optimization
        try:
            x0 = best_points.flatten()
            cons = {'type': 'ineq', 'fun': constraint_func}

            result = minimize(objective, x0, method='SLSQP',
                             constraints=cons,
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

    # Multiple restart strategy with different initialization methods
    best_points = None
    best_ratio = 0.0

    # Strategy 1: Fibonacci sphere initialization
    fib_points = fibonacci_sphere(14)
    fib_points = project_to_unit_sphere(fib_points)
    # Add small noise
    fib_points += np.random.normal(0, 0.01, fib_points.shape)
    fib_points = project_to_unit_sphere(fib_points)

    points, ratio = two_phase_optimization(fib_points)
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = points.copy()

    # Strategy 2: Random initialization with normalization
    np.random.seed(123)
    rand_points = np.random.uniform(-1, 1, (14, 3))
    rand_points = project_to_unit_sphere(rand_points)
    points, ratio = two_phase_optimization(rand_points)
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = points.copy()

    # Strategy 3: Perturbed Fibonacci sphere
    np.random.seed(456)
    pert_fib_points = fib_points.copy()
    pert_fib_points += np.random.normal(0, 0.02, pert_fib_points.shape)
    pert_fib_points = project_to_unit_sphere(pert_fib_points)
    points, ratio = two_phase_optimization(pert_fib_points)
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = points.copy()

    # Final fallback to best of all attempts
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
        points = project_to_unit_sphere(points)
        best_points = points

    return best_points


# EVOLVE-BLOCK-END