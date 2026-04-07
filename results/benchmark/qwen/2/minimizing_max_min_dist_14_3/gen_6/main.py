# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        # Calculate pairwise distances
        distances = pdist(points)
        # Return the ratio of min to max distances
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist

    def initialize_points_good_seed(n):
        """Initialize points using a known good geometric configuration for 14 points.
        Using a truncated octahedron-like structure but adapted for 14 points.
        """
        # Precomputed good starting configuration for 14 points
        # These points are designed to have relatively uniform distribution
        # and reasonable minimum/maximum distance ratios
        points = np.array([
            [0.000000, 0.000000, 1.000000],
            [0.000000, 0.000000, -1.000000],
            [0.707107, 0.707107, 0.000000],
            [-0.707107, -0.707107, 0.000000],
            [0.707107, -0.707107, 0.000000],
            [-0.707107, 0.707107, 0.000000],
            [0.500000, 0.500000, 0.500000],
            [-0.500000, -0.500000, -0.500000],
            [0.500000, -0.500000, 0.500000],
            [-0.500000, 0.500000, -0.500000],
            [0.500000, 0.500000, -0.500000],
            [-0.500000, -0.500000, 0.500000],
            [0.500000, -0.500000, -0.500000],
            [-0.500000, 0.500000, 0.500000]
        ])

        # Normalize to unit sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] /= norm

        return points

    def optimize_with_gradient_descent(initial_points, max_iterations=5000):
        """Optimize using a combination of gradient-based and local search approaches."""

        def objective_func(x_flat):
            """Objective function to minimize (negative ratio since we want to maximize)"""
            points = x_flat.reshape(-1, 3)
            ratio = calculate_min_max_ratio(points)
            # We want to maximize ratio, so return negative for minimization
            return -ratio

        # Flatten initial points for optimization
        x0 = initial_points.flatten()

        # Use L-BFGS-B with bounds to ensure constraints (unit cube)
        bounds = [(0, 1)] * len(x0)

        try:
            # Try gradient-based optimization first
            result = minimize(
                objective_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations//2},
                tol=1e-8
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                return optimized_points, -result.fun
        except:
            pass

        # Fall back to simpler local search if needed
        current_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        best_points = current_points.copy()

        # Local search with small perturbations
        for iteration in range(max_iterations):
            # Copy current points
            neighbor_points = current_points.copy()

            # Choose a random point to perturb
            point_idx = np.random.randint(len(neighbor_points))

            # Make a small random perturbation
            perturbation = np.random.normal(0, 0.001, 3)
            neighbor_points[point_idx] += perturbation

            # Clip to [0,1]^3 to maintain constraints
            neighbor_points = np.clip(neighbor_points, 0, 1)

            # Calculate new ratio
            new_ratio = calculate_min_max_ratio(neighbor_points)

            # Accept if better
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = neighbor_points.copy()
                current_points = neighbor_points.copy()
            else:
                # Occasionally accept worse solutions to escape local minima
                if np.random.rand() < 0.01:
                    current_points = neighbor_points.copy()

        return best_points, best_ratio

    def perturb_and_refine(points, iterations=1000):
        """Apply multiple rounds of perturbation and refinement."""
        current_points = points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        best_points = current_points.copy()

        for iter_num in range(iterations):
            # Create neighbor by perturbing multiple points
            neighbor_points = current_points.copy()

            # Perturb a few points at once (more aggressive than single point)
            num_perturbed = max(1, len(neighbor_points) // 10)  # Perturb ~10% of points
            indices_to_perturb = np.random.choice(len(neighbor_points), num_perturbed, replace=False)

            for idx in indices_to_perturb:
                # Use larger perturbation for exploration
                perturbation = np.random.normal(0, 0.01, 3)
                neighbor_points[idx] += perturbation

            # Clip to [0,1]^3
            neighbor_points = np.clip(neighbor_points, 0, 1)

            # Calculate new ratio
            new_ratio = calculate_min_max_ratio(neighbor_points)

            # Accept if better or occasionally if worse (simulated annealing)
            if new_ratio > best_ratio or (np.random.rand() < 0.05 and new_ratio > 0):
                current_points = neighbor_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = neighbor_points.copy()

        return best_points, best_ratio

    # Initialize with a good starting configuration
    np.random.seed(42)
    initial_points = initialize_points_good_seed(14)

    # First optimization round with gradient descent
    optimized_points, ratio1 = optimize_with_gradient_descent(initial_points, 2000)

    # Second round with local search
    refined_points, ratio2 = perturb_and_refine(optimized_points, 1000)

    # Final optimization with more aggressive local search
    final_points, final_ratio = perturb_and_refine(refined_points, 2000)

    return final_points


# EVOLVE-BLOCK-END