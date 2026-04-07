# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-start optimization approach to find better solutions.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective_quadratic(x):
        """
        Quadratic objective that approximates the ratio maximization by focusing on
        minimizing squared distances while maintaining reasonable spread.
        """
        points = x.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return np.inf

        # We want to maximize min_dist/max_dist, which is equivalent to minimizing max_dist/min_dist
        # Using a quadratic approximation that penalizes both small min_dist and large max_dist
        return -(min_dist / max_dist)

    def create_initial_grid():
        """Create initial 4x4 grid points"""
        grid_size = 4
        x_vals = np.linspace(0.05, 0.95, grid_size)
        y_vals = np.linspace(0.05, 0.95, grid_size)
        return np.array([[x, y] for x in x_vals for y in y_vals])

    def create_spiral_pattern():
        """Create a spiral-like initial pattern"""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])
        # Add small perturbations
        noise = np.random.normal(0, 0.02, spiral_points.shape)
        spiral_points = spiral_points + noise
        return np.clip(spiral_points, 0, 1)

    def create_hexagonal_pattern():
        """Create a hexagonal pattern"""
        np.random.seed(42)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add small random perturbation
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                points.append([x, y])
        return np.array(points)

    def create_random_initial():
        """Create random initial points"""
        np.random.seed(42)
        return np.random.rand(16, 2)

    def compute_distance_ratios(points):
        """
        Helper function to compute the ratio metrics for evaluation.
        """
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 0, 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0, 0, 0
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def optimize_from_initial_points(initial_points, max_iter=2000):
        """
        Run optimization from given initial points using multiple methods.
        """
        x0 = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        best_result = None
        best_ratio = 0

        # Try multiple optimization approaches
        methods = ['L-BFGS-B', 'SLSQP']
        for method in methods:
            try:
                result = minimize(
                    objective_quadratic,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio, _, _ = compute_distance_ratios(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = final_points

            except Exception as e:
                continue

        # If no successful optimization, return original points
        if best_result is None:
            return initial_points
        return best_result

    def multi_start_optimization():
        """
        Perform multi-start optimization with multiple initialization strategies.
        """
        initial_strategies = [
            ('grid', create_initial_grid),
            ('spiral', create_spiral_pattern),
            ('hexagonal', create_hexagonal_pattern),
            ('random', create_random_initial)
        ]

        best_points = None
        best_ratio = 0

        # Try each initialization strategy multiple times with different random seeds
        for strategy_name, strategy_func in initial_strategies:
            for seed_offset in range(3):  # Try 3 different seeds per strategy
                np.random.seed(42 + seed_offset)
                try:
                    initial_points = strategy_func()
                    optimized_points = optimize_from_initial_points(initial_points, max_iter=1500)
                    ratio, _, _ = compute_distance_ratios(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

                except Exception as e:
                    continue

        # If we still don't have a good solution, try a few more refinements
        if best_points is None:
            # Fallback to a simple grid approach
            points = create_initial_grid()
            best_points = optimize_from_initial_points(points)

        return best_points

    # Main execution flow
    try:
        # Run multi-start optimization
        final_points = multi_start_optimization()

        # Apply final refinement if needed
        if final_points is not None:
            ratio, _, _ = compute_distance_ratios(final_points)
            if ratio < 0.25:  # If ratio is still relatively low, do additional refinement
                # Try optimizing from a random start
                np.random.seed(42)
                random_points = np.random.rand(16, 2)
                refined_points = optimize_from_initial_points(random_points, max_iter=1000)
                refined_ratio, _, _ = compute_distance_ratios(refined_points)

                if refined_ratio > ratio:
                    final_points = refined_points

        return final_points if final_points is not None else create_initial_grid()

    except Exception as e:
        # Fallback to simplest approach if everything else fails
        points = create_initial_grid()
        return points

# EVOLVE-BLOCK-END