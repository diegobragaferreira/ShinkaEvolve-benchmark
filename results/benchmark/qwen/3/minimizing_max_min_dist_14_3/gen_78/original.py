# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral (good initial distribution)"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def hybrid_optimization(initial_points, max_iter=200):
        """Combine global and local optimization strategies"""
        points = initial_points.copy()

        # First, try global optimization with differential evolution
        def objective_function(points_flat):
            points = points_flat.reshape(-1, 3)
            # Ensure points are in valid range [0,1]
            points = np.clip(points, 0, 1)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative because we minimize

        # Define bounds for differential evolution
        bounds = [(0, 1)] * (14 * 3)

        # Global optimization with differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                disp=False,
                polish=True
            )
            if result.success:
                points = result.x.reshape(-1, 3)
                points = np.clip(points, 0, 1)
        except:
            pass

        # Local refinement with L-BFGS-B
        def obj_func(x_flat):
            points = x_flat.reshape(-1, 3)
            points = np.clip(points, 0, 1)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative because we minimize

        try:
            x0 = points.flatten()
            result = minimize(obj_func, x0, method='L-BFGS-B', bounds=[(0, 1)] * (14 * 3),
                            options={'maxiter': 100, 'disp': False})
            if result.success:
                points = result.x.reshape(-1, 3)
                points = np.clip(points, 0, 1)
        except:
            pass

        return points

    def strategic_initialization():
        """Generate strategically placed initial points"""
        # Start with Fibonacci sphere distribution
        fib_points = fibonacci_sphere(14)

        # Slightly perturb to avoid symmetries
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.05, fib_points.shape)
        points = fib_points + perturbation

        # Normalize and scale to [0,1] cube
        # Find bounding box
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)

        # Normalize to [0,1]
        if max_vals[0] != min_vals[0]:
            points[:, 0] = (points[:, 0] - min_vals[0]) / (max_vals[0] - min_vals[0])
        if max_vals[1] != min_vals[1]:
            points[:, 1] = (points[:, 1] - min_vals[1]) / (max_vals[1] - min_vals[1])
        if max_vals[2] != min_vals[2]:
            points[:, 2] = (points[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2])

        # Scale to [0,1] range (in case normalization didn't fully work)
        points = np.clip(points, 0, 1)

        return points

    # Multiple restart strategy with better initialization
    best_solution = None
    best_ratio = 0.0

    # Try multiple restart configurations with different strategies
    for restart in range(20):  # More restarts for better exploration
        # Different initialization strategies
        if restart == 0:
            # High-quality Fibonacci-based initialization
            points = strategic_initialization()
        elif restart < 5:
            # Random initialization within bounds
            points = np.random.rand(14, 3)
        elif restart < 10:
            # Slightly perturbed Fibonacci
            fib_points = fibonacci_sphere(14)
            np.random.seed(42 + restart)
            perturbation = np.random.normal(0, 0.03, fib_points.shape)
            points = fib_points + perturbation
            # Normalize to [0,1]
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            if max_vals[0] != min_vals[0]:
                points[:, 0] = (points[:, 0] - min_vals[0]) / (max_vals[0] - min_vals[0])
            if max_vals[1] != min_vals[1]:
                points[:, 1] = (points[:, 1] - min_vals[1]) / (max_vals[1] - min_vals[1])
            if max_vals[2] != min_vals[2]:
                points[:, 2] = (points[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2])
            points = np.clip(points, 0, 1)
        else:
            # Another random approach
            points = np.random.rand(14, 3) * 0.8 + 0.1  # Centered distribution

        # Hybrid optimization
        optimized_points = hybrid_optimization(points, max_iter=200)
        ratio = compute_min_max_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Final refinement if we have a good solution
    if best_solution is not None:
        # Try one more round of focused optimization
        final_points = hybrid_optimization(best_solution, max_iter=100)
        return final_points

    # Fallback to strategic initialization if nothing worked
    return strategic_initialization()

# EVOLVE-BLOCK-END