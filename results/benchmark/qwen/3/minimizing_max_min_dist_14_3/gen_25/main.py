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

    def calculate_ratio(points):
        """Calculate min/max distance ratio"""
        if len(points) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = pdist(points)

        if len(distances) == 0:
            return 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Handle edge case where all points are identical
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(n_points):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def objective(x):
        """Objective function for optimization (minimize negative ratio)"""
        points = x.reshape(-1, 3)

        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)

        distances = pdist(points)

        if len(distances) == 0:
            return 1.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Handle edge case where all points are identical or very close
        if d_max == 0:
            return 1.0

        # Maximize min/max ratio (so minimize negative ratio)
        return -d_min / d_max

    def get_better_initialization():
        """Get a better initial configuration using known good arrangements"""
        np.random.seed(42)

        # Start with Fibonacci spiral on sphere
        initial_points = fibonacci_sphere(14)

        # Add some randomness to break symmetries
        noise_scale = 0.05
        noise = np.random.normal(0, noise_scale, initial_points.shape)
        initial_points += noise

        # Normalize to unit sphere
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        initial_points = initial_points / norms

        # Scale to fit within unit cube [0,1]^3
        # Center around origin first
        initial_points = initial_points - np.mean(initial_points, axis=0)
        # Scale to fit nicely in [0,1]^3
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / (2 * max_coord) + 0.5

        return initial_points

    # Get better initial configuration
    initial_points = get_better_initialization()

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define bounds for optimization (coordinates in [0,1])
    bounds = [(0, 1) for _ in range(42)]

    # Set up time limit
    start_time = time.time()
    time_limit = 350  # seconds

    # Run optimization with time limit
    try:
        # First run with moderate parameters
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-6,
            tol=1e-6,
            callback=lambda x, convergence: time.time() - start_time > time_limit,
            disp=False,
            polish=True  # Use local optimization for better results
        )

        optimized_points = result.x.reshape(-1, 3)
        optimized_points = np.clip(optimized_points, 0, 1)

        # Apply additional local refinement if time allows
        if time.time() - start_time < time_limit - 5:
            # Second optimization pass with different parameters
            try:
                result_fine = differential_evolution(
                    objective,
                    bounds,
                    seed=42,
                    maxiter=50,
                    popsize=10,
                    mutation=(0.8, 1),
                    recombination=0.9,
                    atol=1e-8,
                    tol=1e-8,
                    disp=False,
                    polish=True
                )

                fine_points = result_fine.x.reshape(-1, 3)
                fine_points = np.clip(fine_points, 0, 1)

                # Compare and keep better solution
                current_ratio = calculate_ratio(optimized_points)
                fine_ratio = calculate_ratio(fine_points)

                if fine_ratio > current_ratio:
                    optimized_points = fine_points

            except:
                pass

    except Exception as e:
        # Fallback to initial points if optimization fails
        print(f"Optimization failed with error: {e}")
        optimized_points = initial_points

    # Ensure final result is within bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points

# EVOLVE-BLOCK-END