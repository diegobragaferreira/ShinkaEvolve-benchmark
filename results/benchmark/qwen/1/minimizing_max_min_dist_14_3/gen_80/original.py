# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    n = 14
    d = 3

    # Generate initial points using Fibonacci spiral on sphere for good distribution
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    # Additional initialization methods
    def random_points(samples=14):
        return np.random.rand(samples, 3)

    def perturbed_fibonacci(samples=14, sigma=0.05):
        fib_points = fibonacci_sphere(samples)
        noise = np.random.normal(0, sigma, fib_points.shape)
        perturbed = fib_points + noise
        # Normalize to unit sphere again
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return perturbed / norms

    # Multi-start optimization parameters
    num_starts = 10
    perturbation_scales = [0.02, 0.05, 0.1]

    best_ratio = -np.inf
    best_points = None

    # Run multiple optimizations from different starting points
    for start_idx in range(num_starts):
        # Choose initialization method
        if start_idx == 0:
            # First start: Fibonacci spiral
            initial_points = fibonacci_sphere(n)
        elif start_idx <= 3:
            # Next 3 starts: Random points
            initial_points = random_points(n)
        else:
            # Remaining starts: Perturbed Fibonacci with different scales
            scale_idx = (start_idx - 4) % len(perturbation_scales)
            initial_points = perturbed_fibonacci(n, perturbation_scales[scale_idx])

        # Normalize to unit cube [0,1]^3
        # First center around origin and scale appropriately
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        # Then shift to [0,1]^3
        initial_points = initial_points + 0.5

        # Add slight random perturbation to break symmetry
        if start_idx > 0:
            # Use different perturbation scales for diversity
            scale_idx = (start_idx - 1) % len(perturbation_scales)
            perturbation = np.random.normal(0, perturbation_scales[scale_idx], initial_points.shape)
            initial_points += perturbation
            # Clip to stay within bounds
            initial_points = np.clip(initial_points, 0, 1)

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        def objective(x_flat):
            # Reshape back to points
            points = x_flat.reshape((n, d))

            # Calculate pairwise distances
            distances = pdist(points)

            # Compute min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Avoid division by zero
            if d_max == 0:
                return -np.inf

            # Minimize negative of ratio (since we want to maximize ratio)
            ratio = d_min / d_max

            # Return negative because we're minimizing
            return -ratio

        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(n * d)]

        # Perform optimization with more robust settings
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}
        )

        # Extract optimized points
        optimized_points = result.x.reshape((n, d))

        # Ensure all points are within [0,1]^3
        optimized_points = np.clip(optimized_points, 0, 1)

        # Calculate the actual ratio for this optimization run
        final_distances = pdist(optimized_points)
        if len(final_distances) > 0:
            d_min = np.min(final_distances)
            d_max = np.max(final_distances)
            if d_max > 0:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

    # If we didn't find a good solution, return the initial points
    if best_points is None:
        initial_points = fibonacci_sphere(n)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        initial_points = initial_points + 0.5
        return initial_points

    return best_points

# EVOLVE-BLOCK-END