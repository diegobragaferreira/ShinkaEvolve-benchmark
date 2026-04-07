# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc
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

    # Generate Sobol sequence points for better space-filling properties
    def sobol_points(samples=14, seed=42):
        # Use QMC Sobol sequence for better space filling
        sampler = qmc.Sobol(d=d, seed=seed)
        points = sampler.random(samples)
        # Scale to [-1, 1]^3 then normalize to unit sphere
        points = points * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    # Generate improved Sobol points with better distribution
    def enhanced_sobol_points(samples=14, seed=42):
        """Enhanced Sobol points with better sphere distribution"""
        # Generate Sobol points in [0,1]^3
        sampler = qmc.Sobol(d=d, seed=seed)
        points = sampler.random(samples)
        # Transform to [-1, 1]^3
        points = points * 2 - 1
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

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

    def jittered_grid(samples=14):
        # Create a simple grid-based initialization
        # For 14 points, use a 2x2x4 grid pattern with jitter
        grid_x = np.linspace(0, 1, 2)
        grid_y = np.linspace(0, 1, 2)
        grid_z = np.linspace(0, 1, 4)

        points = []
        for x in grid_x:
            for y in grid_y:
                for z in grid_z:
                    if len(points) < samples:
                        points.append([x, y, z])

        # Add small random jitter
        points = np.array(points)
        if len(points) < samples:
            extra_points = samples - len(points)
            extra = np.random.rand(extra_points, 3)
            points = np.vstack([points, extra])
        else:
            points = points[:samples]

        # Add jitter
        jitter = np.random.normal(0, 0.02, points.shape)
        points = points + jitter
        points = np.clip(points, 0, 1)
        return points

    # Multi-start optimization parameters
    num_starts = 30  # Increased number of starts for better exploration
    perturbation_scales = [0.005, 0.01, 0.02, 0.05, 0.1]  # Added smaller scale for fine tuning

    best_ratio = -np.inf
    best_points = None

    # Run multiple optimizations from different starting points
    for start_idx in range(num_starts):
        # Choose initialization method with more diversity
        if start_idx == 0:
            # First start: Fibonacci spiral
            initial_points = fibonacci_sphere(n)
        elif start_idx == 1:
            # Second start: Standard Sobol sequence
            initial_points = sobol_points(n)
        elif start_idx == 2:
            # Third start: Enhanced Sobol points
            initial_points = enhanced_sobol_points(n)
        elif start_idx <= 6:
            # Next 4 starts: Perturbed Sobol with different scales
            scale_idx = (start_idx - 3) % len(perturbation_scales)
            sobol_init = sobol_points(n, seed=42 + start_idx)
            noise = np.random.normal(0, perturbation_scales[scale_idx], sobol_init.shape)
            initial_points = sobol_init + noise
            # Normalize to unit sphere
            norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            initial_points = initial_points / norms
        elif start_idx <= 10:
            # Next 4 starts: Random points
            initial_points = random_points(n)
        elif start_idx <= 17:
            # Next 7 starts: Perturbed Fibonacci with different scales
            scale_idx = (start_idx - 11) % len(perturbation_scales)
            initial_points = perturbed_fibonacci(n, perturbation_scales[scale_idx])
        else:
            # Last 8 starts: Jittered grid
            initial_points = jittered_grid(n)

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

            # Calculate pairwise distances using cdist for better performance
            distances = cdist(points, points)

            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distances, np.inf)

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
        # Use multiple refinement approaches with different strategies
        refinement_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
        best_local_ratio = -np.inf
        best_local_points = None

        # Try multiple optimization approaches
        for method in refinement_methods:
            try:
                result = minimize(
                    objective,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}
                )

                # Extract optimized points
                optimized_points = result.x.reshape((n, d))

                # Ensure all points are within [0,1]^3
                optimized_points = np.clip(optimized_points, 0, 1)

                # Calculate the actual ratio for this optimization run
                final_distances = cdist(optimized_points, optimized_points)
                np.fill_diagonal(final_distances, np.inf)
                if len(final_distances[final_distances != np.inf]) > 0:
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_local_ratio:
                            best_local_ratio = ratio
                            best_local_points = optimized_points.copy()
            except:
                continue

        # If all methods fail, try one more time with a different approach
        if best_local_points is None:
            try:
                # Try with a different tolerance
                result = minimize(
                    objective,
                    initial_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
                )

                optimized_points = result.x.reshape((n, d))
                optimized_points = np.clip(optimized_points, 0, 1)

                final_distances = cdist(optimized_points, optimized_points)
                np.fill_diagonal(final_distances, np.inf)
                if len(final_distances[final_distances != np.inf]) > 0:
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_local_ratio:
                            best_local_ratio = ratio
                            best_local_points = optimized_points.copy()
            except:
                pass

        # Update global best if we found a better solution
        if best_local_points is not None and best_local_ratio > best_ratio:
            best_ratio = best_local_ratio
            best_points = best_local_points.copy()

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