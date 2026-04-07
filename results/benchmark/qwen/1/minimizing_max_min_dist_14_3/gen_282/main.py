# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral method"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)


def sobol_sphere(n):
    """Generate n points on a sphere using Sobol sequence for optimal space-filling"""
    try:
        # Use scipy's quasi-Monte Carlo generators for true Sobol sequence
        from scipy.stats.qmc import Sobol
        # Generate Sobol sequence in 3D
        sobol_gen = Sobol(d=3, scramble=True, seed=42)
        samples = sobol_gen.random(n)

        # Transform to sphere using inverse transform sampling
        points = []
        for i in range(n):
            # Convert uniform [0,1]^3 to points on unit sphere
            u1, u2, u3 = samples[i]

            # Map to sphere using inverse transform
            theta = 2 * np.pi * u1  # azimuthal angle
            phi = np.arccos(2 * u2 - 1)  # polar angle

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    except ImportError:
        # Fallback to improved Fibonacci-based approach if Sobol not available
        points = []
        phi = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            # Base Fibonacci point
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            # Apply more sophisticated perturbations
            # Use multiple low-discrepancy sequences to create better distribution
            perturbation_factor = 0.025

            # Multiple quasi-random components for better spread
            seq1 = (i * 1.618033988749895) % 1.0  # golden ratio
            seq2 = (i * 0.7853981633974483) % 1.0  # pi/4
            seq3 = (i * 2.3561944901923448) % 1.0  # 3pi/4

            # Apply different perturbation strengths
            x_pert = perturbation_factor * (seq1 - 0.5)
            y_pert = perturbation_factor * (seq2 - 0.5)
            z_pert = perturbation_factor * (seq3 - 0.5)

            x += x_pert
            y += y_pert
            z += z_pert

            # Normalize to unit sphere
            norm = np.sqrt(x*x + y*y + z*z)
            if norm > 0:
                x /= norm
                y /= norm
                z /= norm

            points.append([x, y, z])

        return np.array(points)


def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance between all point pairs"""
    distances = pdist(points)
    return np.min(distances) / np.max(distances)


def optimize_points(initial_points, max_iter=1000):
    """Optimize point positions to maximize min/max distance ratio"""

    def objective(x):
        # Reshape flat array back to points
        points = x.reshape(-1, 3)
        # We want to maximize the ratio, so we minimize its negative
        return -min_max_dist_ratio(points)

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Define bounds (points should stay within [-1,1] for sphere)
    bounds = [(-1, 1) for _ in range(len(x0))]

    # Use L-BFGS-B for optimization with bounds
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter},
        tol=1e-8
    )

    # Return optimized points
    return result.x.reshape(-1, 3)


def optimize_with_multi_start(initial_points, num_starts=5, max_iter=1000):
    """Run optimization from multiple starting points to find better solution"""
    best_ratio = -np.inf
    best_points = initial_points.copy()

    for i in range(num_starts):
        # Create perturbed version of initial points
        if i == 0:
            # First start uses original points (no perturbation)
            perturbed_points = initial_points.copy()
        else:
            # Other starts use perturbed points
            perturbed_points = initial_points + np.random.normal(0, 0.05, initial_points.shape)

        # Optimize from this starting point
        optimized_points = optimize_points(perturbed_points, max_iter)

        # Calculate ratio for this optimization run
        ratio = min_max_dist_ratio(optimized_points)

        # Keep track of best solution
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3

    # Try multiple initialization strategies and select the best one
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Fibonacci initialization (traditional approach)
    np.random.seed(42)
    fib_points = fibonacci_sphere(n)
    optimized_fib = optimize_with_multi_start(fib_points, num_starts=3, max_iter=500)
    ratio_fib = min_max_dist_ratio(optimized_fib)

    if ratio_fib > best_ratio:
        best_ratio = ratio_fib
        best_points = optimized_fib.copy()

    # Strategy 2: Sobol-inspired initialization (improved space-filling)
    np.random.seed(123)
    sobol_points = sobol_sphere(n)
    optimized_sobol = optimize_with_multi_start(sobol_points, num_starts=3, max_iter=500)
    ratio_sobol = min_max_dist_ratio(optimized_sobol)

    if ratio_sobol > best_ratio:
        best_ratio = ratio_sobol
        best_points = optimized_sobol.copy()

    # Strategy 3: Random initialization for diversity
    np.random.seed(456)
    random_points = np.random.randn(n, 3)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    random_points = random_points / norms
    optimized_random = optimize_with_multi_start(random_points, num_starts=3, max_iter=500)
    ratio_random = min_max_dist_ratio(optimized_random)

    if ratio_random > best_ratio:
        best_ratio = ratio_random
        best_points = optimized_random.copy()

    # Return the best configuration found
    if best_points is None:
        # Fallback to Fibonacci if everything failed
        np.random.seed(42)
        return fibonacci_sphere(n)

    return best_points


# EVOLVE-BLOCK-END