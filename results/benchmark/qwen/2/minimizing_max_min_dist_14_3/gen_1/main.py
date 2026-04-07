# EVOLVE-BLOCK-START
import numpy as np
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

    # Initialize points on a sphere to get a good starting configuration
    points = initialize_points_sphere(n, d)

    # Optimize using simulated annealing
    best_points, best_ratio = optimize_points_simulated_annealing(points)

    return best_points

def initialize_points_sphere(n, d):
    """Initialize points on a unit sphere for better spread"""
    # Generate points uniformly on sphere using Fibonacci sphere algorithm
    points = np.zeros((n, d))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = [x, y, z]

    return points

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances
    distances = pdist(points)

    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max == 0:
        return 0.0

    return d_min / d_max

def optimize_points_simulated_annealing(initial_points, max_time=300):
    """
    Optimize point positions using simulated annealing to maximize min/max distance ratio
    """
    points = initial_points.copy()
    current_ratio = compute_min_max_ratio(points)

    # Parameters for simulated annealing
    temp = 1.0
    min_temp = 1e-8
    cooling_rate = 0.9995
    max_iter = 100000
    iter_count = 0

    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio

    start_time = time.time()

    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Try small perturbations
        new_points = points.copy()

        # Pick a random point to move
        point_idx = np.random.randint(0, len(points))

        # Make a small random displacement
        displacement = np.random.normal(0, 0.01, 3)
        new_points[point_idx] += displacement

        # Keep within bounds (unit sphere)
        norm = np.linalg.norm(new_points[point_idx])
        if norm > 1:
            new_points[point_idx] /= norm

        # Compute new ratio
        new_ratio = compute_min_max_ratio(new_points)

        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio

            # Update best solution if improved
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio

        # Cool down temperature
        temp *= cooling_rate
        iter_count += 1

    return best_points, best_ratio


# EVOLVE-BLOCK-END