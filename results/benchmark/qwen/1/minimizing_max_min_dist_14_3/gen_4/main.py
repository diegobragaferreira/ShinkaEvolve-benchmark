# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses simulated annealing optimization to find a good configuration.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Initialize points randomly within a unit cube [0,1]^3
    points = np.random.rand(n, d)

    # Simulated Annealing parameters
    max_iter = 100000
    initial_temp = 1.0
    cooling_rate = 0.9995
    min_temp = 1e-8

    # Track best solution
    best_points = points.copy()
    best_ratio = 0.0

    # Calculate initial distances
    def calculate_ratio(points):
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid considering same points
        np.fill_diagonal(distances, np.inf)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 0:
            return d_min / d_max
        else:
            return 0.0

    current_ratio = calculate_ratio(points)
    temperature = initial_temp

    # Optimization loop
    start_time = time.time()
    for iteration in range(max_iter):
        # Make small random perturbation
        new_points = points.copy()
        # Perturb one point at a time
        idx = random.randint(0, n-1)
        new_points[idx] += np.random.normal(0, temperature * 0.01, d)
        # Keep points in [0,1]^3 bounds
        new_points[idx] = np.clip(new_points[idx], 0, 1)

        # Calculate new ratio
        new_ratio = calculate_ratio(new_points)

        # Accept or reject the move
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature
            delta = new_ratio - current_ratio
            if random.random() < np.exp(delta / temperature):
                points = new_points
                current_ratio = new_ratio

        # Cool down
        temperature *= cooling_rate

        # Check for early termination
        if temperature < min_temp:
            break

        # Check time limit
        if time.time() - start_time > 350:  # Leave some buffer for cleanup
            break

    # Return best solution found
    return best_points


# EVOLVE-BLOCK-END