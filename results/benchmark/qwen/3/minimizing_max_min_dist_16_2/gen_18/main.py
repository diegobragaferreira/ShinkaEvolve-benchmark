# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(points):
        # Compute distance matrix
        distances = squareform(pdist(points))

        # Zero out diagonal
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative ratio (since we want to maximize)
        return -min_dist / max_dist

    def ensure_bounds(points):
        """Ensure all points are within the unit square [0,1] x [0,1]"""
        return np.clip(points, 0, 1)

    def get_min_max_ratio(points):
        """Calculate the min/max distance ratio"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    # Initialize with a better configuration - circle packing inspired layout
    np.random.seed(42)

    # Create a more sophisticated initial pattern inspired by good point distributions
    # Use a combination of circle packing and hexagonal arrangement
    initial_points = []

    # Place points in a pattern that tries to distribute them evenly
    # Start with a few strategic points at corners and center
    corner_points = [[0, 0], [1, 0], [0, 1], [1, 1]]
    center_points = [[0.5, 0.5]]

    # Add some points along edges
    edge_points = [
        [0.25, 0], [0.5, 0], [0.75, 0],
        [0, 0.25], [0, 0.5], [0, 0.75],
        [0.25, 1], [0.5, 1], [0.75, 1],
        [1, 0.25], [1, 0.5], [1, 0.75]
    ]

    # Combine all initial points
    initial_points = corner_points + center_points + edge_points

    # Fill remaining points with random distribution
    remaining_points = 16 - len(initial_points)
    for _ in range(remaining_points):
        initial_points.append([random.random(), random.random()])

    # Convert to numpy array and ensure bounds
    points = np.array(initial_points)
    points = ensure_bounds(points)

    # Simulated Annealing optimization
    current_points = points.copy()
    current_ratio = get_min_max_ratio(current_points)

    # Parameters for simulated annealing
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 1e-8
    max_iterations = 10000
    iteration = 0

    # Store best solution found
    best_points = current_points.copy()
    best_ratio = current_ratio

    # Main optimization loop
    while temperature > min_temperature and iteration < max_iterations:
        # Create new candidate solution by slightly perturbing one point
        candidate_points = current_points.copy()

        # Pick a random point to perturb
        idx = random.randint(0, 15)

        # Perturb that point
        candidate_points[idx, 0] += (random.random() - 0.5) * 0.1 * temperature
        candidate_points[idx, 1] += (random.random() - 0.5) * 0.1 * temperature

        # Ensure the point stays within bounds
        candidate_points[idx] = ensure_bounds(candidate_points[idx].reshape(1, -1)).reshape(-1)

        # Calculate new ratio
        new_ratio = get_min_max_ratio(candidate_points)

        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or random.random() < np.exp((new_ratio - current_ratio) / temperature):
            current_points = candidate_points
            current_ratio = new_ratio

            # Update best solution if needed
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()

        # Cool down
        temperature *= cooling_rate
        iteration += 1

    return best_points


# EVOLVE-BLOCK-END