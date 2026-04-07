# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time

def compute_distance_matrix(points):
    """Compute pairwise distance matrix for given points."""
    return squareform(pdist(points))

def calculate_min_max_ratio(distance_matrix):
    """Calculate the ratio of minimum to maximum distances."""
    # Exclude diagonal (distance to self)
    off_diagonal = distance_matrix[distance_matrix > 0]
    if len(off_diagonal) == 0:
        return 0.0
    d_min = np.min(off_diagonal)
    d_max = np.max(off_diagonal)
    return d_min / d_max if d_max > 0 else 0.0

def initialize_points_lattice():
    """Initialize points using a lattice-based approach."""
    # Create a 4x4 grid layout
    grid_size = 4
    points = np.array([[i/grid_size, j/grid_size] for i in range(grid_size) for j in range(grid_size)])
    return points

def optimize_point_placement(initial_points, max_iterations=1000):
    """Optimize point placement using scipy optimization."""
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        dist_matrix = compute_distance_matrix(points)
        ratio = calculate_min_max_ratio(dist_matrix)
        # Return negative ratio since we want to maximize
        return -ratio

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(len(x0))]

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )

    # Reshape optimized points
    optimized_points = result.x.reshape(-1, 2)
    return optimized_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize with lattice-based configuration for better starting point
    initial_points = initialize_points_lattice()

    # Add some noise to break symmetry and improve optimization
    np.random.seed(42)
    initial_points += np.random.normal(0, 0.01, initial_points.shape)

    # Clip to valid range [0,1]
    initial_points = np.clip(initial_points, 0, 1)

    # Optimize the point placement
    final_points = optimize_point_placement(initial_points)

    return final_points

# EVOLVE-BLOCK-END