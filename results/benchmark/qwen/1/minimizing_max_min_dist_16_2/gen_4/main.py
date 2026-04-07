# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time

def compute_distances(points):
    """Compute pairwise distances between all points."""
    return squareform(pdist(points))

def calculate_min_max_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    
    distances = compute_distances(points)
    # Set diagonal to large value to ignore self-distances
    np.fill_diagonal(distances, np.inf)
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0.0
    
    return min_dist / max_dist

def initialize_grid_points(n_points=16):
    """Initialize points on a regular grid for better starting configuration."""
    # Create a grid layout that approximates uniform distribution
    n_per_side = int(np.ceil(np.sqrt(n_points)))
    x = np.linspace(0.1, 0.9, n_per_side)
    y = np.linspace(0.1, 0.9, n_per_side)
    
    # Generate grid points
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # If we have more points than grid can hold, add random points
    if len(points) < n_points:
        extra_points = n_points - len(points)
        np.random.seed(42)
        additional = np.random.rand(extra_points, 2) * 0.8 + 0.1
        points = np.vstack([points, additional])
    else:
        points = points[:n_points]
        
    return points

def optimize_points(initial_points):
    """Optimize point configuration using scipy optimization."""
    def objective(x):
        # Reshape flat array back to 2D points
        points = x.reshape(-1, 2)
        ratio = calculate_min_max_ratio(points)
        # We minimize negative ratio since we want to maximize ratio
        return -ratio
    
    # Flatten initial points for scipy
    x0 = initial_points.flatten()
    
    # Use differential evolution for global search first
    result1 = differential_evolution(
        objective, 
        bounds=[(0.0, 1.0)] * len(x0),
        maxiter=50,
        popsize=15,
        seed=42
    )
    
    # Refine with local optimization
    result2 = minimize(
        objective,
        result1.x,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * len(x0),
        options={'maxiter': 100}
    )
    
    optimized_points = result2.x.reshape(-1, 2)
    return optimized_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize with grid-based approach for better starting configuration
    initial_points = initialize_grid_points(16)
    
    # Optimize using evolutionary and local methods
    optimized_points = optimize_points(initial_points)
    
    return optimized_points

# EVOLVE-BLOCK-END
