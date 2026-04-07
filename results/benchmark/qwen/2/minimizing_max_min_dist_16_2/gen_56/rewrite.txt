# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def compute_min_max_ratio(points):
    """Compute the minimum to maximum distance ratio for given points."""
    if len(points) < 2:
        return 0.0
    
    # Compute pairwise distances
    distances = cdist(points, points)
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Find min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Avoid division by zero
    if max_dist == 0:
        return 0.0
        
    return min_dist / max_dist

def initialize_points(n_points=16):
    """Initialize points using a refined grid-based approach."""
    np.random.seed(42)
    
    # Create a 4x4 grid pattern
    points = []
    rows, cols = 4, 4
    
    for i in range(rows):
        for j in range(cols):
            # Evenly spaced grid points
            x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
            y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5
            
            # Apply adaptive perturbation based on position type
            if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                # Corner points - smallest perturbation
                perturbation = 0.005
            elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                # Edge points - medium perturbation  
                perturbation = 0.01
            else:
                # Interior points - larger perturbation
                perturbation = 0.02
                
            x += np.random.normal(0, perturbation)
            y += np.random.normal(0, perturbation)
            points.append([x, y])
    
    # Ensure points are within bounds [0,1]
    points = np.clip(points, 0, 1)
    return np.array(points[:n_points])

def optimize_points(initial_points, max_time=150):
    """Optimize point distribution using a custom adaptive perturbation method."""
    start_time = time.time()
    
    # Start with initial configuration
    current_points = initial_points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio(current_points)
    
    # Adaptive parameters
    max_iterations = 5000
    current_iteration = 0
    temperature = 1.0
    cooling_rate = 0.9995
    
    # Perform adaptive perturbation
    while current_iteration < max_iterations and (time.time() - start_time) < max_time - 1:
        # Create a new candidate by perturbing one point at a time
        candidate_points = current_points.copy()
        
        # Select a random point to perturb
        point_idx = np.random.randint(0, len(candidate_points))
        
        # Determine perturbation magnitude based on iteration
        if current_iteration < 1000:
            # Early phase: aggressive perturbation
            pert_magnitude = 0.02
        elif current_iteration < 3000:
            # Middle phase: moderate perturbation
            pert_magnitude = 0.01
        else:
            # Late phase: fine tuning
            pert_magnitude = 0.005
            
        # Apply perturbation to selected point
        candidate_points[point_idx, 0] += np.random.normal(0, pert_magnitude)
        candidate_points[point_idx, 1] += np.random.normal(0, pert_magnitude)
        
        # Keep within bounds
        candidate_points[point_idx, 0] = np.clip(candidate_points[point_idx, 0], 0, 1)
        candidate_points[point_idx, 1] = np.clip(candidate_points[point_idx, 1], 0, 1)
        
        # Evaluate candidate
        candidate_ratio = compute_min_max_ratio(candidate_points)
        
        # Accept or reject based on Metropolis criterion
        if candidate_ratio > best_ratio:
            # Always accept improvement
            current_points = candidate_points.copy()
            best_points = candidate_points.copy()
            best_ratio = candidate_ratio
        elif np.random.random() < np.exp((candidate_ratio - best_ratio) / temperature):
            # Accept worse solution with probability
            current_points = candidate_points.copy()
            
        # Cool down temperature
        temperature *= cooling_rate
        
        current_iteration += 1
        
        # Early stopping if we're getting close to a good solution
        if best_ratio > 0.3:
            break
            
    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize points with refined grid pattern
    initial_points = initialize_points(n_points=16)
    
    # Optimize the point distribution
    optimized_points = optimize_points(initial_points, max_time=150)
    
    return optimized_points

# EVOLVE-BLOCK-END