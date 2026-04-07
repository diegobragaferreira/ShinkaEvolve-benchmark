# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles, x, y, r, tree=None, max_radius=None):
    """Check if placing a circle at (x,y) with radius r is valid."""
    # Check boundary constraints
    if r > x or r > y or r > (1-x) or r > (1-y):
        return False
    
    # If we have a KDTree, use it for efficient overlap detection
    if tree is not None and max_radius is not None:
        # Find potential overlapping circles within distance 2*(r+max_radius)
        candidates = tree.query_ball_point([x, y], 2*(r + max_radius))
        for i in candidates:
            cx, cy, cr = circles[i]
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (r + cr):
                return False
    else:
        # Fallback to original method if no tree provided
        for i in range(len(circles)):
            cx, cy, cr = circles[i]
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (r + cr):
                return False

    return True

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k nearest neighbors."""
    if len(circles) < 2:
        return 0

    # Convert circles to array for KDTree query
    circle_points = np.array([[c[0], c[1]] for c in circles])
    tree = cKDTree(circle_points)

    # Query k nearest neighbors
    dists, _ = tree.query(point, k=min(k, len(circle_points)))

    # Return average distance to nearest neighbors
    return np.mean(dists[dists > 0]) if np.any(dists > 0) else 1.0

def initialize_circles_heuristic(n=32):
    """Initialize circle positions using a density-adaptive heuristic approach."""
    circles = []
    
    # Precompute some parameters for better distribution
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    # Create initial placements with varying radii based on density
    for i in range(grid_size):
        for j in range(grid_size):
            if len(circles) >= n:
                break
            x = (i + 1) * spacing
            y = (j + 1) * spacing
            
            # Initial radius estimate based on available space
            r_min = min(x, y, 1-x, 1-y)
            
            # Density-based adjustment: circles in denser regions get smaller radii
            point = np.array([x, y])
            density = compute_local_density(circles, point)
            # Adjust radius based on how dense the area is (lower density = larger radius)
            r_factor = max(0.2, 1.0 / (1.0 + density * 5))
            r = min(r_min * r_factor, 0.15)
            
            # Only add if valid
            if is_valid_placement(circles, x, y, r):
                circles.append([x, y, r])

    # Fill remaining spots with smaller circles
    while len(circles) < n:
        best_r = 0
        best_x, best_y = 0, 0
        
        # Sample potential positions
        for _ in range(1000):  # Sample many points
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Estimate max radius at this location
            r_max = min(x, y, 1-x, 1-y)
            if r_max <= 0:
                continue
                
            # Density-based radius estimation
            point = np.array([x, y])
            density = compute_local_density(circles, point)
            # Adjust radius based on how dense the area is
            r_factor = max(0.1, 1.0 / (1.0 + density * 5))
            adjusted_r_max = r_max * r_factor
            
            # Try different radii
            test_radii = np.linspace(0.005, adjusted_r_max * 0.4, 10)
            for r in test_radii:
                if is_valid_placement(circles, x, y, r):
                    if r > best_r:
                        best_r = r
                        best_x, best_y = x, y
                        break
                        
        if best_r > 0:
            circles.append([best_x, best_y, best_r])
    
    return np.array(circles)

def optimize_circles(circles, max_iter=1000):
    """Apply local optimization to improve the circle packing configuration."""
    # Build KDTree for fast neighbor queries
    circle_points = np.array([[c[0], c[1]] for c in circles])
    tree = cKDTree(circle_points)
    
    # Find maximum radius for neighbor search optimization
    max_radius = np.max(circles[:, 2]) if len(circles) > 0 else 0.1
    
    # Store the best solution found so far
    best_circles = circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    for iteration in range(max_iter):
        # Try to improve by adjusting one circle at a time
        for i in range(len(circles)):
            cx, cy, cr = circles[i]
            
            # Save original values
            orig_x, orig_y, orig_r = cx, cy, cr
            
            # Try small perturbations
            for _ in range(100):
                # Generate random perturbation
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                dr = np.random.uniform(-0.005, 0.005)
                
                new_x = orig_x + dx
                new_y = orig_y + dy
                new_r = orig_r + dr
                
                # Ensure new_r is positive
                if new_r <= 0:
                    continue
                    
                # Ensure new position is inside the square
                if new_r > new_x or new_r > new_y or new_r > (1-new_x) or new_r > (1-new_y):
                    continue
                    
                # Check if new placement is valid
                if is_valid_placement(circles, new_x, new_y, new_r, tree, max_radius):
                    # Temporarily update this circle
                    circles[i] = [new_x, new_y, new_r]
                    
                    # Check if this gives a better total radius
                    new_sum = np.sum(circles[:, 2])
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = circles.copy()
                    else:
                        # Revert the change
                        circles[i] = [orig_x, orig_y, orig_r]
        
        # Update circles to best found so far
        circles[:] = best_circles[:]
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = initialize_circles_heuristic(n)
    
    # Apply local optimization to further improve the solution
    circles = optimize_circles(circles)
    
    return circles

# EVOLVE-BLOCK-END