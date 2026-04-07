# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    
    def constraint_circle_containment(circles_flat):
        """Ensure all circles are fully contained within the unit square"""
        x = circles_flat[::3]
        y = circles_flat[1::3]
        r = circles_flat[2::3]
        
        # Check containment constraints: r <= x <= 1-r and r <= y <= 1-r
        containment_x = np.logical_and(r <= x, x <= 1 - r)
        containment_y = np.logical_and(r <= y, y <= 1 - r)
        return np.all(containment_x) and np.all(containment_y)
    
    def constraint_circle_overlap(circles_flat):
        """Ensure no two circles overlap"""
        x = circles_flat[::3]
        y = circles_flat[1::3]
        r = circles_flat[2::3]
        
        # Compute pairwise distances between circle centers
        centers = np.column_stack([x, y])
        distances = cdist(centers, centers)
        
        # Create mask for pairs of circles (excluding diagonal)
        n_pairs = len(x) * (len(x) - 1) // 2
        indices = []
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                indices.append((i, j))
                
        # Check overlap constraint: distance >= r_i + r_j
        for i, j in indices:
            distance = distances[i, j]
            min_distance = r[i] + r[j]
            if distance < min_distance:
                return False
        return True
    
    def objective_f(circles_flat):
        """Minimize negative sum of radii (maximize sum of radii)"""
        return -np.sum(circles_flat[2::3])
    
    def constraint_func(circles_flat):
        """Combined constraint function"""
        # For optimization, we need to handle the constraints properly
        # We'll return a value that penalizes constraint violations heavily
        penalty = 0
        
        # Check containment
        x = circles_flat[::3]
        y = circles_flat[1::3]
        r = circles_flat[2::3]
        
        # Penalty for containment violations
        containment_violations = 0
        for i in range(len(x)):
            if r[i] > x[i] or x[i] > 1 - r[i]:
                containment_violations += (r[i] - x[i])**2 + (x[i] - (1 - r[i]))**2
            if r[i] > y[i] or y[i] > 1 - r[i]:
                containment_violations += (r[i] - y[i])**2 + (y[i] - (1 - r[i]))**2
        
        # Penalty for overlap violations
        overlap_violations = 0
        centers = np.column_stack([x, y])
        distances = cdist(centers, centers)
        
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                distance = distances[i, j]
                min_distance = r[i] + r[j]
                if distance < min_distance:
                    overlap_violations += (min_distance - distance)**2
        
        penalty = containment_violations + overlap_violations
        return penalty
    
    # Initialize with a structured approach - grid-like configuration
    # Start with a regular grid arrangement
    sqrt_n = int(math.ceil(math.sqrt(n)))
    grid_size = sqrt_n
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    # Initialize positions and radii
    circles_flat = np.zeros(3 * n)
    
    # Place circles in a grid pattern
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            
            # Initial radius - small but ensure containment
            r = min(spacing_x, spacing_y) / 2.0 * 0.9
            
            circles_flat[count*3] = x  # x coordinate
            circles_flat[count*3 + 1] = y  # y coordinate
            circles_flat[count*3 + 2] = r  # radius
            count += 1
        if count >= n:
            break
    
    # If there are fewer circles placed, fill the rest randomly but still contained
    for i in range(count, n):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.01, 0.1)
        circles_flat[i*3] = x
        circles_flat[i*3 + 1] = y
        circles_flat[i*3 + 2] = r
    
    # Refine using optimization
    # Define bounds (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Run optimization
    try:
        result = minimize(
            objective_f,
            circles_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8},
            callback=None  # Can add monitoring if needed
        )
        
        if result.success:
            circles_flat = result.x
    except Exception as e:
        # If optimization fails, keep the initial configuration
        pass
    
    # Convert back to standard array format
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = circles_flat[i*3]      # x
        circles[i][1] = circles_flat[i*3 + 1]  # y
        circles[i][2] = circles_flat[i*3 + 2]  # r
    
    return circles

# EVOLVE-BLOCK-END
