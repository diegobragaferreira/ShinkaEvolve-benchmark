# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    def check_overlap(circles, i, x, y, r):
        """Check if a new circle overlaps with existing ones."""
        for j in range(i):
            ex, ey, er = circles[j]
            distance = math.sqrt((x - ex)**2 + (y - ey)**2)
            if distance < r + er:
                return True
        return False
    
    def is_valid_position(circles, i, x, y, r):
        """Check if a circle is within bounds and doesn't overlap."""
        # Check boundary constraints
        if r > x or r > y or r > (1 - x) or r > (1 - y):
            return False
        
        # Check overlap
        return not check_overlap(circles, i, x, y, r)
    
    def compute_total_radius(circles):
        """Compute the sum of all radii."""
        return sum(circles[:, 2])
    
    def objective_function(params):
        """Objective function to minimize (negative sum of radii)."""
        circles = params.reshape(-1, 3)
        return -compute_total_radius(circles)
    
    def constraint_function(params):
        """Constraint function ensuring no overlaps."""
        circles = params.reshape(-1, 3)
        penalty = 0
        n = len(circles)
        
        for i in range(n):
            x, y, r = circles[i]
            # Boundary constraints
            if r > x or r > y or r > (1 - x) or r > (1 - y):
                penalty += 1000
            
            # Overlap constraints
            for j in range(i):
                ex, ey, er = circles[j]
                distance = math.sqrt((x - ex)**2 + (y - ey)**2)
                if distance < r + er:
                    penalty += 1000 * (r + er - distance)
                    
        return penalty
    
    # Phase 1: Generate initial configuration using hexagonal packing
    circles = np.zeros((32, 3))
    
    # Create a grid-based initial configuration
    n_rows = 6
    n_cols = 6
    padding = 0.05
    grid_size = (1 - 2*padding) / max(n_rows, n_cols)
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx >= 32:
                break
            x = padding + (j + 0.5) * grid_size
            y = padding + (i + 0.5) * grid_size
            r = grid_size / 4  # Initial radius
            
            # Ensure it's within bounds and doesn't overlap with others
            if is_valid_position(circles[:idx], idx, x, y, r):
                circles[idx] = [x, y, r]
                idx += 1
        if idx >= 32:
            break
    
    # Fill remaining slots with random valid placements
    while idx < 32:
        x = np.random.uniform(padding + 0.01, 1 - padding - 0.01)
        y = np.random.uniform(padding + 0.01, 1 - padding - 0.01)
        r = np.random.uniform(0.005, 0.05)
        
        if is_valid_position(circles[:idx], idx, x, y, r):
            circles[idx] = [x, y, r]
            idx += 1
    
    # Phase 2: Optimization using scipy minimize
    # Flatten the circles array for optimization
    initial_params = circles.flatten()
    
    # Define bounds for each parameter (x, y, r)
    bounds = []
    for i in range(32):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])
    
    # Use L-BFGS-B optimizer which handles bounds well
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            print("Optimization did not converge, using initial configuration")
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation and adjustment
    final_circles = np.zeros((32, 3))
    valid_count = 0
    
    # Reconstruct valid circles
    for i in range(32):
        x, y, r = circles[i]
        # Apply boundary constraints
        r = min(r, x, y, 1-x, 1-y)
        
        # Check if this circle is valid
        if r > 0 and is_valid_position(final_circles[:valid_count], valid_count, x, y, r):
            final_circles[valid_count] = [x, y, r]
            valid_count += 1
        else:
            # Try to find a valid replacement
            found_valid = False
            for attempt in range(1000):
                new_x = np.random.uniform(max(0.01, r), min(0.99, 1-r))
                new_y = np.random.uniform(max(0.01, r), min(0.99, 1-r))
                new_r = np.random.uniform(max(0.001, r/2), min(0.1, r*2))
                
                if is_valid_position(final_circles[:valid_count], valid_count, new_x, new_y, new_r):
                    final_circles[valid_count] = [new_x, new_y, new_r]
                    valid_count += 1
                    found_valid = True
                    break
                    
            if not found_valid:
                # If we can't find a valid replacement, use a conservative approach
                if valid_count < 32:
                    final_circles[valid_count] = [0.5, 0.5, 0.01]
                    valid_count += 1
    
    # Ensure we have exactly 32 circles
    final_circles = final_circles[:32]
    
    return final_circles

# EVOLVE-BLOCK-END