# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    # Initialize with Voronoi-inspired spread
    n_circles = 26
    circles = np.zeros((n_circles, 3))
    
    # Step 1: Generate initial configuration using a more systematic approach
    # Create a grid pattern with slight perturbation to avoid perfect symmetry
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / (grid_size + 1)
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n_circles:
                break
            # Add slight jitter to avoid symmetric issues
            x = (j + 1) * spacing + np.random.normal(0, spacing * 0.1)
            y = (i + 1) * spacing + np.random.normal(0, spacing * 0.1)
            # Clamp to valid range
            x = np.clip(x, spacing * 0.5, 1 - spacing * 0.5)
            y = np.clip(y, spacing * 0.5, 1 - spacing * 0.5)
            
            # Initial radius estimation based on spacing
            r = spacing * 0.3
            circles[count] = [x, y, r]
            count += 1
    
    # Step 2: Local optimization using iterative improvement
    max_iterations = 500
    tolerance = 1e-6
    best_sum_radii = 0
    best_circles = circles.copy()
    
    for iteration in range(max_iterations):
        # Try to maximize each circle's radius individually
        improved = False
        new_circles = circles.copy()
        
        # Try to increase radius for each circle
        for i in range(n_circles):
            # Find maximum possible radius for circle i without violating constraints
            max_radius = find_max_radius(new_circles, i)
            
            if max_radius > new_circles[i, 2]:
                new_circles[i, 2] = max_radius
                improved = True
        
        # If no radius improvements, try local position adjustments
        if not improved:
            for i in range(n_circles):
                # Try to slightly adjust position to allow larger radii
                new_pos = adjust_position(new_circles, i)
                if new_pos is not None:
                    new_circles[i, :2] = new_pos
                    improved = True
        
        # Update if improvement was made
        if improved:
            circles = new_circles
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_circles = circles.copy()
        else:
            # No improvement, stop early
            break
    
    # Final refinement with optimization approach
    final_circles = optimize_circles(best_circles.copy())
    
    return final_circles

def find_max_radius(circles, target_idx):
    """Find maximum possible radius for circle at target_idx without overlapping others."""
    x, y, current_r = circles[target_idx]
    
    # Determine maximum possible radius
    max_radius = min(x, 1-x, y, 1-y)  # Boundary constraints
    
    # Check overlaps with all other circles
    for i in range(len(circles)):
        if i != target_idx:
            x_other, y_other, r_other = circles[i]
            # Distance to center of other circle
            dist = np.sqrt((x - x_other)**2 + (y - y_other)**2)
            # Maximum radius to avoid overlap
            max_for_this_circle = dist - r_other
            max_radius = min(max_radius, max_for_this_circle)
    
    # Ensure non-negative radius
    return max(0.001, max_radius)

def adjust_position(circles, target_idx):
    """Try to improve position of a circle to allow for larger radius."""
    x, y, r = circles[target_idx]
    best_pos = None
    best_radius = r
    
    # Sample nearby positions
    step_sizes = [0.005, 0.01, 0.02]
    for step in step_sizes:
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                new_x = x + dx
                new_y = y + dy
                # Check if new position is valid
                if (new_x >= r and new_x <= 1-r and 
                    new_y >= r and new_y <= 1-r):
                    # Check if this move allows for larger radius
                    temp_circles = circles.copy()
                    temp_circles[target_idx, :2] = [new_x, new_y]
                    new_max_radius = find_max_radius(temp_circles, target_idx)
                    if new_max_radius > best_radius:
                        best_radius = new_max_radius
                        best_pos = [new_x, new_y]
    
    return best_pos

def optimize_circles(initial_circles):
    """Use a more direct optimization approach."""
    n = len(initial_circles)
    
    # Flatten the problem for optimization
    # We'll work with positions and radii separately
    initial_params = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_params.extend([x, y, r])
    
    def objective(params):
        """Minimize negative sum of radii (equivalent to maximizing sum)."""
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        """Constraint: no overlaps and containment."""
        circles = params.reshape(-1, 3)
        penalties = []
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                penalties.append(1000.0)
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1 + r2):
                    # Penalty proportional to violation amount
                    penalties.append(10000.0 * (r1 + r2 - dist))
        
        return sum(penalties)
    
    # Try to optimize using scipy minimize with constraints
    try:
        # Define bounds for each variable (x, y, r for each circle)
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((0.001, 0.999))  # x coordinate
            bounds.append((0.001, 0.999))  # y coordinate  
            bounds.append((0.001, 0.4))    # radius (reasonable upper bound)
        
        # Use L-BFGS-B with bounds
        from scipy.optimize import minimize
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 100})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all radii are positive and valid
            for i in range(n):
                if optimized_circles[i, 2] < 0.001:
                    optimized_circles[i, 2] = 0.001
                # Ensure containment
                x, y, r = optimized_circles[i]
                optimized_circles[i] = [x, y, min(r, x, 1-x, y, 1-y)]
            return optimized_circles
    except:
        pass
    
    return initial_circles

# EVOLVE-BLOCK-END