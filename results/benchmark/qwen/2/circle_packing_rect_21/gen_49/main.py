# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
from scipy.optimize import differential_evolution
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, use 1x1 for simplicity (perimeter = 4)
    width, height = 1.0, 1.0
    
    # Initialize parameters
    n_circles = 21
    max_iter = 1000
    
    # Phase 1: Hexagonal grid initialization for good density
    circles = np.zeros((n_circles, 3))
    
    # Hexagonal packing parameters
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Calculate spacing for hexagonal packing
    hex_radius = 0.1
    x_spacing = hex_radius * 2
    y_spacing = hex_radius * np.sqrt(3)
    
    # Grid positions with offset for hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            x = (j + 0.5 * (i % 2)) * x_spacing + hex_radius
            y = i * y_spacing + hex_radius
            # Adjust to stay within bounds
            x = min(width - hex_radius, max(hex_radius, x))
            y = min(height - hex_radius, max(hex_radius, y))
            circles[idx] = [x, y, hex_radius]
            idx += 1
        if idx >= n_circles:
            break
    
    # Phase 2: Physics-based optimization with spatial indexing
    def get_spatial_index(circles_array):
        points = circles_array[:, :2]
        tree = cKDTree(points)
        return tree
    
    # Check if circle is within bounds
    def is_valid_position(circle):
        x, y, r = circle
        return (r <= x <= width - r and 
                r <= y <= height - r)
    
    # Check overlap with existing circles efficiently
    def check_overlap(circle, existing_circles, tree=None):
        x, y, r = circle
        if tree is None:
            # Simple check for small arrays
            for cx, cy, cr in existing_circles:
                dx = x - cx
                dy = y - cy
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (r + cr):  # Overlap detected
                    return True
            return False
        else:
            # Use spatial index for large arrays
            neighbors = tree.query_ball_point([x, y], 2 * (r + 1e-6), p=2)
            for j in neighbors:
                if j >= len(existing_circles):
                    continue
                cx, cy, cr = existing_circles[j]
                dx = x - cx
                dy = y - cy
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (r + cr):  # Overlap detected
                    return True
            return False
    
    # Constraint violation penalty calculation
    def calculate_penalty(circles_array):
        penalty = 0
        tree = get_spatial_index(circles_array)
        
        for i, circle in enumerate(circles_array):
            x, y, r = circle
            # Boundary penalty
            boundary_dist = min(x, y, width - x, height - y)
            if boundary_dist < r:
                penalty += (r - boundary_dist) ** 2
            
            # Overlap penalty
            neighbors = tree.query_ball_point([x, y], 2 * (r + 1e-6), p=2)
            for j in neighbors:
                if i != j:
                    cx, cy, cr = circles_array[j]
                    dx = x - cx
                    dy = y - cy
                    distance = math.sqrt(dx*dx + dy*dy)
                    overlap = max(0, (r + cr) - distance)
                    if overlap > 0:
                        penalty += overlap ** 2
        
        return penalty
    
    # Apply physics-inspired update with improved force model
    def apply_physics_update(circles_array, max_iter=500):
        tree = get_spatial_index(circles_array)
        
        # Spring constants
        repulsion_strength = 0.5
        boundary_strength = 1.0
        attraction_strength = 0.02
        
        for iteration in range(max_iter):
            forces = np.zeros_like(circles_array)
            
            # Calculate repulsion forces
            for i in range(len(circles_array)):
                x1, y1, r1 = circles_array[i]
                neighbors = tree.query_ball_point([x1, y1], 2 * (r1 + 1e-6), p=2)
                
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles_array[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            # Repulsion force: stronger when closer and overlapping
                            repulsion = 0
                            if distance < (r1 + r2):
                                repulsion = repulsion_strength / (distance + 0.001) ** 2
                                repulsion *= max(0, (r1 + r2) - distance)
                            
                            if repulsion > 0:
                                forces[i, 0] -= repulsion * dx / distance
                                forces[i, 1] -= repulsion * dy / distance
            
            # Boundary forces
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                
                # Boundary penalties
                if x - r < 0:
                    forces[i, 0] += boundary_strength * (r - x)
                elif x + r > width:
                    forces[i, 0] -= boundary_strength * (x + r - width)
                    
                if y - r < 0:
                    forces[i, 1] += boundary_strength * (r - y)
                elif y + r > height:
                    forces[i, 1] -= boundary_strength * (y + r - height)
            
            # Update positions
            step_size = 0.01
            for i in range(len(circles_array)):
                circles_array[i, 0] += forces[i, 0] * step_size
                circles_array[i, 1] += forces[i, 1] * step_size
                
                # Keep radii positive and within bounds
                circles_array[i, 2] = max(0.0001, circles_array[i, 2])
                
                # Enforce valid positions
                if not is_valid_position(circles_array[i]):
                    x, y, r = circles_array[i]
                    x = max(r, min(width - r, x))
                    y = max(r, min(height - r, y))
                    circles_array[i] = [x, y, r]
            
            # Update spatial index
            tree = get_spatial_index(circles_array)
            
            # Check convergence
            if iteration % 50 == 0:
                penalty = calculate_penalty(circles_array)
                if penalty < 1e-6:
                    break
        
        return circles_array
    
    # Phase 3: Local radius maximization
    def maximize_radii(circles_array):
        # Convert to list of tuples for easier manipulation
        circle_list = [tuple(row) for row in circles_array]
        tree = get_spatial_index(circles_array)
        
        improvement = True
        iterations = 0
        while improvement and iterations < 50:
            improvement = False
            iterations += 1
            
            # Try to increase each radius
            for i in range(len(circle_list)):
                x, y, r = circle_list[i]
                
                # Determine maximum possible radius
                # Distance to nearest circle
                max_radius = float('inf')
                neighbors = tree.query_ball_point([x, y], 2, p=2)
                for j in neighbors:
                    if i != j:
                        cx, cy, cr = circle_list[j]
                        dx = x - cx
                        dy = y - cy
                        distance = math.sqrt(dx*dx + dy*dy)
                        max_radius = min(max_radius, distance - cr)
                
                # Distance to boundaries
                boundary_dist = min(x, y, width - x, height - y)
                max_radius = min(max_radius, boundary_dist)
                
                # If we can increase radius
                if max_radius > r and max_radius > 0:
                    new_r = min(max_radius, r + 0.01)
                    test_circle = (x, y, new_r)
                    
                    # Check validity
                    if is_valid_position(test_circle) and not check_overlap(test_circle, circle_list, tree):
                        circle_list[i] = test_circle
                        improvement = True
            
            # Update array and rebuild spatial index
            if improvement:
                for i, circle in enumerate(circle_list):
                    circles_array[i] = circle
                tree = get_spatial_index(circles_array)
        
        return circles_array
    
    # Phase 4: Global optimization with differential evolution
    def optimize_global(circles_array):
        # Flatten the problem for optimization: [x1, y1, r1, x2, y2, r2, ...]
        def objective(params):
            # Reconstruct circles
            circles_copy = circles_array.copy()
            for i in range(n_circles):
                circles_copy[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
            # Penalty for constraints (negative because we want to maximize)
            penalty = calculate_penalty(circles_copy)
            # Negative because we minimize negative penalty, which means maximize radii sum
            return -np.sum(circles_copy[:, 2]) + penalty * 1000
        
        # Bounds for each variable (x, y, r for each circle)
        bounds = []
        for i in range(n_circles):
            bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
        
        # Differential evolution optimization
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=50,
                popsize=15,
                seed=42,
                disp=False
            )
            
            # Reconstruct from result
            if result.success:
                for i in range(n_circles):
                    circles_array[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        except:
            pass  # Fall back to other methods if DE fails
            
        return circles_array
    
    # Execute phases
    # Phase 1: Physics optimization
    optimized_circles = apply_physics_update(circles.copy(), 300)
    
    # Phase 2: Radius maximization
    optimized_circles = maximize_radii(optimized_circles)
    
    # Phase 3: Global optimization
    optimized_circles = optimize_global(optimized_circles)
    
    # Phase 4: Fine-tune with additional physics
    optimized_circles = apply_physics_update(optimized_circles, 200)
    
    # Phase 5: Final radius maximization
    optimized_circles = maximize_radii(optimized_circles)
    
    return optimized_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
