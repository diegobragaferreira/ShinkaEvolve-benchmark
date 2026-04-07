# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For deterministic results
    
    # Hexagonal grid initialization
    n = 32
    circles = np.zeros((n, 3))
    
    # Create hexagonal grid pattern
    # Calculate grid parameters based on circle packing density
    rows = int(math.sqrt(n / math.sqrt(3))) + 2
    cols = int(n / rows) + 2
    
    # Ensure we have enough space for all circles
    while rows * cols < n:
        rows += 1
        cols = int(n / rows) + 1
    
    # Hexagonal packing parameters
    # Estimate radius based on area coverage
    estimated_area = 1.0  # unit square area
    required_area = n * math.pi * 0.1**2  # assuming initial radius of 0.1
    ratio = required_area / estimated_area
    
    # More accurate calculation of hexagonal packing parameters
    hex_radius = 0.15  # Initial estimate
    spacing_x = hex_radius * 2
    spacing_y = hex_radius * math.sqrt(3)
    
    # Adjust spacing to fit within bounds with some margin
    max_radius = 0.15
    min_radius = 0.01
    
    # Initialize with hexagonal arrangement
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row to create hexagonal pattern
            x_offset = (i % 2) * (spacing_x / 2)
            x = x_offset + j * spacing_x + hex_radius
            y = i * spacing_y + hex_radius
            
            # Check if within bounds with a safety margin
            if x <= 1 - hex_radius and y <= 1 - hex_radius:
                circles[idx] = [x, y, hex_radius]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining circles with random valid positions if necessary
    if idx < n:
        for i in range(idx, n):
            attempts = 0
            while attempts < 1000:
                x = np.random.uniform(hex_radius, 1 - hex_radius)
                y = np.random.uniform(hex_radius, 1 - hex_radius)
                # Check against existing circles for overlap
                valid = True
                for k in range(i):
                    dist = math.sqrt((x - circles[k][0])**2 + (y - circles[k][1])**2)
                    if dist < circles[k][2] + hex_radius:
                        valid = False
                        break
                if valid:
                    circles[i] = [x, y, hex_radius]
                    break
                attempts += 1
    
    # Local optimization to improve configuration
    def objective(params):
        # params is flattened array [x1,y1,r1,x2,y2,r2,...]
        total_radius = 0
        positions = []
        
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            positions.append([x, y])
            total_radius += r
        
        # Convert to numpy array for distance calculation
        pos_array = np.array(positions)
        
        # Penalty for overlapping circles
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = math.sqrt((pos_array[i][0] - pos_array[j][0])**2 + (pos_array[i][1] - pos_array[j][1])**2)
                min_dist = params[3*i+2] + params[3*j+2]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2 * 1000  # Scale up penalty
        
        # Penalty for going out of bounds
        for i in range(n):
            if params[3*i] < params[3*i+2] or params[3*i] > 1 - params[3*i+2]:
                penalty += 100000
            if params[3*i+1] < params[3*i+2] or params[3*i+1] > 1 - params[3*i+2]:
                penalty += 100000
                
        # Return negative because we want to maximize
        return -total_radius + penalty
    
    def constraint_func(params):
        # Ensure no overlaps and boundary constraints
        positions = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            positions.append([x, y])
            
        pos_array = np.array(positions)
        
        # Check overlap constraints
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = math.sqrt((pos_array[i][0] - pos_array[j][0])**2 + (pos_array[i][1] - pos_array[j][1])**2)
                min_dist = params[3*i+2] + params[3*j+2]
                constraints.append(dist - min_dist)  # Should be >= 0
        
        # Boundary constraints
        for i in range(n):
            constraints.append(params[3*i] - params[3*i+2])  # x >= r
            constraints.append(1 - params[3*i] - params[3*i+2])  # 1-x >= r
            constraints.append(params[3*i+1] - params[3*i+2])  # y >= r
            constraints.append(1 - params[3*i+1] - params[3*i+2])  # 1-y >= r
            
        return np.array(constraints)
    
    # Flatten initial configuration
    initial_params = np.zeros(n * 3)
    for i in range(n):
        initial_params[3*i] = circles[i][0]
        initial_params[3*i+1] = circles[i][1]
        initial_params[3*i+2] = circles[i][2]
    
    # Use scipy optimization with bounds - use L-BFGS-B which handles bounds well
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])  # [x, y, r] bounds
    
    try:
        # Optimize with L-BFGS-B method which works well with bounds
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
        
        if result.success:
            final_params = result.x
            for i in range(n):
                circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
    
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # Final verification and adjustment
    def verify_and_correct(circles_arr):
        # Ensure no overlaps and boundaries
        changed = True
        iterations = 0
        while changed and iterations < 30:
            changed = False
            for i in range(n):
                x, y, r = circles_arr[i]
                
                # Check boundary constraints
                if x < r:
                    x = r
                    changed = True
                elif x > 1 - r:
                    x = 1 - r
                    changed = True
                    
                if y < r:
                    y = r
                    changed = True
                elif y > 1 - r:
                    y = 1 - r
                    changed = True
                
                # Check overlap constraints with other circles
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles_arr[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < r + r2:
                            # Move circle slightly away
                            dx = x2 - x
                            dy = y2 - y
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance > 0:
                                scale = (r + r2 - distance) / distance
                                x -= dx * scale * 0.5
                                y -= dy * scale * 0.5
                                changed = True
                
                circles_arr[i] = [x, y, r]
            
            iterations += 1
        
        return circles_arr
    
    circles = verify_and_correct(circles)
    
    return circles

# EVOLVE-BLOCK-END
