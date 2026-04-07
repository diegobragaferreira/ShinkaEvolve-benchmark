# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import sqrt, ceil

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # We'll use width = 1.2 and height = 0.8 for a reasonable aspect ratio
    width = 1.2
    height = 0.8
    
    # For 21 circles, we'll try a hexagonal lattice arrangement
    # Calculate grid parameters
    n_circles = 21
    
    # Try to arrange in roughly a hexagonal pattern
    # Find the closest hexagonal packing that fits 21 circles
    rows = int(ceil(sqrt(n_circles)))
    cols = int(ceil(n_circles / rows))
    
    # If we have too many columns, adjust
    if cols * rows < n_circles:
        cols += 1
    
    # Hexagonal packing spacing
    # For circles of equal radius r, centers should be at distance 2r apart
    # But we need to account for hexagonal arrangement
    # Let's start with a reasonable guess for radius
    initial_radius = 0.1
    
    # Create the initial hexagonal grid
    x_positions = []
    y_positions = []
    
    # Hexagonal grid points
    row_spacing = initial_radius * 2 * sqrt(3) / 2  # Vertical spacing for hexagon
    col_spacing = initial_radius * 2  # Horizontal spacing
    
    # Adjust to fit within bounds
    max_x = width - initial_radius
    max_y = height - initial_radius
    
    # Generate the hexagonal grid points
    for row in range(rows):
        y = initial_radius + row * row_spacing
        if y > max_y:
            break
            
        for col in range(cols):
            x = initial_radius + col * col_spacing
            if col % 2 == 1:  # Offset every other row
                x += col_spacing / 2
                
            if x <= max_x:
                x_positions.append(x)
                y_positions.append(y)
                
            if len(x_positions) >= n_circles:
                break
        if len(x_positions) >= n_circles:
            break
    
    # Trim to exactly 21 circles
    x_positions = x_positions[:n_circles]
    y_positions = y_positions[:n_circles]
    
    # Initial guess for all radii
    radii = [initial_radius] * n_circles
    
    # Flatten into single array for optimization: [x1, y1, r1, x2, y2, r2, ...]
    initial_guess = []
    for i in range(n_circles):
        initial_guess.extend([x_positions[i], y_positions[i], radii[i]])
        
    # Optimization bounds
    bounds = []
    for i in range(n_circles):
        # x bounds
        bounds.append((initial_radius, width - initial_radius))
        # y bounds  
        bounds.append((initial_radius, height - initial_radius))
        # r bounds
        bounds.append((1e-6, min(width, height) / 2))
    
    def objective(params):
        # Extract parameters
        total_radius = 0
        circles = []
        for i in range(n_circles):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append((x, y, r))
            total_radius += r
            
        # Penalty for constraint violations
        penalty = 0
        
        # Check boundary constraints
        for i in range(n_circles):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty += 1000000  # Large penalty
        
        # Check overlap constraints
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_squared = (x2-x1)**2 + (y2-y1)**2
                min_distance_squared = (r1+r2)**2
                if distance_squared < min_distance_squared:
                    # Penalty based on how much they overlap
                    overlap = min_distance_squared - distance_squared
                    penalty += 100000 * overlap
                    
        return -(total_radius - penalty)  # Negative because we want to maximize
    
    # Use L-BFGS-B or Nelder-Mead for optimization
    try:
        result = minimize(objective, initial_guess, method='Nelder-Mead', 
                         bounds=bounds, options={'maxiter': 5000, 'adaptive': True})
        
        if result.success:
            final_params = result.x
            circles = []
            for i in range(n_circles):
                x = final_params[3*i]
                y = final_params[3*i+1]
                r = final_params[3*i+2]
                circles.append([x, y, r])
        else:
            # Fallback to initial solution
            circles = []
            for i in range(n_circles):
                circles.append([x_positions[i], y_positions[i], radii[i]])
    except Exception:
        # Fallback to initial solution
        circles = []
        for i in range(n_circles):
            circles.append([x_positions[i], y_positions[i], radii[i]])
    
    return np.array(circles)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
