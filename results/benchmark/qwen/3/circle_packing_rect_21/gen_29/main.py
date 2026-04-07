# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Using width=1.2, height=0.8 as a reasonable ratio for good packing
    width = 1.2
    height = 0.8
    
    n = 21
    circles = np.zeros((n, 3))
    
    # Define bounds for optimization
    # Each circle has (x, y, r) coordinates
    # x in [r, width-r], y in [r, height-r], r in [0, min(width,height)/2]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, width-0.001), (0.001, height-0.001), (0.001, min(width, height)/2)])
    
    def objective(x):
        """Minimize negative sum of radii (maximize sum of radii)"""
        total_radius = 0
        for i in range(n):
            total_radius += x[2*i+2]  # radius is at index 2*i+2
        return -total_radius
    
    def constraint_func(x):
        """Ensure no overlaps and all circles fit in rectangle"""
        penalty = 0
        
        # Check boundary constraints
        for i in range(n):
            xi, yi, ri = x[2*i], x[2*i+1], x[2*i+2]
            if xi - ri < 0 or xi + ri > width or yi - ri < 0 or yi + ri > height:
                penalty += 1000  # Large penalty for boundary violations
        
        # Check overlap constraints
        positions = x.reshape(-1, 3)[:, :2]  # Extract (x,y) positions
        radii = x.reshape(-1, 3)[:, 2]      # Extract radii
        
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Add penalty proportional to how much they overlap
                    penalty += (min_dist - dist) * 10000
        
        return penalty
    
    # Initial guess: hexagonal packing
    rows = 4
    cols = 6
    
    # Calculate spacing
    col_spacing = width / (cols + 1)
    row_spacing = height / (rows + 1)
    
    # Hexagonal offset
    hex_offset = col_spacing * 0.5
    
    initial_guess = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col + 1) * col_spacing
            if row % 2 == 1:  # Offset odd rows
                x += hex_offset
            y = (row + 1) * row_spacing
            
            # Set initial radius to some small value
            initial_guess.extend([x, y, 0.05])
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining slots with zeros if needed
    for i in range(len(initial_guess), 3*n):
        initial_guess.append(0.5)
    
    # Optimization settings
    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        args=(),
        maxiter=1000,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        callback=None,
        disp=False
    )
    
    # Extract optimized solution
    optimized_solution = result.x
    circles = optimized_solution.reshape(-1, 3)
    
    # Ensure all circles are within bounds
    for i in range(n):
        x, y, r = circles[i]
        # Clip values to ensure they're within rectangle
        circles[i] = [
            max(r, min(width - r, x)), 
            max(r, min(height - r, y)), 
            max(0.001, min(min(width, height)/2, r))
        ]
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")