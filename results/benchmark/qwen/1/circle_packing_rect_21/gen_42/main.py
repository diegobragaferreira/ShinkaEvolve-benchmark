# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle with perimeter = 4, so width + height = 2
    # Using a square for simplicity: width = height = 1
    rect_width = 1.0
    rect_height = 1.0
    
    # Initialize circles with hexagonal lattice pattern
    circles = np.zeros((n, 3))
    
    # Hexagonal packing arrangement
    rows = 4
    cols = 6
    
    # Calculate spacing
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        offset = spacing_x * (i % 2) * 0.5  # Offset every other row
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y
            
            # Ensure position is within bounds
            x = max(0.01, min(rect_width - 0.01, x))
            y = max(0.01, min(rect_height - 0.01, y))
            
            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles if needed
    while idx < n:
        x = np.random.uniform(0.01, rect_width - 0.01)
        y = np.random.uniform(0.01, rect_height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1
    
    # Enhanced optimization using multiple restarts
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()
    
    # Multiple optimization runs with different starting points
    for run in range(5):
        # Create a copy for this run
        current_circles = circles.copy()
        
        # More sophisticated optimization loop
        for iteration in range(500):
            improved = False
            
            # Shuffle circle indices for diverse optimization
            indices = list(range(n))
            np.random.shuffle(indices)
            
            for i in indices:
                # Calculate maximum allowable radius for this circle
                max_radius = min(
                    current_circles[i][0],  # Distance to left edge
                    rect_width - current_circles[i][0],  # Distance to right edge
                    current_circles[i][1],  # Distance to bottom edge
                    rect_height - current_circles[i][1]   # Distance to top edge
                ) - 0.001
                
                # Consider collision constraints with neighbors
                for j in range(n):
                    if i != j:
                        dx = current_circles[i][0] - current_circles[j][0]
                        dy = current_circles[i][1] - current_circles[j][1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        collision_radius = distance - current_circles[j][2] - 0.001
                        if collision_radius > 0:
                            max_radius = min(max_radius, collision_radius)
                
                # Attempt to increase radius
                if max_radius > current_circles[i][2] and max_radius > 0.001:
                    # Adaptive increment based on available space
                    delta = min(0.02, max_radius - current_circles[i][2])
                    if delta > 0.001:
                        current_circles[i][2] += delta
                        improved = True
            
            # If no improvement, reduce learning rate slightly
            if not improved:
                break
        
        # Update best solution
        current_sum = np.sum(current_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = current_circles.copy()
    
    # Final refinement step
    final_circles = best_circles.copy()
    for _ in range(100):
        improved = False
        for i in range(n):
            # Calculate maximum allowable radius for this circle
            max_radius = min(
                final_circles[i][0],  # Distance to left edge
                rect_width - final_circles[i][0],  # Distance to right edge
                final_circles[i][1],  # Distance to bottom edge
                rect_height - final_circles[i][1]   # Distance to top edge
            ) - 0.001
            
            # Consider collision constraints with neighbors
            for j in range(n):
                if i != j:
                    dx = final_circles[i][0] - final_circles[j][0]
                    dy = final_circles[i][1] - final_circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - final_circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)
            
            # Increase radius if beneficial
            if max_radius > final_circles[i][2] and max_radius > 0.001:
                # Try to increase by a small amount
                new_radius = min(max_radius, final_circles[i][2] + 0.01)
                if new_radius > final_circles[i][2]:
                    final_circles[i][2] = new_radius
                    improved = True
        
        # Stop if no improvement made
        if not improved:
            break
    
    return final_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")