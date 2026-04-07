# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimize width/height ratio - try 2:1 ratio which often works well for circle packing
    width, height = 1.2, 0.8

    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))

    # Create hexagonal grid pattern with better spacing
    rows = 5
    cols = 5
    
    # Estimate initial spacing based on container size
    estimated_radius = min(width, height) * 0.08
    
    # Generate hexagonal grid points
    hex_points = []
    
    # Use proper hexagonal lattice pattern
    for i in range(rows):
        for j in range(cols):
            if len(hex_points) >= n:
                break
            # Hexagonal coordinate calculation
            x = j * 2 * estimated_radius + (i % 2) * estimated_radius
            y = i * np.sqrt(3) * estimated_radius
            
            # Only add if within bounds
            if 0 <= x <= width and 0 <= y <= height:
                hex_points.append([x, y])
        
        if len(hex_points) >= n:
            break

    # If we don't have enough points, add some centered points
    while len(hex_points) < n:
        hex_points.append([width/2, height/2])

    # Take first n points
    points = np.array(hex_points[:n])

    # Initialize with small radii
    for i in range(n):
        circles[i] = [points[i][0], points[i][1], 0.01]

    # Local optimization to maximize radii - more aggressive optimization
    max_iterations = 500
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase each circle's radius
        for i in range(n):
            # Find maximum possible radius for circle i
            max_radius = calculate_max_radius(circles, i, width, height)
            
            if max_radius > circles[i][2]:
                circles[i][2] = max_radius
                improved = True
        
        if not improved:
            break

    # Final refinement using a local search approach with systematic exploration
    for _ in range(300):
        # Try moving each circle slightly to see if we can improve the configuration
        for i in range(n):
            current_x, current_y, current_r = circles[i]
            
            # Try small movements in different directions
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r
            
            # Examine 5x5 grid around current position
            for dx in [-0.08, -0.04, 0, 0.04, 0.08]:
                for dy in [-0.08, -0.04, 0, 0.04, 0.08]:
                    new_x, new_y = current_x + dx, current_y + dy
                    
                    # Check if new position is within bounds
                    if 0 <= new_x <= width and 0 <= new_y <= height:
                        # Calculate max radius at new position
                        max_radius = calculate_max_radius_at_position(
                            circles, i, new_x, new_y, width, height
                        )
                        
                        if max_radius > best_radius:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]
            
            # Update if we found a better position
            if best_pos[2] > circles[i][2]:
                circles[i] = best_pos

    return circles


def calculate_max_radius(circles, index, width, height):
    """Calculate maximum radius for circle at given index without overlapping others."""
    x, y, current_radius = circles[index]
    
    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)
    
    # Maximum radius based on other circles
    max_radius_overlap = float('inf')
    
    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr
            
            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle
    
    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


def calculate_max_radius_at_position(circles, index, x, y, width, height):
    """Calculate maximum radius for circle at given position without overlapping others."""
    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)
    
    # Maximum radius based on other circles
    max_radius_overlap = float('inf')
    
    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr
            
            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle
    
    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")