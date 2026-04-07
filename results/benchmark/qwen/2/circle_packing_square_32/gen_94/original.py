# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.spatial import distance

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with hexagonal grid
    rows = 6
    cols = 6
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x
    
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            if x <= 1.0 and y <= 1.0:
                grid_points.append([x, y])
    
    # Take first 32 points
    points = np.array(grid_points[:n])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = points[i][0]
        circles[i][1] = points[i][1]
        circles[i][2] = 0.02  # Small initial radius
    
    # Refinement using Voronoi-based approach
    for iteration in range(100):
        # Store previous state for comparison
        prev_circles = circles.copy()
        
        # Compute Voronoi diagram of circle centers
        vor = Voronoi(circles[:, :2])
        
        # Update each circle's radius based on Voronoi regions
        for i in range(n):
            x, y, r = circles[i]
            
            # Calculate boundary constraints
            boundary_constraint = min(x, 1-x, y, 1-y)
            
            # Find neighbors using Voronoi regions
            neighbors = []
            for j in range(n):
                if i != j:
                    # Check if circles are close enough to interfere
                    dist = np.sqrt((x - circles[j][0])**2 + (y - circles[j][1])**2)
                    if dist < r + circles[j][2] + 0.01:  # Add small buffer
                        neighbors.append(j)
            
            # Calculate maximum possible radius considering neighbors
            max_radius = boundary_constraint
            
            if neighbors:
                # Consider interference with neighbors
                for j in neighbors:
                    # Distance to center of neighbor circle
                    dist = np.sqrt((x - circles[j][0])**2 + (y - circles[j][1])**2)
                    # Maximum radius to prevent overlap
                    max_radius = min(max_radius, dist - circles[j][2])
            
            # Update radius with some relaxation factor
            if max_radius > 0.001:
                # Use a small fraction of the maximum possible radius to avoid getting stuck
                new_radius = min(max_radius, r * 1.05)  # Allow small increases
                circles[i][2] = max(0.001, min(0.15, new_radius))
        
        # Move circles to maximize total radius while respecting constraints
        # Simple force-based relaxation method
        for i in range(n):
            # Calculate forces from boundaries
            fx, fy = 0.0, 0.0
            
            # Boundary forces (repulsive)
            x, y, r = circles[i]
            boundary_forces = [
                (0.01 * (r - x), 0),      # Left boundary
                (0.01 * (x + r - 1), 0),  # Right boundary  
                (0.01 * (r - y), 0),      # Bottom boundary
                (0.01 * (y + r - 1), 0)   # Top boundary
            ]
            
            for bx, by in boundary_forces:
                if bx > 0: fx += bx
                if by > 0: fy += by
            
            # Neighbor forces (repulsive)
            for j in range(n):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < r1 + r2:
                        # Repulsive force
                        force_magnitude = 0.01 * (r1 + r2 - dist)
                        if dist > 0:
                            fx += force_magnitude * dx / dist
                            fy += force_magnitude * dy / dist
            
            # Apply force to position
            new_x = x + 0.001 * fx
            new_y = y + 0.001 * fy
            
            # Clamp to valid range
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            circles[i][0] = new_x
            circles[i][1] = new_y
        
        # Check for convergence
        if np.allclose(prev_circles, circles, rtol=1e-6):
            break
    
    # Final refinement to ensure no overlaps
    for _ in range(20):
        updated = False
        for i in range(n):
            x, y, r = circles[i]
            
            # Check all other circles for overlap
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < r + r2:
                        # Reduce radius to maintain separation
                        new_r = max(0.001, (r + r2 - dist) * 0.99)
                        if new_r < r:
                            circles[i][2] = new_r
                            updated = True
                            
        if not updated:
            break
    
    return circles

# EVOLVE-BLOCK-END