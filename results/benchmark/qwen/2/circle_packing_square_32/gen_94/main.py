# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Strategic Initialization with Density Awareness
    # Create a more sophisticated grid pattern with better distribution
    rows = 6
    cols = 6
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x
    
    # Generate hexagonal grid with slight randomness for better distribution
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 + np.random.normal(0, 0.05)) * spacing_x
            y = (i + 0.5 + np.random.normal(0, 0.05)) * spacing_y
            if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0:
                grid_points.append([x, y])
    
    # Take first 32 points
    points = np.array(grid_points[:n])
    
    # Initialize with calculated initial radii based on local density
    circles = np.zeros((n, 3))
    
    # Calculate initial radii using neighbor density estimation
    tree = cKDTree(points)
    initial_radii = []
    
    for i, point in enumerate(points):
        # Find 5 nearest neighbors to estimate local density
        distances, indices = tree.query(point, k=min(6, len(points)), p=2)
        
        # Skip self-distance and compute average distance to neighbors
        if len(distances) > 1:
            avg_neighbor_distance = np.mean(distances[1:])
            # Larger spacing = larger radius possible
            max_possible_radius = min(avg_neighbor_distance/2, 0.15)
        else:
            max_possible_radius = 0.05
            
        # Apply boundary constraints
        boundary_radius = min(point[0], 1-point[0], point[1], 1-point[1])
        initial_radius = min(max_possible_radius, boundary_radius * 0.8, 0.15)
        initial_radii.append(max(0.005, initial_radius))
    
    for i in range(n):
        circles[i][0] = points[i][0]
        circles[i][1] = points[i][1]
        circles[i][2] = initial_radii[i]
    
    # Phase 2: Hybrid Optimization - Voronoi + Force-based Refinement
    
    # Precompute neighbor relationships for efficiency
    neighbor_tree = cKDTree(circles[:, :2])
    
    for iteration in range(200):  # Increased iterations for better convergence
        prev_circles = circles.copy()
        
        # Voronoi-based radius maximization (Phase A)
        # Compute Voronoi diagram for current configuration
        try:
            vor = Voronoi(circles[:, :2])
        except:
            # Fallback to standard approach if Voronoi fails
            vor = None
            
        # Update radii using Voronoi insights
        new_radii = np.zeros(n)
        
        for i in range(n):
            x, y, r = circles[i]
            
            # Boundary constraints
            boundary_constraint = min(x, 1-x, y, 1-y)
            
            # Use k-d tree for neighbor search instead of brute force
            # Find neighbors within a reasonable distance
            neighbors = neighbor_tree.query_ball_point([x, y], r*3)
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Find closest neighbor to determine maximum safe radius
            max_safe_radius = boundary_constraint
            
            if neighbors:
                for j in neighbors:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist > 0:  # Avoid division by zero
                        max_safe_radius = min(max_safe_radius, dist - r2)
            
            # Apply adaptive radius update
            if max_safe_radius > 0.001:
                # Allow some increase but be cautious about large jumps
                new_radii[i] = min(max_safe_radius, r * 1.1, 0.2)
            else:
                new_radii[i] = max(0.001, r * 0.95)
        
        # Apply updated radii
        for i in range(n):
            circles[i][2] = new_radii[i]
        
        # Force-based position refinement (Phase B)
        # Apply forces to circles based on constraints
        forces = np.zeros((n, 2))  # (fx, fy) for each circle
        
        # Boundary forces
        for i in range(n):
            x, y, r = circles[i]
            # Repulsion from boundaries
            fx = 0.0
            fy = 0.0
            
            # Left boundary
            if x - r < 0:
                fx += 0.1 * (r - x)
            # Right boundary  
            if x + r > 1:
                fx += 0.1 * (1 - (x + r))
            # Bottom boundary
            if y - r < 0:
                fy += 0.1 * (r - y)
            # Top boundary
            if y + r > 1:
                fy += 0.1 * (1 - (y + r))
                
            forces[i][0] += fx
            forces[i][1] += fy
        
        # Neighbor forces (avoid overlaps)
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):  # Only check upper triangle to avoid double counting
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < r1 + r2:  # Overlap detected
                    # Repulsive force magnitude
                    force_magnitude = 0.1 * (r1 + r2 - dist)
                    if dist > 0.001:  # Avoid division by zero
                        fx = force_magnitude * dx / dist
                        fy = force_magnitude * dy / dist
                    else:
                        # Random force if too close to avoid singularity
                        angle = np.random.uniform(0, 2*np.pi)
                        fx = force_magnitude * np.cos(angle)
                        fy = force_magnitude * np.sin(angle)
                    
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # Apply forces with adaptive step size
        step_size = 0.001 * (1.0 + 0.5 * np.sin(iteration/10.0))  # Slight oscillation for escape
        for i in range(n):
            # Apply force to position
            new_x = circles[i][0] + step_size * forces[i][0]
            new_y = circles[i][1] + step_size * forces[i][1]
            
            # Clamp to valid range
            r = circles[i][2]
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            circles[i][0] = new_x
            circles[i][1] = new_y
        
        # Update neighbor tree for next iteration
        neighbor_tree = cKDTree(circles[:, :2])
        
        # Check for convergence
        max_change = np.max(np.abs(prev_circles - circles))
        if max_change < 1e-6:
            break
    
    # Phase 3: Final Overlap Resolution
    # More aggressive overlap checking and resolution
    for _ in range(50):
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
                        # Use smaller reduction factor to preserve total sum
                        reduction = min(0.01, (r + r2 - dist) * 0.5)
                        new_r = max(0.001, r - reduction)
                        if new_r < r:
                            circles[i][2] = new_r
                            updated = True
                            
        if not updated:
            break
    
    # Final boundary enforcement
    for i in range(n):
        circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
        circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
    
    return circles

# EVOLVE-BLOCK-END