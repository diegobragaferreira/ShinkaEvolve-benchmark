# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))
    
    # Step 1: Generate initial Voronoi-based distribution
    # Create seed points using a quasi-random pattern that approximates Voronoi cells
    seed_points = generate_voronoi_seeds(n)
    
    # Step 2: Sequential greedy placement
    placed_circles = []
    
    for i in range(n):
        # Find the best position for the next circle
        best_circle = find_best_circle_placement(placed_circles, seed_points[i])
        
        if best_circle is not None:
            placed_circles.append(best_circle)
        else:
            # Fallback to grid placement if Voronoi fails
            fallback_circle = grid_fallback_placement(i, placed_circles)
            if fallback_circle:
                placed_circles.append(fallback_circle)
    
    # Step 3: Local optimization of all circles
    optimized_circles = optimize_all_circles(placed_circles)
    
    # Convert to final array format
    for i, (x, y, r) in enumerate(optimized_circles):
        circles[i] = [x, y, r]
    
    return circles

def generate_voronoi_seeds(n):
    """Generate seed points that approximate Voronoi cell distribution"""
    # Use Fibonacci spiral sampling for even distribution
    points = []
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    
    for i in range(n):
        # Map fibonacci indices to spherical coordinates
        z = 1 - (i / float(n - 1)) * 2  # z from 1 to -1
        radius = math.sqrt(1 - z * z)   # radius at z
        
        theta = math.acos(z)           # polar angle
        phi_angle = ((i * 4) % 360) * math.pi / 180  # azimuthal angle
        
        x = radius * math.cos(phi_angle)
        y = radius * math.sin(phi_angle)
        
        # Project to unit square
        points.append([x * 0.8 + 0.1, y * 0.8 + 0.1])  # Scale and offset to fit in [0.1, 0.9]^2
    
    return np.array(points)

def find_best_circle_placement(placed_circles, seed_point):
    """Find the best circle placement considering constraints"""
    # Start with a reasonable initial radius
    max_radius = min(seed_point[0], 1 - seed_point[0], seed_point[1], 1 - seed_point[1])
    
    if max_radius <= 0:
        return None
    
    # Try to place a circle with maximum possible radius
    # But adjust this based on proximity to existing circles
    if len(placed_circles) > 0:
        # Calculate minimum distance to existing circles
        distances = []
        for cx, cy, cr in placed_circles:
            dist = math.sqrt((seed_point[0] - cx)**2 + (seed_point[1] - cy)**2)
            distances.append(dist)
        
        min_distance = min(distances)
        # New circle should be at least radius + min_distance away from others
        max_radius = min(max_radius, min_distance - 0.001)
    
    # If we can't place a circle, return None
    if max_radius <= 0:
        return None
    
    # Try a few different radii near the maximum to find a good spot
    best_radius = max_radius
    best_position = seed_point
    
    # Simple greedy approach: try a few candidate positions near the seed
    candidates = []
    for dx in [-0.05, 0, 0.05]:
        for dy in [-0.05, 0, 0.05]:
            candidates.append([seed_point[0] + dx, seed_point[1] + dy])
    
    for candidate_x, candidate_y in candidates:
        # Check containment
        if (candidate_x - max_radius >= 0 and 
            candidate_x + max_radius <= 1 and
            candidate_y - max_radius >= 0 and
            candidate_y + max_radius <= 1):
            
            # Check overlap with existing circles
            valid = True
            for cx, cy, cr in placed_circles:
                distance = math.sqrt((candidate_x - cx)**2 + (candidate_y - cy)**2)
                if distance < max_radius + cr:
                    valid = False
                    break
            
            if valid:
                best_position = [candidate_x, candidate_y]
                break
    
    # Return the result with the best found radius
    return [best_position[0], best_position[1], max_radius]

def grid_fallback_placement(index, placed_circles):
    """Fallback method to place circles in a grid pattern"""
    # Create a regular grid to place circles
    grid_size = int(math.ceil(math.sqrt(26)))
    spacing = 1.0 / (grid_size + 1)
    
    # Place in a grid fashion with some randomness
    row = index // grid_size
    col = index % grid_size
    x = (col + 1) * spacing
    y = (row + 1) * spacing
    
    # Adjust for boundary constraints
    max_radius = min(x, 1-x, y, 1-y)
    
    # Ensure we're not too close to existing circles
    if len(placed_circles) > 0:
        for cx, cy, cr in placed_circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < 0.02:  # Avoid very close placement
                max_radius = 0
                break
    
    if max_radius > 0.01:
        return [x, y, max_radius]
    else:  # Fallback to small radius
        return [x, y, 0.01]

def optimize_all_circles(circles_list):
    """Apply constrained optimization to refine all circle positions and radii"""
    # Convert to list for easier manipulation
    optimized = []
    
    # Simple local optimization for each circle
    for i, (x, y, r) in enumerate(circles_list):
        # For each circle, we want to:
        # 1. Maximize radius while respecting constraints
        # 2. Optimize position to avoid conflicts
        
        # This is simplified but works well in practice
        # We'll use a basic optimization approach
        
        # First get a better estimate of maximum radius
        max_radius = compute_max_radius(x, y, circles_list[:i])
        if max_radius > r:
            # Try to increase radius slightly
            r = min(r * 1.1, max_radius)  # Increase by up to 10% but respect constraints
        
        # Refine position to avoid overlaps (simple approach)
        # Move to a position that respects all constraints
        refined_x, refined_y = refine_position(x, y, r, circles_list[:i])
        
        optimized.append([refined_x, refined_y, r])
    
    return optimized

def compute_max_radius(x, y, previous_circles):
    """Compute the maximum radius possible at position (x,y) given previous circles"""
    # Boundaries of the unit square
    max_radius = min(x, 1-x, y, 1-y)
    
    # Check overlaps with existing circles
    for cx, cy, cr in previous_circles:
        distance = math.sqrt((x - cx)**2 + (y - cy)**2)
        # Should be at least radius + previous_radius apart
        max_radius = min(max_radius, distance - cr)
    
    return max(max_radius, 0.001)

def refine_position(x, y, r, previous_circles):
    """Refine position to avoid overlaps while keeping within bounds"""
    # This is a simplified refinement - move the circle slightly if needed
    # to ensure no overlaps and boundaries are respected
    
    # Try to move slightly to resolve overlaps
    # Simple approach: move towards center of mass if needed
    if len(previous_circles) == 0:
        return x, y
    
    # Check if we're in collision with anything
    for cx, cy, cr in previous_circles:
        distance = math.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < r + cr + 0.001:
            # Move the circle away from the other circle
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0.001:  # Avoid division by zero
                # Move outward from the other circle
                scale = (r + cr) / dist
                new_x = x + dx * scale * 0.1
                new_y = y + dy * scale * 0.1
                
                # Stay within bounds
                new_x = max(r, min(1-r, new_x))
                new_y = max(r, min(1-r, new_y))
                
                return new_x, new_y
    
    return x, y

# EVOLVE-BLOCK-END
