# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
import time
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a greedy insertion approach with geometric optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    n = 26
    circles = np.zeros((n, 3))
    
    # Helper function to check if a circle fits at a given position
    def is_valid_position(x, y, r, existing_circles):
        if r <= x <= 1-r and r <= y <= 1-r:
            # Check for overlaps with existing circles
            for ox, oy, oradius in existing_circles:
                dx = x - ox
                dy = y - oy
                distance_sq = dx*dx + dy*dy
                min_distance_sq = (r + oradius)**2
                if distance_sq < min_distance_sq:
                    return False
            return True
        return False
    
    # Helper function to compute maximum radius at a position
    def compute_max_radius(x, y, existing_circles):
        # Compute maximum radius constrained by boundaries
        max_radius = min(x, 1-x, y, 1-y)
        
        # Reduce radius based on existing circles
        for ox, oy, oradius in existing_circles:
            dx = x - ox
            dy = y - oy
            distance = math.sqrt(dx*dx + dy*dy)
            # Maximum radius such that new circle doesn't overlap
            max_radius = min(max_radius, distance - oradius)
            
        return max(0, max_radius)
    
    # Helper function to find best position for a new circle
    def find_best_position_and_radius(existing_circles):
        if len(existing_circles) == 0:
            # Place first circle at center with maximum possible radius
            return 0.5, 0.5, 0.5
            
        # Strategy: sample potential positions around existing circles
        best_x, best_y, best_r = None, None, 0
        best_score = -1
        
        # Sample from various regions
        samples_per_region = 100
        regions = [
            # Center area
            [(0.25, 0.25), (0.75, 0.75)],
            # Top-left
            [(0.1, 0.1), (0.4, 0.4)],
            # Top-right
            [(0.6, 0.1), (0.9, 0.4)],
            # Bottom-left
            [(0.1, 0.6), (0.4, 0.9)],
            # Bottom-right
            [(0.6, 0.6), (0.9, 0.9)]
        ]
        
        for (min_x, min_y), (max_x, max_y) in regions:
            for _ in range(samples_per_region):
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                max_r = compute_max_radius(x, y, existing_circles)
                if max_r > 0 and max_r > best_r:
                    # Refine the position for this radius
                    r = max_r
                    if is_valid_position(x, y, r, existing_circles):
                        score = r  # We want to maximize radius sum
                        if score > best_score:
                            best_score = score
                            best_x, best_y, best_r = x, y, r
        
        # If we couldn't find a better position, try to place at boundary
        if best_x is None:
            # Try placing at boundary
            candidates = []
            for i in range(200):
                # Random boundary points
                side = np.random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    x = np.random.uniform(0.05, 0.95)
                    y = 1 - 0.05
                elif side == 'bottom':
                    x = np.random.uniform(0.05, 0.95)
                    y = 0.05
                elif side == 'left':
                    x = 0.05
                    y = np.random.uniform(0.05, 0.95)
                else:  # right
                    x = 1 - 0.05
                    y = np.random.uniform(0.05, 0.95)
                
                max_r = compute_max_radius(x, y, existing_circles)
                if max_r > 0:
                    candidates.append((x, y, max_r))
            
            if candidates:
                # Pick the one with maximum radius
                best_x, best_y, best_r = max(candidates, key=lambda p: p[2])
        
        if best_x is None:
            # Last resort - place at a corner or center
            best_x = 0.5
            best_y = 0.5
            best_r = compute_max_radius(0.5, 0.5, existing_circles)
            
        return best_x, best_y, best_r
    
    # Greedy insertion approach
    for i in range(n):
        # Find best position for the next circle
        best_x, best_y, best_r = find_best_position_and_radius(circles[:i])
        
        # Validate and update
        if best_r > 0 and is_valid_position(best_x, best_y, best_r, circles[:i]):
            circles[i] = [best_x, best_y, best_r]
        else:
            # Fallback: place at center with tiny radius
            circles[i] = [0.5, 0.5, 0.001]
    
    # Post-processing: local optimization to improve total radius sum
    def improve_solution(circles_array, iterations=50):
        for _ in range(iterations):
            improved = False
            # Try to increase each radius slightly
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                if r <= 0.001:
                    continue
                    
                # Compute max possible radius for this circle
                max_radius = compute_max_radius(x, y, circles_array[:i])
                if i+1 < len(circles_array):
                    max_radius = min(max_radius, compute_max_radius(x, y, circles_array[i+1:]))
                
                # Try to increase radius if possible
                if max_radius > r:
                    # Try to increase radius up to max_radius
                    new_r = min(r + 0.005, max_radius)
                    # Check if this works
                    valid = True
                    for j in range(len(circles_array)):
                        if i != j:
                            dx = x - circles_array[j, 0]
                            dy = y - circles_array[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (new_r + circles_array[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid and new_r <= 1-x and new_r <= x and new_r <= 1-y and new_r <= y:
                        circles_array[i, 2] = new_r
                        improved = True
            
            if not improved:
                break
                
        return circles_array
    
    # Apply local optimization
    circles = improve_solution(circles.copy())
    
    return circles

# EVOLVE-BLOCK-END