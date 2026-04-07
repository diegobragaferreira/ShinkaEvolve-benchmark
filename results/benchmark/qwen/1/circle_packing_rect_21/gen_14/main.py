# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
import random
from typing import Tuple
import time

def distance_matrix(circles):
    """Calculate pairwise distances between circle centers"""
    centers = circles[:, :2]
    return cdist(centers, centers)

def check_constraints(circles, rect_width=1.0, rect_height=1.0):
    """Check if all circles are within bounds and non-overlapping"""
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Check overlap constraints
    dists = distance_matrix(circles)
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if dists[i,j] < circles[i,2] + circles[j,2]:
                return False
    return True

def calculate_sum_radii(circles):
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def initialize_hexagonal_grid(n_circles, rect_width=1.0, rect_height=1.0):
    """Initialize circles using a hexagonal lattice pattern"""
    # For 21 circles, we'll create a triangular lattice
    # Calculate approximate spacing
    max_radius = min(rect_width, rect_height) / 4.0
    rows = int(np.sqrt(n_circles))
    cols = int(n_circles / rows) + 1
    
    # Adjust spacing to fit within rectangle
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    # Create a hexagonal pattern
    circles = []
    radius = max_radius * 0.8  # Leave some margin
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
            y = (i + 1) * spacing_y
            
            # Only add if within bounds
            if x - radius > 0 and x + radius < rect_width and y - radius > 0 and y + radius < rect_height:
                circles.append([x, y, radius])
    
    # Fill remaining circles with smaller radii
    while len(circles) < n_circles:
        # Place randomly with small radius
        x = random.uniform(radius, rect_width - radius)
        y = random.uniform(radius, rect_height - radius)
        radius = min(0.05, max_radius * 0.5)
        circles.append([x, y, radius])
        
    return np.array(circles)

def local_refinement(circles, rect_width=1.0, rect_height=1.0, iterations=100):
    """Use physics-inspired local refinement"""
    def force_magnitude(distance, r1, r2):
        # Repulsive force between overlapping circles
        if distance < r1 + r2:
            return 1.0 / (distance + 1e-8)  # Prevent division by zero
        return 0.0
    
    def boundary_force(x, y, r, rect_width, rect_height):
        # Force away from boundaries
        fx = 0.0
        fy = 0.0
        
        # Left/right boundaries
        if x - r < 0:
            fx += 1.0 / (x - r + 1e-8)
        elif x + r > rect_width:
            fx -= 1.0 / (x + r - rect_width + 1e-8)
            
        # Top/bottom boundaries
        if y - r < 0:
            fy += 1.0 / (y - r + 1e-8)
        elif y + r > rect_height:
            fy -= 1.0 / (y + r - rect_height + 1e-8)
            
        return fx, fy
    
    # Simple gradient ascent approach
    for _ in range(iterations):
        new_circles = circles.copy()
        for i in range(len(circles)):
            x, y, r = circles[i]
            total_fx, total_fy = 0.0, 0.0
            
            # Add repulsion forces from other circles
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    force = force_magnitude(distance, r, r2)
                    if force > 0:
                        total_fx += force * dx / (distance + 1e-8)
                        total_fy += force * dy / (distance + 1e-8)
            
            # Add boundary forces
            bf_x, bf_y = boundary_force(x, y, r, rect_width, rect_height)
            total_fx += bf_x
            total_fy += bf_y
            
            # Update position (with some damping)
            step_size = 0.01
            new_x = x + step_size * total_fx
            new_y = y + step_size * total_fy
            
            # Ensure new position still allows for radius
            if (new_x - r > 0 and new_x + r < rect_width and 
                new_y - r > 0 and new_y + r < rect_height):
                new_circles[i, 0] = new_x
                new_circles[i, 1] = new_y
                
        circles = new_circles
    
    return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Rectangle dimensions with perimeter 4 implies width + height = 2
    # Using a square (1x1) for simplicity, but we might try other ratios
    rect_width = 1.0
    rect_height = 1.0
    
    # Initialize with hexagonal grid
    circles = initialize_hexagonal_grid(21, rect_width, rect_height)
    
    # Refine using local optimization
    circles = local_refinement(circles, rect_width, rect_height, 50)
    
    # Evolve with a simple hill climbing approach
    best_sum = calculate_sum_radii(circles)
    best_solution = circles.copy()
    
    # Simple evolutionary improvement
    for iteration in range(100):
        # Create a new candidate by slightly modifying existing solution
        new_circles = circles.copy()
        
        # Randomly select one circle to modify
        idx = random.randint(0, 20)
        # Slightly change position and/or radius
        new_circles[idx, 0] += random.uniform(-0.02, 0.02)
        new_circles[idx, 1] += random.uniform(-0.02, 0.02)
        new_circles[idx, 2] *= random.uniform(0.95, 1.05)
        
        # Ensure within bounds
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], new_circles[idx, 2], rect_width - new_circles[idx, 2])
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], new_circles[idx, 2], rect_height - new_circles[idx, 2])
        
        # Try to grow circle radius if possible
        if check_constraints(new_circles, rect_width, rect_height):
            # Try to increase radius slightly
            original_radius = new_circles[idx, 2]
            for attempt in range(10):
                new_circles[idx, 2] += 0.005
                if not check_constraints(new_circles, rect_width, rect_height):
                    new_circles[idx, 2] = original_radius
                    break
            
            current_sum = calculate_sum_radii(new_circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = new_circles.copy()
                
        circles = best_solution.copy()
    
    # Final local refinement
    circles = local_refinement(best_solution, rect_width, rect_height, 100)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
