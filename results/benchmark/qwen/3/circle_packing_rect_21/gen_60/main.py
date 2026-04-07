# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Test multiple aspect ratios to find optimal configuration
    test_ratios = [(1.2, 0.8), (1.5, 0.5), (1.0, 1.0)]
    best_sum = 0
    best_circles = None
    
    # Multi-start approach to find best configuration
    for width, height in test_ratios:
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Initialize with adaptive grid refinement similar to the third approach
        circles = initialize_adaptive_layout(width, height, 21)
        
        # Optimize using multi-stage approach with simulated annealing inspiration
        optimized_circles = optimize_layout(circles, width, height)
        
        # Check if this is better than our previous best
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
    
    return best_circles

def initialize_adaptive_layout(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles using adaptive grid refinement with hexagonal pattern."""
    # Start with a more sophisticated hexagonal grid pattern
    rows = 5
    cols = 5
    
    # Create hexagonal grid points with proper spacing
    hex_points = []
    radius_estimate = min(width, height) * 0.1
    
    # Generate hexagonal lattice
    for i in range(rows):
        for j in range(cols):
            if len(hex_points) >= n:
                break
            # Hexagonal coordinate calculation
            x = j * 2 * radius_estimate + (i % 2) * radius_estimate
            y = i * np.sqrt(3) * radius_estimate
            
            # Add if within bounds
            if 0 <= x <= width and 0 <= y <= height:
                hex_points.append([x, y])
        
        if len(hex_points) >= n:
            break
    
    # Fill remaining positions with random placements if needed
    while len(hex_points) < n:
        x = np.random.uniform(0.1, width - 0.1)
        y = np.random.uniform(0.1, height - 0.1)
        hex_points.append([x, y])
    
    # Take first n points
    points = np.array(hex_points[:n])
    
    # Initialize circles with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [points[i][0], points[i][1], 0.01]
    
    # Refine initial placement by computing actual maximum radius for each position
    for i in range(n):
        x, y, _ = circles[i]
        max_radius = compute_max_radius(x, y, width, height, circles[:i])
        circles[i] = [x, y, max_radius]
    
    return circles

def compute_max_radius(x: float, y: float, width: float, height: float, existing_circles: np.ndarray) -> float:
    """Compute maximum radius for a circle at (x,y) given existing circles and container boundaries."""
    # Boundary constraints
    min_dist_from_edge = min(x, width - x, y, height - y)
    
    if min_dist_from_edge <= 0:
        return 0
    
    # Overlap constraints with existing circles
    min_dist_from_others = float('inf')
    
    for circle in existing_circles:
        if circle[2] > 0:  # Only consider placed circles
            cx, cy, cr = circle
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            min_dist_from_others = min(min_dist_from_others, dist - cr)
    
    # Take minimum of boundary and overlap constraints
    max_radius = min(min_dist_from_edge, min_dist_from_others)
    
    return max(0, max_radius)

def optimize_layout(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize layout using multi-stage approach with progressive refinement."""
    current_circles = circles.copy()
    
    # Stage 1: Coarse optimization with large steps
    for iteration in range(200):
        improved = False
        # Shuffle circle order for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)
        
        for i in indices:
            max_radius = compute_max_radius(current_circles[i][0], current_circles[i][1], 
                                          width, height, 
                                          np.vstack([current_circles[:i], current_circles[i+1:]]))
            
            if max_radius > current_circles[i][2] + 1e-6:
                current_circles[i][2] = max_radius
                improved = True
                
        if not improved:
            break
    
    # Stage 2: Fine-grained local search with grid-based optimization
    for iteration in range(300):
        improved = False
        # Try each circle in shuffled order
        indices = list(range(len(current_circles)))
        random.shuffle(indices)
        
        for i in indices:
            current_x, current_y, current_r = current_circles[i]
            
            # Try to improve position by searching nearby locations
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r
            
            # Grid search around current position
            step_sizes = [0.1, 0.05, 0.02] if iteration < 150 else [0.05, 0.02, 0.01]
            
            for step in step_sizes:
                for dx in [-step, -step/2, 0, step/2, step]:
                    for dy in [-step, -step/2, 0, step/2, step]:
                        new_x = current_x + dx
                        new_y = current_y + dy
                        
                        # Check bounds
                        if (0.01 <= new_x <= width - 0.01 and 
                            0.01 <= new_y <= height - 0.01):
                            
                            # Compute max radius at new position
                            max_radius = compute_max_radius(new_x, new_y, width, height,
                                                          np.vstack([current_circles[:i], current_circles[i+1:]]))
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_pos = [new_x, new_y, max_radius]
            
            if best_pos[2] > current_circles[i][2] + 1e-6:
                current_circles[i] = best_pos
                improved = True
                
        if not improved:
            break
    
    # Stage 3: Simulated annealing inspired approach for further improvement
    current_sum = np.sum(current_circles[:, 2])
    temp = 0.1
    cooling_rate = 0.999
    min_temp = 0.001
    max_iterations = 500
    
    for iteration in range(max_iterations):
        if temp < min_temp:
            break
            
        # Try to improve the solution
        new_circles = current_circles.copy()
        
        # Choose a random circle to modify
        idx = np.random.randint(0, len(new_circles))
        
        # Perturb the circle's position slightly
        old_x, old_y, old_r = new_circles[idx]
        
        # Try to move circle with some probability
        if np.random.random() < 0.7:
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Ensure it stays within bounds
            new_x = np.clip(new_x, 0.01, width - 0.01)
            new_y = np.clip(new_y, 0.01, height - 0.01)
            
            # Compute new radius
            new_r = compute_max_radius(new_x, new_y, width, height, 
                                    np.vstack([new_circles[:idx], new_circles[idx+1:]]))
            
            if new_r > 0:
                new_circles[idx] = [new_x, new_y, new_r]
        
        else:
            # Or try to change just the radius (but make it more conservative)
            old_r = new_circles[idx][2]
            delta_r = np.random.uniform(-0.05, 0.05)
            new_r = old_r + delta_r
            if new_r > 0.01:
                new_circles[idx][2] = new_r
        
        # Check constraints and accept or reject
        if is_valid_solution(new_circles, width, height):
            new_sum = np.sum(new_circles[:, 2])
            delta = new_sum - current_sum
            
            # Accept if better or with probability based on temperature
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_circles = new_circles
                current_sum = new_sum
                
        # Cool down temperature
        temp *= cooling_rate
    
    return current_circles

def is_valid_solution(circles: np.ndarray, width: float, height: float) -> bool:
    """Check if the solution satisfies all constraints."""
    # Check boundary constraints
    for circle in circles:
        x, y, r = circle
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False
    
    # Check overlap constraints
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            ci = circles[i]
            cj = circles[j]
            if ci[2] > 0 and cj[2] > 0:
                dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                if dist < ci[2] + cj[2]:
                    return False
    
    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")