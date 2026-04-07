# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, let's try width=1.2, height=0.8
    width = 1.2
    height = 0.8
    
    # Start with a structured grid of potential centers
    n_circles = 21
    
    # Create a grid covering the rectangle with some padding
    grid_size = int(math.ceil(math.sqrt(n_circles)))
    x_coords = np.linspace(0.1, width - 0.1, grid_size)
    y_coords = np.linspace(0.1, height - 0.1, grid_size)
    
    # Generate initial candidate centers
    candidates = []
    for x in x_coords:
        for y in y_coords:
            if len(candidates) < n_circles:
                candidates.append([x, y])
    
    # Ensure we have exactly n_circles candidates
    if len(candidates) < n_circles:
        # Add more candidates randomly in valid region
        np.random.seed(42)
        while len(candidates) < n_circles:
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            candidates.append([x, y])
    
    candidates = np.array(candidates[:n_circles])
    
    # Greedy placement: place circles one by one with maximum possible radius
    placed_circles = []
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # First, let's try a greedy approach with initial candidates
    for i in range(n_circles):
        best_radius = 0
        best_center = None
        
        # For each candidate position, compute max radius it can have
        for candidate in candidates[i:]:
            cx, cy = candidate
            
            # Check constraints with already placed circles
            min_distance = float('inf')
            for pcx, pcy, pr in placed_circles:
                d = distance((cx, cy), (pcx, pcy))
                min_distance = min(min_distance, d)
            
            # Maximum radius is limited by boundaries and other circles
            boundary_radius = min(cx, width - cx, cy, height - cy)
            if len(placed_circles) > 0:
                max_radius = min(boundary_radius, min_distance - 0.001)  # Small buffer
            else:
                max_radius = boundary_radius
                
            if max_radius > best_radius:
                best_radius = max_radius
                best_center = (cx, cy)
        
        if best_center is not None and best_radius > 0:
            placed_circles.append((best_center[0], best_center[1], best_radius))
    
    # Now refine by optimizing radii using local optimization
    # Convert to array format
    final_circles = np.array(placed_circles)
    
    # Objective function to maximize sum of radii with constraints
    def objective(radii):
        # This is a simplified version - we'll focus on local optimization for radii
        return -np.sum(radii)  # Negative because we're minimizing
    
    def constraint_func(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        circles = []
        for i in range(0, len(params), 3):
            if i+2 < len(params):
                circles.append((params[i], params[i+1], params[i+2]))
        
        # Check boundary constraints
        penalty = 0
        for x, y, r in circles:
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty += 1000 * abs(x - r) + 1000 * abs(x + r - width) + 1000 * abs(y - r) + 1000 * abs(y + r - height)
        
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    penalty += 1000 * (r1 + r2 - dist)
        
        return penalty
    
    # Extract initial values for optimization
    initial_params = []
    for x, y, r in final_circles:
        initial_params.extend([x, y, r])
    
    # We'll do a simple iterative improvement approach
    # Since this needs to be fast and efficient, we'll focus on local optimization
    
    # Try a simpler approach - refine the existing solution
    refined_circles = final_circles.copy()
    
    # Simple iterative procedure: allow each circle to grow as much as possible without violating constraints
    max_iterations = 100
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase each circle's radius
        for i in range(len(refined_circles)):
            cx, cy, r = refined_circles[i]
            
            # Calculate the maximum possible radius for this circle
            min_dist = float('inf')
            
            # Distance to boundaries
            boundary_dist = min(cx, width-cx, cy, height-cy)
            
            # Distance to other circles
            for j in range(len(refined_circles)):
                if i != j:
                    cx2, cy2, r2 = refined_circles[j]
                    dist = math.sqrt((cx - cx2)**2 + (cy - cy2)**2)
                    min_dist = min(min_dist, dist)
            
            if min_dist > 0:
                max_radius = min(boundary_dist, min_dist - 0.0001)
                if max_radius > r and max_radius > 0.001:
                    refined_circles[i] = (cx, cy, max_radius)
                    improved = True
        
        # If no improvements, break early
        if not improved:
            break
    
    # Final cleanup
    final_result = np.zeros((21, 3))
    for i, (x, y, r) in enumerate(refined_circles):
        final_result[i] = [x, y, r]
        
    return final_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
