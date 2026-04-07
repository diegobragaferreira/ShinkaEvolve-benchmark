# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
import random

# Physics-based circle packing solver
def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # Optimized ratio based on prior analysis
    width, height = 1.3, 0.7
    
    # Physics simulation parameters
    n_circles = 21
    max_iterations = 1000
    
    # Initialize circles with adaptive grid
    circles = np.zeros((n_circles, 3))
    
    # Grid-based initialization
    cols = max(1, int(np.ceil(np.sqrt(n_circles * (width/height)))))
    rows = max(1, int(np.ceil(n_circles / cols)))
    
    if cols * rows < n_circles:
        cols = max(cols, int(np.ceil(n_circles / rows)))
    
    cell_width = width / cols
    cell_height = height / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            # Grid placement with jittering
            x = (j + 0.5) * cell_width + random.uniform(-cell_width*0.1, cell_width*0.1)
            y = (i + 0.5) * cell_height + random.uniform(-cell_height*0.1, cell_height*0.1)
            
            # Keep within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Initial radius based on available space
            max_radius = min(x, width-x, y, height-y) * 0.3
            r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))
            
            circles[idx] = [x, y, r]
            idx += 1
    
    # Physics simulation with gradient descent
    circles = physics_based_optimization(circles, width, height, max_iterations)
    
    return circles

def physics_based_optimization(circles, width, height, max_iter):
    """
    Physics-based optimization using force simulation
    """
    n_circles = len(circles)
    learning_rate = 0.1
    damping = 0.95
    epsilon = 1e-8
    
    # Precompute distances for efficiency
    force_matrix = np.zeros((n_circles, n_circles))
    
    for iteration in range(max_iter):
        # Calculate forces between circles
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Repulsive forces between overlapping circles
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Only consider if circles could potentially overlap
                    max_dist = radii[i] + radii[j]
                    if distance < max_dist:
                        # Repulsive force (inverse square law)
                        force_magnitude = 1.0 / (distance*distance + epsilon)
                        
                        # Direction from j to i
                        if distance > epsilon:
                            fx = force_magnitude * dx / distance
                            fy = force_magnitude * dy / distance
                        else:
                            fx = random.uniform(-1, 1)
                            fy = random.uniform(-1, 1)
                        
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
        
        # Boundary forces (spring-like constraints)
        for i in range(n_circles):
            x, y, r = circles[i]
            # Force away from boundaries
            force_x = 0
            force_y = 0
            
            # Left boundary
            if x - r < 0:
                force_x += 100 * (r - x)
            # Right boundary
            if x + r > width:
                force_x += 100 * (width - x - r)
            # Bottom boundary
            if y - r < 0:
                force_y += 100 * (r - y)
            # Top boundary
            if y + r > height:
                force_y += 100 * (height - y - r)
            
            forces[i, 0] += force_x
            forces[i, 1] += force_y
        
        # Update positions (Verlet integration style)
        for i in range(n_circles):
            # Apply forces
            circles[i, 0] += forces[i, 0] * learning_rate
            circles[i, 1] += forces[i, 1] * learning_rate
            
            # Apply damping
            learning_rate *= damping
            
            # Keep within bounds
            circles[i, 0] = max(circles[i, 2], min(width - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(height - circles[i, 2], circles[i, 1]))
        
        # Optional: radius optimization step to maximize sum
        # Only do this every few iterations to avoid oscillation
        if iteration % 5 == 0:
            # Try to increase radii while maintaining non-overlap
            for i in range(n_circles):
                # Try to increase radius while not violating constraints
                current_x, current_y, current_r = circles[i]
                new_r = current_r
                
                # Find maximum possible radius
                max_r = min(current_x, width-current_x, current_y, height-current_y)
                
                # Check overlap with other circles
                for j in range(n_circles):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dx = current_x - x2
                        dy = current_y - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Maximum radius without overlap
                        if distance < (current_r + r2):
                            max_r = min(max_r, distance - r2)
                
                # Increase radius if beneficial
                if max_r > current_r and max_r > 0.001:
                    # Conservative increase
                    new_r = min(max_r, current_r + (max_r - current_r) * 0.1)
                    circles[i, 2] = new_r
        
        # Early stopping if forces are small
        force_magnitude = np.linalg.norm(forces)
        if force_magnitude < 1e-4 and iteration > 100:
            break
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")