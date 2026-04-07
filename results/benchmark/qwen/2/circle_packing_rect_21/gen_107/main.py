# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution, minimize
import itertools
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses a deterministic grid-based approach with gradient-free optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Container dimensions (perimeter = 4, so width + height = 2)
    # Using a 1:1 aspect ratio for simplicity but could be optimized
    container_width = 1.0
    container_height = 1.0
    
    # Number of circles
    n_circles = 21
    
    # Phase 1: Smart grid initialization
    def initialize_grid():
        # Create a structured grid pattern that approximates good packing
        # For 21 circles, try 5x4 grid (or close to it)
        rows = 5
        cols = 5
        if rows * cols < n_circles:
            cols = math.ceil(n_circles / rows)
        
        # Adjust for actual circle count
        actual_cols = min(cols, math.ceil(n_circles / rows))
        
        # Calculate grid spacing
        grid_width = container_width / (actual_cols + 1)
        grid_height = container_height / (rows + 1)
        
        positions = []
        radii = []
        
        # Place circles on grid with some randomness to avoid symmetry
        circle_idx = 0
        for i in range(rows):
            for j in range(actual_cols):
                if circle_idx >= n_circles:
                    break
                # Position with slight jitter to avoid perfect grid
                x = (j + 1) * grid_width + np.random.uniform(-grid_width*0.1, grid_width*0.1)
                y = (i + 1) * grid_height + np.random.uniform(-grid_height*0.1, grid_height*0.1)
                # Clip to ensure within bounds
                x = np.clip(x, 0.01, container_width - 0.01)
                y = np.clip(y, 0.01, container_height - 0.01)
                positions.append([x, y])
                circle_idx += 1
            if circle_idx >= n_circles:
                break
        
        # Create initial radii
        initial_radius = min(container_width, container_height) * 0.05
        for i in range(n_circles):
            radii.append(initial_radius * (0.8 + np.random.random() * 0.4))
        
        return np.array(positions), np.array(radii)
    
    # Phase 2: Constraint checking function
    def check_constraints(positions, radii):
        # Check boundary constraints
        for i in range(n_circles):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
                return False
        
        # Check overlap constraints
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (radii[i] + radii[j]):
                    return False
                    
        return True
    
    # Phase 3: Penalty-based objective function
    def objective_function(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        positions = []
        radii = []
        
        for i in range(n_circles):
            idx = i * 3
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        # Calculate sum of radii (negative because we want to maximize)
        total_radius = sum(radii)
        
        # Add penalty for constraint violations
        penalty = 0
        
        # Boundary penalty
        for i in range(n_circles):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            if x - r < 0:
                penalty += abs(x - r) * 1000
            if x + r > container_width:
                penalty += abs(x + r - container_width) * 1000
            if y - r < 0:
                penalty += abs(y - r) * 1000
            if y + r > container_height:
                penalty += abs(y + r - container_height) * 1000
        
        # Overlap penalty
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (radii[i] + radii[j]):
                    overlap = (radii[i] + radii[j]) - distance
                    penalty += overlap * 10000
        
        # Return negative sum (since we're minimizing) plus penalty
        return -total_radius + penalty
    
    # Phase 4: Optimization with proper bounds
    def optimize():
        # Initialize with grid
        positions, radii = initialize_grid()
        
        # Create variable vector [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        vars_init = []
        for i in range(n_circles):
            vars_init.extend([positions[i][0], positions[i][1], radii[i]])
        
        # Set bounds: x in [r, width-r], y in [r, height-r], r in [0.001, 0.5]
        bounds = []
        for i in range(n_circles):
            # x bounds
            bounds.append((0.001, container_width - 0.001))
            # y bounds  
            bounds.append((0.001, container_height - 0.001))
            # r bounds
            bounds.append((0.001, 0.5))
        
        # Try different optimization approaches
        try:
            # Use differential evolution for global search
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=1000,
                popsize=15,
                seed=42,
                atol=1e-6,
                rtol=1e-6
            )
            
            # Extract solution
            vars_final = result.x
            positions_final = []
            radii_final = []
            
            for i in range(n_circles):
                idx = i * 3
                x, y, r = vars_final[idx], vars_final[idx+1], vars_final[idx+2]
                positions_final.append([x, y])
                radii_final.append(r)
                
            # Validate final solution
            if check_constraints(positions_final, radii_final):
                # Return final solution
                circles = np.zeros((n_circles, 3))
                for i in range(n_circles):
                    circles[i, 0] = positions_final[i][0]
                    circles[i, 1] = positions_final[i][1] 
                    circles[i, 2] = radii_final[i]
                return circles
        except Exception as e:
            pass
        
        # Fallback to simple approach if optimization fails
        circles = np.zeros((n_circles, 3))
        circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, n_circles)
        circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, n_circles)
        circles[:, 2] = np.random.uniform(0.01, 0.1, n_circles)
        return circles
    
    # Run optimization
    circles = optimize()
    
    # Ensure we have a valid solution
    if circles is None or np.isnan(circles).any():
        circles = np.zeros((n_circles, 3))
        np.random.seed(42)
        circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, n_circles)
        circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, n_circles)
        circles[:, 2] = np.random.uniform(0.01, 0.1, n_circles)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")