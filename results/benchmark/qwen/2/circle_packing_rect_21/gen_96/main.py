# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Optimize rectangle aspect ratio for better packing efficiency
    # Using 2:1 ratio (width:height) which typically works well for circle packing
    rect_width = 1.3333333333333333  # 2/3
    rect_height = 0.6666666666666666  # 1/3

    # Number of circles
    n = 21

    def objective(x):
        # x contains [cx1, cy1, r1, cx2, cy2, r2, ..., cxn, cyn, rn]
        circles = x.reshape(-1, 3)
        # Calculate sum of radii (we want to maximize this)
        total_radius = np.sum(circles[:, 2])
        # Return negative because we're minimizing in scipy
        return -total_radius

    def penalty_function(x):
        """Calculate penalty for constraint violations"""
        circles = x.reshape(-1, 3)
        
        penalty = 0
        
        # Boundary penalties (penalize if circles go outside)
        for i in range(n):
            cx, cy, r = circles[i]
            # Penalty for going outside
            if cx - r < 0:
                penalty += 1000 * (r - cx)**2
            if cx + r > rect_width:
                penalty += 1000 * (cx + r - rect_width)**2
            if cy - r < 0:
                penalty += 1000 * (r - cy)**2
            if cy + r > rect_height:
                penalty += 1000 * (cy + r - rect_height)**2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, r1 = circles[i]
                cx2, cy2, r2 = circles[j]
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                overlap = (r1 + r2) - dist
                
                if overlap > 0:  # Overlapping
                    penalty += 10000 * overlap**2
        
        return penalty

    def fast_overlap_check(circles, tree, i, new_radius):
        """Fast overlap check using spatial indexing"""
        cx, cy, _ = circles[i]
        # Query nearby circles within 2*(max_radius) distance
        nearby_indices = tree.query_ball_point([cx, cy], 2 * new_radius)
        
        for j in nearby_indices:
            if i != j:
                other_cx, other_cy, other_r = circles[j]
                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                if dist < new_radius + other_r:
                    return False
        return True

    # Generate initial guess with enhanced hexagonal arrangement
    def generate_initial_guess():
        circles = np.zeros((n, 3))
        
        # Use hexagonal packing for better initial distribution
        # Calculate optimal grid dimensions for hexagonal packing
        rows = int(np.ceil(np.sqrt(n * 1.5)))  # Slightly denser than square grid
        cols = int(np.ceil(n / rows))
        
        # Ensure minimum grid size for stability
        rows = max(rows, 4)
        cols = max(cols, 4)
        
        # Calculate spacing based on rectangle and grid dimensions
        spacing_x = rect_width / (cols + 1) * 0.9  # Small margin
        spacing_y = rect_height / (rows + 1) * 0.9  # Small margin
        
        # Hexagon packing factor (sqrt(3)/2 ≈ 0.866)
        hex_spacing_x = spacing_x * 0.8
        hex_spacing_y = spacing_y * 0.866
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * (hex_spacing_x / 2)
                x = hex_spacing_x * j + x_offset + hex_spacing_x
                y = hex_spacing_y * i + hex_spacing_y
                
                # Ensure within bounds with safety margin
                x = max(hex_spacing_x, min(rect_width - hex_spacing_x, x))
                y = max(hex_spacing_y, min(rect_height - hex_spacing_y, y))
                
                # Initialize with a reasonable starting radius
                # Base radius on spacing with safety factor
                base_radius = min(hex_spacing_x, hex_spacing_y) * 0.3
                
                circles[idx] = [x, y, base_radius]
                idx += 1
                if idx >= n:
                    break
        
        # Fill remaining slots if needed with better distribution
        if idx < n:
            for i in range(idx, n):
                # Better random positioning with clustering avoidance
                x = np.random.uniform(hex_spacing_x, rect_width - hex_spacing_x)
                y = np.random.uniform(hex_spacing_y, rect_height - hex_spacing_y)
                base_radius = min(hex_spacing_x, hex_spacing_y) * 0.3
                circles[i] = [x, y, base_radius]
        
        return circles.flatten()

    # Start with a good initial configuration
    initial_guess = generate_initial_guess()

    # Run multi-stage optimization for better results
    try:
        # Stage 1: Coarse global optimization with relaxed constraints
        # Using COBYLA method which handles constraints better in early stages
        combined_objective_1 = lambda x: objective(x) + penalty_function(x)
        
        result1 = minimize(
            combined_objective_1,
            initial_guess,
            method='COBYLA',
            options={'maxiter': 800, 'disp': False, 'catol': 1e-6}
        )
        
        if result1.success:
            stage1_solution = result1.x.reshape(-1, 3)
        else:
            stage1_solution = initial_guess.reshape(-1, 3)
            
        # Stage 2: Fine-grained local optimization with stricter constraints
        # Use L-BFGS-B for more precise optimization
        combined_objective_2 = lambda x: objective(x) + penalty_function(x)
        
        result2 = minimize(
            combined_objective_2,
            stage1_solution.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 800, 'disp': False}
        )
        
        if result2.success:
            final_circles = result2.x.reshape(-1, 3)
        else:
            final_circles = stage1_solution
            
    except Exception as e:
        # Fallback to initial guess if anything goes wrong
        final_circles = initial_guess.reshape(-1, 3)

    # Final adjustment using advanced greedy refinement
    # This is a critical refinement step that maximizes sum of radii
    refined_circles = final_circles.copy()
    
    # Build spatial index for efficient overlap checking
    tree = cKDTree(refined_circles[:, :2])
    
    # Adaptive refinement with better constraint validation and smarter steps
    max_iter = 300
    improvement_threshold = 1e-7
    
    for iteration in range(max_iter):
        improved = False
        total_improvement = 0
        
        # Shuffle circle indices for better exploration
        indices = list(range(n))
        np.random.shuffle(indices)
        
        # Track progress to adjust step sizes
        iteration_improvements = []
        
        for i in indices:
            current_cx, current_cy, current_r = refined_circles[i]
            
            # Find maximum allowable radius using spatial indexing for efficiency
            max_radius = float('inf')
            
            # Check boundary constraints
            boundary_radius = min([
                current_cx,  # left
                rect_width - current_cx,  # right  
                current_cy,  # bottom
                rect_height - current_cy   # top
            ])
            max_radius = min(max_radius, boundary_radius)
            
            # Check overlap constraints with nearby circles only (using spatial index)
            nearby_indices = tree.query_ball_point([current_cx, current_cy], 2 * max_radius)
            for j in nearby_indices:
                if i != j:
                    other_cx, other_cy, other_r = refined_circles[j]
                    dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                    # Max radius to prevent overlap
                    max_allowed_radius = dist - other_r
                    if max_allowed_radius > 0:
                        max_radius = min(max_radius, max_allowed_radius)
            
            # Try to increase radius with adaptive step size
            if max_radius > current_r and max_radius > 0:
                # Adaptive step size: start large, decrease as we approach convergence
                step_size = min(0.01, max_radius - current_r)  # Limit step size
                new_r = min(current_r + step_size, max_radius)
                
                # Validate with full constraint check
                valid = True
                for j in range(n):
                    if i != j:
                        other_cx, other_cy, other_r = refined_circles[j]
                        dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                        if dist < new_r + other_r:
                            valid = False
                            break
                
                if valid:
                    refined_circles[i, 2] = new_r
                    improved = True
                    total_improvement += (new_r - current_r)
                    iteration_improvements.append(new_r - current_r)
        
        # Rebuild spatial index after updates
        tree = cKDTree(refined_circles[:, :2])
        
        # Stop if no significant improvement
        if not improved or total_improvement < improvement_threshold:
            break

    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")