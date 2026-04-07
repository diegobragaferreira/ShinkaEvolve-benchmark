# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # We'll try different aspect ratios to find optimal configuration
    width = 1.0
    height = 1.0
    
    # Initial grid-based placement
    n_circles = 21
    
    # Try to arrange in roughly a 3x7 grid (or similar)
    rows = 3
    cols = 7
    
    # Create initial guess with grid placement
    circles = np.zeros((n_circles, 3))
    
    # Calculate spacing based on rectangle size
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)
    
    # Initialize circles in a grid pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            
            # Start with small radius - will be optimized
            circles[idx] = [x, y, 0.05]
            idx += 1
    
    # If we have fewer than 21 circles, fill with more grid points
    if idx < n_circles:
        # Add additional circles
        for i in range(idx, n_circles):
            circles[i] = [0.5, 0.5, 0.05]  # Center point with small radius
    
    def objective(radii):
        """Minimize negative sum of radii"""
        return -np.sum(radii)
    
    def constraint_func(params):
        """Check all pairwise distances between circles"""
        # Extract positions and radii
        positions = params[:-n_circles].reshape(-1, 2)
        radii = params[-n_circles:]
        
        # Check boundary constraints
        boundary_violations = []
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                boundary_violations.append(1e6)
            else:
                boundary_violations.append(0)
        
        # Check overlap constraints
        overlap_violations = []
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                min_dist = radii[i] + radii[j]
                violation = max(0, min_dist - dist)
                overlap_violations.append(violation)
        
        return np.concatenate([boundary_violations, overlap_violations])
    
    # Flatten initial parameters: [x1,y1,x2,y2,...,xn,yn,r1,r2,...,rn]
    initial_params = np.concatenate([circles[:, :2].flatten(), circles[:, 2]])
    
    # Define bounds for positions (0,0) to (width,height) and radii (>0)
    bounds = []
    for i in range(n_circles):
        bounds.extend([(0, width), (0, height)])  # x,y bounds
    for i in range(n_circles):
        bounds.append((1e-6, min(width, height)/2))  # radius bounds
    
    # Set up constraints
    # We'll make this simpler: just add a penalty to violations
    def penalty_function(params):
        positions = params[:-n_circles].reshape(-1, 2)
        radii = params[-n_circles:]
        
        # Boundary penalty
        penalty = 0
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty += 1e6
        
        # Overlap penalty
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += 1e6 * (min_dist - dist)
        
        return penalty
    
    # Run optimization with bounds
    def optimized_objective(params):
        positions = params[:-n_circles].reshape(-1, 2)
        radii = params[-n_circles:]
        return -np.sum(radii) + penalty_function(params)
    
    # Use a simple optimization approach with bounds
    best_result = None
    best_sum = -float('inf')
    
    # Try several random restarts to avoid local minima
    for _ in range(5):
        # Random perturbation of initial guess
        perturbed = initial_params.copy()
        # Perturb positions slightly
        for i in range(n_circles * 2):
            perturbed[i] += np.random.normal(0, 0.01)
        # Perturb radii slightly
        for i in range(n_circles):
            perturbed[n_circles * 2 + i] = max(0.001, perturbed[n_circles * 2 + i] + np.random.normal(0, 0.01))
        
        # Simple bounded optimization approach
        bounds_list = []
        for i in range(n_circles * 2):
            bounds_list.append((0, width) if i % 2 == 0 else (0, height))
        for i in range(n_circles):
            bounds_list.append((0.001, min(width, height)/2))
        
        # For simplicity in this constrained environment, we'll do a greedy local search approach
        # First try the grid configuration
        test_positions = circles[:, :2].copy()
        test_radii = circles[:, 2].copy()
        
        # Simplified improvement using coordinate descent
        improved = True
        iterations = 0
        
        while improved and iterations < 50:
            improved = False
            iterations += 1
            
            # Try to increase radii while maintaining constraints
            new_radii = test_radii.copy()
            for i in range(n_circles):
                # Calculate how much we can increase radius i
                max_radius = float('inf')
                
                # Check boundary constraints
                max_radius = min(max_radius, test_positions[i][0] - 0.001)
                max_radius = min(max_radius, width - test_positions[i][0] - 0.001)
                max_radius = min(max_radius, test_positions[i][1] - 0.001)
                max_radius = min(max_radius, height - test_positions[i][1] - 0.001)
                
                # Check overlap constraints with all other circles
                for j in range(n_circles):
                    if i != j:
                        dist = np.sqrt((test_positions[i][0] - test_positions[j][0])**2 + 
                                     (test_positions[i][1] - test_positions[j][1])**2)
                        max_radius = min(max_radius, dist - 0.001)
                
                # Don't let it be larger than our previous value by too much
                if max_radius > test_radii[i]:
                    new_radii[i] = min(max_radius, test_radii[i] * 1.2)
                    improved = True
            
            test_radii = new_radii
            
            # Improve positions slightly by moving to increase total radius
            new_positions = test_positions.copy()
            for i in range(n_circles):
                best_pos = test_positions[i]
                best_radius = test_radii[i]
                current_total = np.sum(test_radii)
                
                # Check 9 nearby positions
                step = 0.02
                directions = [(0,0), (-step,0), (step,0), (0,-step), (0,step), 
                            (-step,-step), (step,step), (-step,step), (step,-step)]
                
                for dx, dy in directions:
                    new_x = test_positions[i][0] + dx
                    new_y = test_positions[i][1] + dy
                    
                    # Boundary check
                    if (new_x - test_radii[i] >= 0 and new_x + test_radii[i] <= width and
                        new_y - test_radii[i] >= 0 and new_y + test_radii[i] <= height):
                        
                        # Check overlaps with others
                        valid = True
                        temp_radii = test_radii.copy()
                        temp_radii[i] = test_radii[i]  # No change to radius yet
                        
                        for j in range(n_circles):
                            if i != j:
                                dist = np.sqrt((new_x - test_positions[j][0])**2 + 
                                             (new_y - test_positions[j][1])**2)
                                if dist < (temp_radii[i] + temp_radii[j]):
                                    valid = False
                                    break
                        
                        if valid:
                            # Compute a rough improvement
                            temp_positions = test_positions.copy()
                            temp_positions[i] = [new_x, new_y]
                            
                            # We want to maximize sum of radii
                            # So we check if we can increase total sum by adjusting position/radius
                            # Here we'll just do a greedy approach of increasing radius slightly
                            new_radius = test_radii[i]
                            if new_radius < min(new_x, width-new_x, new_y, height-new_y):
                                new_radius = min(new_radius*1.05, min(new_x, width-new_x, new_y, height-new_y)*0.9)
                                if new_radius > test_radii[i]:
                                    best_radius = new_radius
                                    best_pos = [new_x, new_y]
                                    improved = True
                                    break
            
            if improved:
                test_positions = new_positions
                test_radii = new_radii
                
            if not improved:
                # Try a different approach, just increase all radii
                for i in range(n_circles):
                    if test_radii[i] < min(test_positions[i][0], width-test_positions[i][0], 
                                         test_positions[i][1], height-test_positions[i][1]):
                        test_radii[i] = min(test_radii[i]*1.02, 
                                          min(test_positions[i][0], width-test_positions[i][0], 
                                              test_positions[i][1], height-test_positions[i][1])*0.9)
                        improved = True
        
        # Final check with our greedy method
        final_positions = test_positions
        final_radii = test_radii
        
        # Apply final constraint checking
        total_sum = 0
        for i in range(n_circles):
            valid = True
            # Check boundary
            if (final_positions[i][0] - final_radii[i] < 0 or 
                final_positions[i][0] + final_radii[i] > width or
                final_positions[i][1] - final_radii[i] < 0 or 
                final_positions[i][1] + final_radii[i] > height):
                valid = False
            
            # Check overlaps
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((final_positions[i][0] - final_positions[j][0])**2 + 
                                 (final_positions[i][1] - final_positions[j][1])**2)
                    if dist < final_radii[i] + final_radii[j]:
                        valid = False
                        break
            
            if valid:
                total_sum += final_radii[i]
        
        if total_sum > best_sum:
            best_sum = total_sum
            best_result = np.column_stack([final_positions, final_radii])
    
    # Return the best result found
    return best_result if best_result is not None else circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
