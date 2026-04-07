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
    # Rectangle dimensions: width + height = 2
    # Using a 1:1 ratio for simplicity, so width = height = 1
    rect_width = 1.0
    rect_height = 1.0
    
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
    
    # Generate initial guess with hexagonal arrangement
    def generate_initial_guess():
        # Place circles in a hexagonal pattern for good initial distribution
        circles = np.zeros((n, 3))
        
        # Try to distribute circles in a hexagonal pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Adjust positions to avoid corners
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                # Add slight offset for odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Initialize with small radius
                r = 0.05
                
                # Constrain to stay within bounds
                x = max(r, min(rect_width - r, x))
                y = max(r, min(rect_height - r, y))
                
                circles[idx] = [x, y, r]
                idx += 1
                if idx >= n:
                    break
        
        # Ensure we have exactly n circles
        if idx < n:
            # Fill remaining slots with small random values
            for i in range(idx, n):
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                r = 0.05
                circles[i] = [x, y, r]
        
        return circles.flatten()
    
    # Start with a good initial configuration
    initial_guess = generate_initial_guess()
    
    # Run optimization using Nelder-Mead method
    try:
        # Combine objective and penalty function
        combined_objective = lambda x: objective(x) + penalty_function(x)
        
        result = minimize(
            combined_objective,
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'disp': False, 'maxfev': 5000}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, fall back to just the initial solution
            final_circles = initial_guess.reshape(-1, 3)
            
    except Exception as e:
        # Fallback to initial guess if anything goes wrong
        final_circles = initial_guess.reshape(-1, 3)
    
    # Final adjustment to ensure all constraints hold and maximize radii
    # Apply a greedy refinement step to increase radii within constraints
    refined_circles = final_circles.copy()
    
    # Local refinement to maximize radii while respecting constraints
    max_iter = 1000
    improvement_threshold = 1e-6
    
    for iteration in range(max_iter):
        improved = False
        total_improvement = 0
        
        for i in range(n):
            current_cx, current_cy, current_r = refined_circles[i]
            
            # Find maximum allowable radius
            max_radius = float('inf')
            
            # Check boundary constraints
            boundary_radius = min([
                current_cx,  # left
                rect_width - current_cx,  # right  
                current_cy,  # bottom
                rect_height - current_cy   # top
            ])
            max_radius = min(max_radius, boundary_radius)
            
            # Check overlap constraints with other circles
            for j in range(n):
                if i != j:
                    other_cx, other_cy, other_r = refined_circles[j]
                    dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                    # Max radius to prevent overlap
                    max_allowed_radius = dist - other_r
                    if max_allowed_radius > 0:
                        max_radius = min(max_radius, max_allowed_radius)
            
            # Try to increase radius if possible
            if max_radius > current_r and max_radius > 0:
                # Try to increase radius by small amount
                new_r = min(current_r + 0.001, max_radius)
                
                # Verify this doesn't violate constraints
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
        
        # Stop if no significant improvement
        if not improved or total_improvement < improvement_threshold:
            break
    
    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
