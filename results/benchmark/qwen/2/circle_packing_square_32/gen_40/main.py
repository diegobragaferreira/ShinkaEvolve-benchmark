# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math

def initialize_hexagonal_grid(n=32):
    """Initialize circle positions using a hexagonal grid pattern."""
    # Calculate grid parameters
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Create hexagonal grid with proper spacing
    spacing = 1.0 / max(rows, cols)
    radius_estimate = spacing * 0.3
    
    circles = []
    
    # Generate hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
                
            # Hexagonal offset pattern
            x = (j + 0.5 + (i % 2) * 0.5) * spacing
            y = (i + 0.5) * spacing
            
            # Adjust to stay within bounds
            x = max(radius_estimate, min(1 - radius_estimate, x))
            y = max(radius_estimate, min(1 - radius_estimate, y))
            
            # Ensure validity
            if x >= radius_estimate and x <= 1 - radius_estimate and \
               y >= radius_estimate and y <= 1 - radius_estimate:
                circles.append([x, y, radius_estimate])
    
    # Fill remaining positions
    while len(circles) < n:
        # Find empty spots by sampling
        for _ in range(1000):
            x = np.random.uniform(radius_estimate, 1 - radius_estimate)
            y = np.random.uniform(radius_estimate, 1 - radius_estimate)
            
            # Check proximity to existing circles
            min_dist = float('inf')
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist = min(min_dist, dist)
            
            # If far enough from others, try to place a circle
            if min_dist > radius_estimate:
                r = min(x, y, 1-x, 1-y) * 0.4  # Conservative estimate
                if r > radius_estimate:
                    circles.append([x, y, r])
                    break
    
    return np.array(circles[:n])

def evaluate_fitness(circles):
    """Calculate the fitness (sum of radii) of the current configuration."""
    return np.sum(circles[:, 2])

def calculate_constraints(circles):
    """Calculate constraint violations for all circles."""
    violations = []
    n = len(circles)
    
    # Boundary violations
    for i in range(n):
        x, y, r = circles[i]
        boundary_violation = 0
        
        # Check all boundaries
        if r > x:
            boundary_violation += (r - x)
        if r > y:
            boundary_violation += (r - y)
        if r > (1 - x):
            boundary_violation += (r - (1 - x))
        if r > (1 - y):
            boundary_violation += (r - (1 - y))
            
        violations.append(boundary_violation)
    
    # Overlap violations
    tree = cKDTree(circles[:, :2])
    for i in range(n):
        x, y, r = circles[i]
        
        # Find nearby circles
        neighbors = tree.query_ball_point([x, y], 2 * r)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                overlap_violation = max(0, (r + r2) - distance)
                violations[i] += overlap_violation
    
    return violations

def optimize_individual_circle(circle_idx, circles, global_params=None):
    """Optimize a single circle while keeping others fixed."""
    def objective(r):
        # Create temporary circles array
        temp_circles = circles.copy()
        temp_circles[circle_idx, 2] = r[0]
        
        # Calculate constraints (penalized fitness)
        violations = calculate_constraints(temp_circles)
        penalty = sum(violations) * 1000  # Large penalty for violations
        
        # Maximize sum of radii (negative because minimize)
        return -(evaluate_fitness(temp_circles) - penalty)
    
    # Get current values
    current_circle = circles[circle_idx]
    current_radius = current_circle[2]
    
    # Optimization bounds
    bounds = [(0.001, 0.5)]
    
    # Optimize just this circle
    result = minimize(objective, [current_radius], method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        return result.x[0]
    else:
        return current_radius

def refine_configuration(circles, max_iterations=50):
    """Refine the circle configuration using iterative optimization."""
    n = len(circles)
    
    for iteration in range(max_iterations):
        # Store old configuration
        old_circles = circles.copy()
        
        # Optimize each circle individually
        for i in range(n):
            new_radius = optimize_individual_circle(i, circles)
            circles[i, 2] = new_radius
        
        # Check for convergence
        if np.allclose(old_circles, circles, atol=1e-6):
            break
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize using hexagonal grid
    circles = initialize_hexagonal_grid(32)
    
    # Refine the configuration
    circles = refine_configuration(circles)
    
    # Final optimization pass
    circles = refine_configuration(circles, max_iterations=20)
    
    return circles

# EVOLVE-BLOCK-END