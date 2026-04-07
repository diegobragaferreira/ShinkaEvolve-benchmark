# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math
import random

def check_containment(circles):
    """Check containment constraints efficiently"""
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    # Check boundaries for all circles at once
    containment_violations = (
        (x_coords - radii < 0.0) |
        (x_coords + radii > 1.0) |
        (y_coords - radii < 0.0) |
        (y_coords + radii > 1.0)
    )
    
    return np.sum(containment_violations)

def calculate_overlap_penalty(circles):
    """Calculate overlap penalty using efficient spatial indexing"""
    if len(circles) <= 1:
        return 0.0
    
    # Build KDTree for efficient neighbor search
    tree = cKDTree(circles[:, :2])
    
    penalty = 0.0
    radii = circles[:, 2]
    
    # For each circle, find neighbors within sum of radii
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        
        # Query nearby points (within 2*(r1+r2) distance)
        neighbors = tree.query_ball_point([x1, y1], 2*(r1 + max(radii)))
        
        # Check overlaps with neighbors
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    penalty += 1000 * (r1 + r2 - distance)
    
    return penalty

def evaluate_fitness(circles):
    """Evaluate fitness of a configuration"""
    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])
    
    # Check constraints
    containment_violations = check_containment(circles)
    overlap_penalty = calculate_overlap_penalty(circles)
    
    # Combine penalties
    total_penalty = 10000 * containment_violations + overlap_penalty
    
    # Return fitness (higher is better)
    return total_radius - total_penalty

def generate_grid_placement(n=26):
    """Generate initial circle positions on a structured grid"""
    # Create a 5x5 grid (25 circles) plus one extra
    rows_cols = 5
    spacing_x = 1.0 / (rows_cols + 1)
    spacing_y = 1.0 / (rows_cols + 1)
    
    circles = []
    count = 0
    
    # Generate initial grid positions
    for i in range(rows_cols):
        for j in range(rows_cols):
            if count >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            
            # Add some jitter to positions
            x += random.uniform(-spacing_x/6, spacing_x/6)
            y += random.uniform(-spacing_y/6, spacing_y/6)
            
            # Ensure positions stay within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radii based on proximity to edges and other circles
            min_dist_to_bound = min(x, 1-x, y, 1-y)
            r = min(0.08, min_dist_to_bound/2)
            
            # Add some randomness to radius
            r *= random.uniform(0.8, 1.2)
            r = max(0.005, min(0.15, r))
            
            circles.extend([x, y, r])
            count += 1
            
        if count >= n:
            break
    
    # If we don't have enough circles, add random ones
    while len(circles) < n * 3:
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        r = random.uniform(0.005, 0.12)
        circles.extend([x, y, r])
    
    return np.array(circles[:n*3]).reshape(-1, 3)

def local_optimize_placement(circles, max_iter=100):
    """Use local optimization to improve circle placement"""
    n = len(circles)
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.4))
    
    # Flatten circles for optimization
    initial_flat = circles.flatten()
    
    def objective(params):
        # Reshape back to circles
        temp_circles = params.reshape(-1, 3)
        
        # Calculate sum of radii (negative because we're minimizing)
        sum_radii = -np.sum(temp_circles[:, 2])
        
        # Penalty for constraint violations
        penalty = 0
        
        # Boundary constraints
        x_coords = temp_circles[:, 0]
        y_coords = temp_circles[:, 1]
        radii = temp_circles[:, 2]
        
        # Check containment violations
        containment_violations = (
            (x_coords - radii < 0.0) |
            (x_coords + radii > 1.0) |
            (y_coords - radii < 0.0) |
            (y_coords + radii > 1.0)
        )
        penalty += 10000 * np.sum(containment_violations)
        
        # Overlap penalties
        if len(temp_circles) > 1:
            for i in range(len(temp_circles)):
                x1, y1, r1 = temp_circles[i]
                for j in range(i+1, len(temp_circles)):
                    x2, y2, r2 = temp_circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        penalty += 1000 * (r1 + r2 - distance)
        
        return sum_radii + penalty
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(objective, initial_flat, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-6})
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure validity
            for i in range(len(optimized_circles)):
                optimized_circles[i][0] = np.clip(optimized_circles[i][0], 0.001, 0.999)
                optimized_circles[i][1] = np.clip(optimized_circles[i][1], 0.001, 0.999)
                optimized_circles[i][2] = np.clip(optimized_circles[i][2], 0.001, 0.4)
            return optimized_circles
    except Exception:
        pass
    
    # If optimization fails, return original
    return circles

def improve_with_local_search(circles, iterations=3):
    """Apply multiple rounds of local optimization"""
    current_best = circles.copy()
    
    for i in range(iterations):
        # Apply local optimization
        optimized = local_optimize_placement(current_best, max_iter=50)
        
        # Check improvement
        current_fitness = evaluate_fitness(current_best)
        optimized_fitness = evaluate_fitness(optimized)
        
        if optimized_fitness > current_fitness:
            current_best = optimized
        else:
            # If no improvement, try a slightly different approach
            # Add small random perturbations to escape local minima
            for j in range(len(current_best)):
                if random.random() < 0.1:  # 10% chance to perturb
                    current_best[j][0] += random.uniform(-0.01, 0.01)
                    current_best[j][1] += random.uniform(-0.01, 0.01)
                    current_best[j][2] += random.uniform(-0.005, 0.005)
                    
                    # Clamp values
                    current_best[j][0] = np.clip(current_best[j][0], 0.001, 0.999)
                    current_best[j][1] = np.clip(current_best[j][1], 0.001, 0.999)
                    current_best[j][2] = np.clip(current_best[j][2], 0.001, 0.4)
    
    return current_best

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with grid-based approach
    circles = generate_grid_placement(26)
    
    # Apply local optimization improvements
    circles = improve_with_local_search(circles, iterations=5)
    
    # Final optimization pass
    circles = local_optimize_placement(circles, max_iter=100)
    
    return circles

# EVOLVE-BLOCK-END