# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
from typing import Tuple

# Global constants for optimization
MAX_ITERATIONS = 1000
OPTIMIZATION_TOLERANCE = 1e-6
INITIAL_STEP_SIZE = 0.01
BOUNDARY_PENALTY = 1000.0
OVERLAP_PENALTY = 1000.0

def validate_placement(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and don't overlap"""
    n = len(circles)
    
    # Check containment constraints efficiently
    radii = circles[:, 2]
    positions = circles[:, :2]
    
    # Vectorized containment check
    if np.any(radii <= 0) or np.any(positions[:, 0] < radii) or np.any(positions[:, 0] > 1 - radii) or \
       np.any(positions[:, 1] < radii) or np.any(positions[:, 1] > 1 - radii):
        return False

    # Check overlap constraints using KDTree for efficiency
    tree = cKDTree(positions)
    
    # Vectorized overlap checking
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby circles (within 2*r distance) - this is more efficient than checking all pairs
        indices = tree.query_ball_point([x, y], 2*r)
        for j in indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False

    return True

def create_grid_initialization(num_circles: int, rows: int, cols: int) -> np.ndarray:
    """Create a grid-based initialization for circles"""
    circles = np.zeros((num_circles, 3))
    
    # Create a grid of positions
    grid_positions = []
    for i in range(rows):
        for j in range(cols):
            if len(grid_positions) >= num_circles:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            grid_positions.append((x, y))
    
    # Fill circles with grid positions
    for i in range(num_circles):
        if i < len(grid_positions):
            x, y = grid_positions[i]
            # Small random offset
            x += (random.random() - 0.5) * 0.05
            y += (random.random() - 0.5) * 0.05
            # Small random radius
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles[i] = [x, y, r]
        else:
            # Random placement for extra circles
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles[i] = [x, y, r]
            
    return circles

def create_multi_scale_grid_initialization(num_circles: int) -> np.ndarray:
    """Create a multi-scale grid-based initialization for circles"""
    circles = np.zeros((num_circles, 3))
    
    # Try different grid configurations to find a good initial setup
    configs = [
        (int(np.ceil(np.sqrt(num_circles))), int(np.ceil(num_circles / np.ceil(np.sqrt(num_circles))))),
        (5, 6),
        (6, 5), 
        (4, 7),
        (7, 4)
    ]
    
    best_config = None
    best_score = -np.inf
    
    for rows, cols in configs:
        if rows * cols >= num_circles:
            # Create positions
            grid_positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(grid_positions) >= num_circles:
                        break
                    x = (j + 0.5) / cols
                    y = (i + 0.5) / rows
                    grid_positions.append((x, y))
            
            if len(grid_positions) >= num_circles:
                # Calculate score for this configuration
                score = 0
                temp_circles = np.zeros((num_circles, 3))
                for i in range(num_circles):
                    x, y = grid_positions[i]
                    # Add small random perturbation
                    x += (random.random() - 0.5) * 0.03
                    y += (random.random() - 0.5) * 0.03
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    temp_circles[i] = [x, y, r]
                    score += r
                
                if score > best_score:
                    best_score = score
                    best_config = (grid_positions, rows, cols)
    
    if best_config:
        grid_positions, rows, cols = best_config
        for i in range(num_circles):
            x, y = grid_positions[i]
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles[i] = [x, y, r]
    else:
        # Fallback to random initialization
        for i in range(num_circles):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles[i] = [x, y, r]
            
    return circles

def compute_constraint_violation(circles: np.ndarray) -> Tuple[float, float]:
    """Compute total violation of boundary and overlap constraints"""
    n = len(circles)
    boundary_violation = 0.0
    overlap_violation = 0.0
    
    # Boundary violations
    for i in range(n):
        x, y, r = circles[i]
        # Check if radius is negative
        if r <= 0:
            boundary_violation += -r
        # Check boundaries
        if x < r:
            boundary_violation += (r - x)
        if x > 1 - r:
            boundary_violation += (x - (1 - r))
        if y < r:
            boundary_violation += (r - y)
        if y > 1 - r:
            boundary_violation += (y - (1 - r))
    
    # Overlap violations
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            overlap = (r1 + r2) - distance
            if overlap > 0:
                overlap_violation += overlap
    
    return boundary_violation, overlap_violation

def objective_and_gradients(circles_flat: np.ndarray, num_circles: int) -> Tuple[float, np.ndarray]:
    """
    Compute objective (negative sum of radii) and gradients
    Returns: (objective_value, gradient_array)
    """
    # Reshape flat array back to circles format
    circles = circles_flat.reshape((num_circles, 3))
    
    # Compute sum of radii (negative for minimization)
    objective = -np.sum(circles[:, 2])
    
    # Compute gradients for radii (negative since we want to maximize sum of radii)
    grad_radii = -np.ones(num_circles)
    
    # Initialize position gradients to zero
    grad_positions = np.zeros((num_circles, 2))
    
    # Compute overlap penalties and their gradients
    n = len(circles)
    
    # Calculate gradients for each circle's position due to overlap constraints
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 0 and distance < r1 + r2:
                    # Overlap exists - compute gradient contribution
                    penalty = OVERLAP_PENALTY * (r1 + r2 - distance)
                    # Gradient of distance w.r.t. positions
                    grad_dist_x1 = dx / distance if distance > 0 else 0.0
                    grad_dist_y1 = dy / distance if distance > 0 else 0.0
                    grad_dist_x2 = -dx / distance if distance > 0 else 0.0
                    grad_dist_y2 = -dy / distance if distance > 0 else 0.0
                    
                    # Gradient w.r.t. i's position due to j's influence
                    grad_positions[i, 0] += penalty * grad_dist_x1
                    grad_positions[i, 1] += penalty * grad_dist_y1
                    
                    # Gradient w.r.t. j's position due to i's influence
                    grad_positions[j, 0] += penalty * grad_dist_x2
                    grad_positions[j, 1] += penalty * grad_dist_y2
    
    # Add boundary penalties and gradients
    for i in range(n):
        x, y, r = circles[i]
        # Boundary penalties
        if x < r:
            penalty = BOUNDARY_PENALTY * (r - x)
            grad_positions[i, 0] += penalty
        elif x > 1 - r:
            penalty = BOUNDARY_PENALTY * (x - (1 - r))
            grad_positions[i, 0] -= penalty
            
        if y < r:
            penalty = BOUNDARY_PENALTY * (r - y)
            grad_positions[i, 1] += penalty
        elif y > 1 - r:
            penalty = BOUNDARY_PENALTY * (y - (1 - r))
            grad_positions[i, 1] -= penalty
    
    # Flatten gradients
    grad = np.zeros_like(circles_flat)
    grad[::3] = grad_positions[:, 0]  # x positions
    grad[1::3] = grad_positions[:, 1]  # y positions
    grad[2::3] = grad_radii  # radii
    
    return objective, grad

def safe_update_circles(circles: np.ndarray, delta: np.ndarray, step_size: float = 0.01) -> np.ndarray:
    """Safely update circles positions/radii with bounded step sizes"""
    updated = circles.copy()
    step_vector = delta * step_size
    
    # Apply updates safely
    for i in range(len(updated)):
        updated[i, 0] = np.clip(updated[i, 0] + step_vector[i*3], 0.001, 0.999)
        updated[i, 1] = np.clip(updated[i, 1] + step_vector[i*3 + 1], 0.001, 0.999)
        updated[i, 2] = np.maximum(0.001, updated[i, 2] + step_vector[i*3 + 2])
    
    return updated

def gradient_based_optimization(initial_circles: np.ndarray, num_circles: int, max_iter: int = MAX_ITERATIONS) -> np.ndarray:
    """Use gradient-based method to optimize circle packing"""
    # Start with the initial solution
    current_circles = initial_circles.copy()
    
    # Flatten for optimization
    current_flat = current_circles.flatten()
    
    # Optimization parameters
    tolerance = OPTIMIZATION_TOLERANCE
    step_size = INITIAL_STEP_SIZE
    
    # Run gradient descent
    for iteration in range(max_iter):
        # Compute objective and gradients
        obj_val, grad = objective_and_gradients(current_flat, num_circles)
        
        # Check if we're close to optimal
        if np.linalg.norm(grad) < tolerance:
            break
            
        # Update with negative gradient (descent direction)
        # But ensure step is safe and not too large
        step_direction = -grad
        
        # Try multiple step sizes to find a good one
        best_circles = current_circles.copy()
        best_obj_val = obj_val
        best_step_size = step_size
        
        for try_step in [step_size, step_size/2, step_size/4]:
            if try_step < 1e-8:
                break
            try_circles = safe_update_circles(current_circles, step_direction, try_step)
            if validate_placement(try_circles):
                try_flat = try_circles.flatten()
                try_obj, _ = objective_and_gradients(try_flat, num_circles)
                if try_obj < best_obj_val:
                    best_obj_val = try_obj
                    best_circles = try_circles
                    best_step_size = try_step
        
        # If we found an improvement, update
        if best_obj_val < obj_val:
            current_circles = best_circles
            current_flat = current_circles.flatten()
            step_size = best_step_size
        else:
            # Reduce step size if no improvement
            step_size *= 0.8
            if step_size < 1e-8:
                break
    
    return current_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    # Create initial population using grid initialization
    initial_circles = create_multi_scale_grid_initialization(26)
    
    # Refine with gradient-based optimization
    refined_circles = gradient_based_optimization(initial_circles, 26)
    
    # Ensure final solution is valid
    if not validate_placement(refined_circles):
        # If invalid, fall back to the best valid configuration from our initial population
        # Use a simpler approach: just make sure all circles satisfy constraints
        for i in range(len(refined_circles)):
            x, y, r = refined_circles[i]
            # Bound radii and positions properly
            r = max(0.001, min(0.5, r))
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            refined_circles[i] = [x, y, r]
    
    return refined_circles

# EVOLVE-BLOCK-END