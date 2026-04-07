# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Global constants
BOUNDARY_PENALTY = 10000
OVERLAP_PENALTY_MULTIPLIER = 1000
MAX_RADIUS = 0.5
MIN_RADIUS = 0.001

def initialize_hexagonal_grid(n_circles: int) -> np.ndarray:
    """Initialize circles using a hexagonal grid pattern for better distribution"""
    circles = np.zeros((n_circles, 3))
    
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Adjust spacing to fit within unit square
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    
    # Hexagonal packing parameters
    hex_radius = min(cell_width, cell_height) * 0.4  # Slightly smaller than cell
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
                
            # Offset odd rows for hexagonal packing
            x_offset = (j * cell_width) + cell_width/2
            y_offset = (i * cell_height) + cell_height/2
            
            # For hexagonal grid, offset odd rows
            if i % 2 == 1:
                x_offset += cell_width/2
                
            # Adjust to stay within bounds
            x = max(hex_radius, min(1 - hex_radius, x_offset))
            y = max(hex_radius, min(1 - hex_radius, y_offset))
            
            # Set initial radius and position
            circles[idx] = [x, y, hex_radius * (0.8 + random.random() * 0.4)]
            idx += 1
            
        if idx >= n_circles:
            break
    
    # Ensure we have exactly n_circles
    if idx < n_circles:
        # Fill remaining circles with random valid positions
        for i in range(idx, n_circles):
            x = random.uniform(hex_radius, 1 - hex_radius)
            y = random.uniform(hex_radius, 1 - hex_radius)
            r = random.uniform(0.01, 0.2)
            circles[i] = [x, y, r]
            
    return circles

def compute_constraints(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute constraint violations for all circles"""
    n = len(circles)
    
    # Boundary constraints
    boundary_violations = []
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        # Penalize if circle goes outside bounds
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            boundary_violations.append(i)
    
    # Overlap constraints
    overlap_violations = []
    
    # Compute all pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    if n > 1:
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r_i = radii[i]
                r_j = radii[j]
                # If circles overlap
                if dist < (r_i + r_j):
                    overlap_violations.append((i, j))
    
    return np.array(boundary_violations), np.array(overlap_violations)

def penalty_function(circles: np.ndarray) -> float:
    """Calculate penalty for constraint violations"""
    penalty = 0
    
    # Boundary penalty
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += BOUNDARY_PENALTY
    
    # Overlap penalty
    positions = circles[:, :2]
    radii = circles[:, 2]
    n = len(circles)
    
    if n > 1:
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r_i = radii[i]
                r_j = radii[j]
                if dist < (r_i + r_j):
                    overlap = (r_i + r_j) - dist
                    penalty += overlap * OVERLAP_PENALTY_MULTIPLIER
    
    return penalty

def objective_function(circles: np.ndarray) -> float:
    """Objective function to maximize: sum of radii minus penalties"""
    sum_radii = np.sum(circles[:, 2])
    penalty = penalty_function(circles)
    return -(sum_radii - penalty)  # Negative because we minimize

def gradient_of_objective(circles: np.ndarray) -> np.ndarray:
    """
    Compute approximate gradient of objective function with respect to circle parameters.
    This is a simplified approximation based on finite differences.
    """
    epsilon = 1e-6
    grad = np.zeros_like(circles)
    
    # For each parameter, compute finite difference
    for i in range(len(circles)):
        for j in range(3):  # x, y, r
            # Perturb parameter
            circles_plus = circles.copy()
            circles_minus = circles.copy()
            circles_plus[i, j] += epsilon
            circles_minus[i, j] -= epsilon
            
            # Calculate gradient using central difference
            f_plus = objective_function(circles_plus)
            f_minus = objective_function(circles_minus)
            grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
    
    return grad

def project_to_feasible(circles: np.ndarray) -> np.ndarray:
    """Project circles to feasible region (ensuring they're within bounds)"""
    projected = circles.copy()
    
    for i in range(len(projected)):
        x, y, r = projected[i]
        # Ensure radius is within bounds
        r = np.clip(r, MIN_RADIUS, MAX_RADIUS)
        
        # Ensure circle is within bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        projected[i] = [x, y, r]
    
    return projected

def solve_with_gradient_descent(initial_circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Solve circle packing using gradient-based optimization
    """
    circles = initial_circles.copy()
    
    # Simple gradient descent with projection
    learning_rate = 0.01
    
    for iteration in range(max_iter):
        # Compute gradient
        try:
            grad = gradient_of_objective(circles)
            
            # Update circles (move in negative gradient direction)
            circles[:, 0] -= learning_rate * grad[:, 0]  # x
            circles[:, 1] -= learning_rate * grad[:, 1]  # y
            circles[:, 2] -= learning_rate * grad[:, 2]  # r
            
            # Project back to feasible region
            circles = project_to_feasible(circles)
            
        except Exception as e:
            # If gradient computation fails, fall back to simple projection
            circles = project_to_feasible(circles)
            continue
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize with hexagonal grid
    initial_circles = initialize_hexagonal_grid(32)
    
    # Try multiple restarts with different random seeds to avoid local minima
    best_solution = initial_circles.copy()
    best_sum_radii = np.sum(initial_circles[:, 2])
    
    # Run several optimization attempts
    for attempt in range(5):
        # Set different random seed for each attempt
        random.seed(attempt)
        np.random.seed(attempt)
        
        # Reinitialize with new seed
        circles = initialize_hexagonal_grid(32)
        
        # Optimize using gradient descent
        optimized_circles = solve_with_gradient_descent(circles, max_iter=500)
        
        # Calculate sum of radii
        sum_radii = np.sum(optimized_circles[:, 2])
        
        # Validate solution
        boundary_violations, overlap_violations = compute_constraints(optimized_circles)
        
        # Only accept valid solutions with better sum of radii
        if len(boundary_violations) == 0 and len(overlap_violations) == 0 and sum_radii > best_sum_radii:
            best_solution = optimized_circles.copy()
            best_sum_radii = sum_radii
    
    return best_solution

# EVOLVE-BLOCK-END