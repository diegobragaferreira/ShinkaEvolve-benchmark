# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_hexagonal_grid(n_circles):
    """Initialize circle positions using a hexagonal grid pattern with better spacing."""
    # Calculate optimal grid dimensions
    sqrt_n = math.ceil(math.sqrt(n_circles))
    
    # Create hexagonal grid with appropriate spacing
    # For hexagonal packing, distance between centers is 2*r
    # Using slightly smaller spacing for optimization flexibility
    spacing = 0.12
    
    rows = math.ceil(n_circles / sqrt_n)
    cols = math.ceil(n_circles / rows)
    
    positions = []
    
    # Generate hexagonal grid with staggered rows
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for hexagonal pattern
            x_offset = 0.5 * (i % 2)
            x = (j + x_offset) * spacing
            y = i * spacing * math.sqrt(3) / 2
            
            # Only add if within bounds
            if x <= 1 - spacing/2 and y <= 1 - spacing/2:
                positions.append([x, y])
            
            if len(positions) >= n_circles:
                break
        if len(positions) >= n_circles:
            break
    
    # Trim to exact number of circles needed
    positions = positions[:n_circles]
    
    # Ensure positions are within bounds (avoid edge cases)
    for pos in positions:
        pos[0] = max(spacing/2, min(1 - spacing/2, pos[0]))
        pos[1] = max(spacing/2, min(1 - spacing/2, pos[1]))
    
    return np.array(positions)

def calculate_smooth_penalties(circles, penalty_weight=1000.0):
    """Calculate smooth exponential penalties for constraints."""
    n = len(circles)
    penalty = 0.0
    
    # Boundary penalties using smooth exponential function
    for i in range(n):
        x, y, r = circles[i]
        
        # Penalties for boundary violations
        left_violation = max(0, r - x)
        right_violation = max(0, x + r - 1)
        bottom_violation = max(0, r - y)
        top_violation = max(0, y + r - 1)
        
        # Exponential penalty for boundary violations
        if left_violation > 0:
            penalty += penalty_weight * np.exp(10 * left_violation)
        if right_violation > 0:
            penalty += penalty_weight * np.exp(10 * right_violation)
        if bottom_violation > 0:
            penalty += penalty_weight * np.exp(10 * bottom_violation)
        if top_violation > 0:
            penalty += penalty_weight * np.exp(10 * top_violation)
    
    # Overlap penalties using smooth exponential function
    # Use spatial indexing for efficiency
    positions = circles[:, :2]
    tree = cKDTree(positions)
    
    # Find nearby points using spatial index (within 2*(max_radius))
    max_radius = np.max(circles[:, 2])
    pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
    
    # Check actual overlaps for candidate pairs
    for i, j in pairs:
        if i < j:  # Avoid duplicate checks
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if dist < r1 + r2:
                # Exponential penalty for overlap
                overlap = (r1 + r2 - dist)
                penalty += penalty_weight * np.exp(10 * overlap)
    
    return penalty

def objective_function(circles_flat):
    """Objective function to maximize the sum of radii."""
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))

    # Sum of radii (negative because we're minimizing)
    sum_radii = -np.sum(circles[:, 2])

    # Add smooth penalty for constraint violations
    penalty = calculate_smooth_penalties(circles)

    return sum_radii + penalty

def enforce_constraints(circles):
    """Enforce geometric constraints on final result."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        
        # Ensure radius is positive and reasonable
        r = max(0.001, min(0.45, r))
        
        # Ensure circle stays within bounds
        x = max(r, min(1 - r, x))
        y = max(r, min(1 - r, y))
        
        circles[i] = [x, y, r]
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Initialize using enhanced hexagonal grid
    initial_positions = initialize_hexagonal_grid(n)

    # Initialize with adaptive radii based on local density
    initial_radii = np.full(n, 0.06)
    
    # Combine into single array [x1, y1, r1, x2, y2, r2, ...]
    initial_circles = np.column_stack([initial_positions, initial_radii]).flatten()

    # Use scipy optimization to improve the packing
    result = minimize(
        objective_function,
        initial_circles,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-6}
    )

    # Extract final circles
    final_circles = result.x.reshape((n, 3))

    # Enforce constraints on final result
    final_circles = enforce_constraints(final_circles)

    return final_circles

# EVOLVE-BLOCK-END