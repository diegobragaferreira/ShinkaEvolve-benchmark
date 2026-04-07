# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _initialize_adaptive_hexagonal_grid(n):
    """Initialize circles using an adaptive hexagonal grid pattern"""
    # Calculate grid dimensions for n circles
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))

    # Adjust to ensure we have enough slots
    while rows * cols < n:
        rows += 1

    # Create hexagonal grid points
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows

    # Hexagonal offset
    offset = spacing_x * 0.5

    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j + (i % 2) * 0.5) * spacing_x
            y = i * spacing_y
            points.append([x, y])

    # Ensure we have exactly n points
    points = points[:n]

    # Initialize radii based on density - circles in denser regions start smaller
    points_np = np.array(points)
    
    # Compute density using nearest neighbors
    tree = cKDTree(points_np)
    densities = []
    
    # For each point, find number of neighbors within a certain radius
    for i in range(n):
        # Find neighbors within 0.1 distance (adjustable parameter)
        neighbors = tree.query_ball_point(points_np[i], 0.1)
        densities.append(len(neighbors))
    
    # Normalize densities and adjust radii inversely
    max_density = max(densities) if densities else 1
    radii = []
    for density in densities:
        normalized_density = density / max_density if max_density > 0 else 0
        # Higher density → smaller initial radius
        base_radius = 0.05 * (1 - normalized_density * 0.8) + 0.01
        radii.append(max(0.005, min(0.1, base_radius)))
    
    return np.array(points), radii

def _compute_overlap_penalty(circles, i, j):
    """Compute penalty for overlap between two circles"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dx = x1 - x2
    dy = y1 - y2
    dist = math.sqrt(dx*dx + dy*dy)
    
    if dist < (r1 + r2):
        # Overlap exists
        overlap = (r1 + r2) - dist
        # Use exponential penalty with adaptive scaling
        return 1000 * math.exp(10 * overlap)
    return 0.0

def _compute_boundary_penalty(circles, i):
    """Compute penalty for boundary violations"""
    x, y, r = circles[i]
    
    # Calculate minimum distance to each boundary
    dist_to_left = x - r
    dist_to_right = 1 - x - r
    dist_to_bottom = y - r
    dist_to_top = 1 - y - r
    
    min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
    
    if min_dist < 0:
        # Violation exists
        return 1000 * math.exp(10 * min_dist)
    return 0.0

def _compute_total_penalty(circles):
    """Compute total penalty for all constraint violations"""
    penalty = 0.0
    
    # Boundary penalties
    for i in range(len(circles)):
        penalty += _compute_boundary_penalty(circles, i)
    
    # Overlap penalties using spatial indexing for efficiency
    tree = cKDTree(circles[:, :2])
    
    # Query pairs within a reasonable distance
    pairs = tree.query_pairs(0.1)  # Adjust radius as needed
    
    for i, j in pairs:
        if i < j:  # Avoid double counting
            penalty += _compute_overlap_penalty(circles, i, j)
    
    return penalty

def _objective_function(params, circles, n):
    """Objective function to maximize sum of radii (negative for minimization)"""
    # Update circles array with current parameters
    for i in range(n):
        circles[i, 0] = params[3*i]
        circles[i, 1] = params[3*i+1]
        circles[i, 2] = params[3*i+2]

    # Negative sum of radii (since we want to maximize)
    objective = -np.sum(circles[:, 2])

    # Add penalty for constraint violations
    objective += _compute_total_penalty(circles)

    return objective

def _compute_constraints(circles):
    """Compute constraint violations for bounds checking"""
    violations = []
    
    # Check containment constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            violations.append(-(x - r))
        if x + r > 1:
            violations.append(x + r - 1)
        if y - r < 0:
            violations.append(-(y - r))
        if y + r > 1:
            violations.append(y + r - 1)
    
    # Check overlap constraints
    tree = cKDTree(circles[:, :2])
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < (r1 + r2):
                violations.append((r1 + r2 - dist))
    
    return np.array(violations) if violations else np.array([])

def _refine_with_local_search(circles, n, max_iter=50):
    """Perform local refinement around current solution"""
    # Create a copy to work with
    current_circles = circles.copy()
    
    # Simple local search: try small adjustments
    for iteration in range(max_iter):
        # Try small random changes to each circle
        for i in range(n):
            # Save current state
            old_x, old_y, old_r = current_circles[i]
            
            # Try small random perturbations
            delta_x = random.uniform(-0.005, 0.005)
            delta_y = random.uniform(-0.005, 0.005)
            delta_r = random.uniform(-0.002, 0.002)
            
            # Apply changes
            new_x = old_x + delta_x
            new_y = old_y + delta_y
            new_r = old_r + delta_r
            
            # Ensure new values are valid
            if new_x < new_r or new_x > 1 - new_r or \
               new_y < new_r or new_y > 1 - new_r or \
               new_r <= 0:
                continue
                
            # Update current circles temporarily
            temp_circles = current_circles.copy()
            temp_circles[i] = [new_x, new_y, new_r]
            
            # Check if this change improves the solution
            old_sum = -np.sum(current_circles[:, 2])
            new_sum = -np.sum(temp_circles[:, 2])
            
            # If improvement (or accept worse solution occasionally), accept it
            if new_sum <= old_sum:
                current_circles[i] = [new_x, new_y, new_r]
    
    return current_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Initialize using adaptive hexagonal grid
    points, radii = _initialize_adaptive_hexagonal_grid(n)

    # Set initial positions and radii
    for i in range(n):
        circles[i, 0] = points[i, 0]
        circles[i, 1] = points[i, 1]
        circles[i, 2] = radii[i]

    # Flatten for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])

    # Optimization with bounds
    bounds = []
    for i in range(n):
        # Bounds for x and y coordinates (slightly away from edges)
        bounds.append((0.001, 0.999))  # Positions
        bounds.append((0.001, 0.999))  # Positions
        # Bounds for radii - allow some flexibility but prevent extreme values
        bounds.append((0.001, 0.49))   # Radii (max radius limited to avoid overflow)

    # Stage 1: Coarse optimization with relaxed constraints
    try:
        # First pass with L-BFGS-B
        result = minimize(
            _objective_function,
            initial_params,
            args=(circles, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-3}
        )

        # Update circles with optimized results
        for i in range(n):
            circles[i, 0] = result.x[3*i]
            circles[i, 1] = result.x[3*i+1]
            circles[i, 2] = result.x[3*i+2]

        # Stage 2: Refinement with more stringent constraints
        result2 = minimize(
            _objective_function,
            result.x,
            args=(circles, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-4}
        )

        # Update circles with refined results
        for i in range(n):
            circles[i, 0] = result2.x[3*i]
            circles[i, 1] = result2.x[3*i+1]
            circles[i, 2] = result2.x[3*i+2]

        # Stage 3: Additional local search to fine-tune
        circles = _refine_with_local_search(circles, n, max_iter=100)

        # Stage 4: Final optimization with even stricter tolerance
        result3 = minimize(
            _objective_function,
            result2.x,
            args=(circles, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
        )

        # Final update
        for i in range(n):
            circles[i, 0] = result3.x[3*i]
            circles[i, 1] = result3.x[3*i+1]
            circles[i, 2] = result3.x[3*i+2]

    except Exception as e:
        # Fallback to hexagonal grid if optimization fails
        print(f"Optimization failed with error: {e}")

    # Ensure final constraints are met
    for i in range(n):
        x, y, r = circles[i]
        # Clamp to valid ranges
        circles[i, 0] = np.clip(x, r, 1-r)
        circles[i, 1] = np.clip(y, r, 1-r)
        circles[i, 2] = np.clip(r, 0.001, min(1-x, 1-y, x, y))

    return circles


# EVOLVE-BLOCK-END