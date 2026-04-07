# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random

def _validate_circle_placement(circles: np.ndarray) -> bool:
    """Validate that circles are within bounds and don't overlap."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # Find all pairs within distance 2*r (minimum separation needed to avoid overlap)
    pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

    for i, j in pairs:
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2
        min_distance_sq = (r1 + r2)**2
        if distance_sq < min_distance_sq:
            return False

    return True

def _generate_voronoi_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a Voronoi-inspired spreading mechanism."""
    np.random.seed(seed)

    # Create a grid of candidate positions
    grid_size = max(3, int(np.ceil(np.sqrt(n_circles))))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)

    # Generate all grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])

    # If we have more circles than grid points, add some random points
    if len(grid_points) < n_circles:
        extra_points = n_circles - len(grid_points)
        for _ in range(extra_points):
            grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

    # Shuffle the points to avoid systematic bias
    random.shuffle(grid_points)

    # Take the first n_circles points
    points = np.array(grid_points[:n_circles])

    # Initialize circles with small radii
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = points[:, 0]  # x coordinates
    circles[:, 1] = points[:, 1]  # y coordinates
    circles[:, 2] = 0.01         # initial small radii

    return circles

def _greedy_fallback(n_circles: int) -> np.ndarray:
    """Fallback method to generate a feasible configuration."""
    # Simple greedy approach: place circles in order of decreasing radius
    circles = np.zeros((n_circles, 3))

    # Start with small radii and gradually increase
    # Place in a way that they don't overlap initially
    positions = []
    radii = []

    # Try to place circles greedily by spacing them out
    placed = 0
    radius = 0.05
    while placed < n_circles and radius > 0.005:
        # Try placing circles in a spiral pattern or grid
        attempt = 0
        while attempt < 100 and placed < n_circles:
            # Place in grid-like fashion
            rows = int(np.sqrt(n_circles)) + 1
            cols = n_circles // rows + 1

            for i in range(rows):
                for j in range(cols):
                    if placed >= n_circles:
                        break
                    x = 0.1 + j * 0.8 / cols
                    y = 0.1 + i * 0.8 / rows

                    # Check if this position is valid
                    valid = True
                    for pos, rad in zip(positions, radii):
                        dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                        if dist_sq < (rad + radius)**2:
                            valid = False
                            break

                    if valid:
                        positions.append([x, y])
                        radii.append(radius)
                        placed += 1
            attempt += 1

        radius *= 0.9  # Decrease radius slightly

    # Fill remaining circles
    while placed < n_circles:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
        radii.append(0.01)
        placed += 1

    circles[:, 0] = [pos[0] for pos in positions]
    circles[:, 1] = [pos[1] for pos in positions]
    circles[:, 2] = radii

    return circles

def _quadratic_programming_approach(n_circles: int = 26, max_iter: int = 1000) -> np.ndarray:
    """
    Solve the circle packing problem using Quadratic Programming approach.
    Formulates the problem as: maximize sum(r_i) subject to:
    - containment constraints: 0 <= x_i <= 1, 0 <= y_i <= 1, 0 <= r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    - non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    """
    
    # Initial guess using Voronoi-like initialization
    circles = _generate_voronoi_initialization(n_circles, seed=42)
    
    # Flatten initial configuration for optimization
    x0 = circles.flatten()
    
    def objective(x_flat):
        """Objective function to maximize (negative because minimize)"""
        circles = x_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize sum of radii

    def constraint_func(x_flat):
        """Constraint function: returns positive values when violated"""
        circles = x_flat.reshape(-1, 3)
        constraints = []
        
        # Non-overlap constraints (penalty method)
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance squared
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                
                # Required minimum distance squared
                min_dist_sq = (r1 + r2)**2
                
                # Penalty for overlap: positive when violated
                penalty = max(0, min_dist_sq - dist_sq)
                constraints.append(penalty)
        
        # Boundary constraints (penalty for going out of bounds)
        for i in range(n_circles):
            x, y, r = circles[i]
            
            # Penalties for boundary violations
            penalties = [
                max(0, r - x),           # Left boundary
                max(0, r - (1 - x)),     # Right boundary
                max(0, r - y),           # Bottom boundary
                max(0, r - (1 - y))      # Top boundary
            ]
            constraints.extend(penalties)
            
        return np.array(constraints)

    def constraint_func_smooth(x_flat):
        """Smooth approximation of constraints for QP"""
        circles = x_flat.reshape(-1, 3)
        penalty = 0
        
        # Non-overlap constraints with penalty terms
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # Smooth penalty: soft violation penalty
                if dist_sq < min_dist_sq:
                    violation = min_dist_sq - dist_sq
                    penalty += violation**2  # Quadratic penalty
        
        # Boundary constraints
        for i in range(n_circles):
            x, y, r = circles[i]
            # Soft constraint penalties
            penalty += max(0, r - x)**2 + max(0, r - (1 - x))**2 + \
                       max(0, r - y)**2 + max(0, r - (1 - y))**2
            
        return penalty

    # Bounds for variables [x1, y1, r1, x2, y2, r2, ...]
    bounds = []
    for i in range(n_circles):
        # x bounds
        bounds.append((0.05, 0.95))
        # y bounds
        bounds.append((0.05, 0.95))
        # r bounds (considering containment)
        bounds.append((0.005, 0.45))

    # Optimization using sequential quadratic programming
    try:
        options = {'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
        
        # First pass: try to solve with a simpler approach
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options=options, tol=1e-6)
        
        if res.success:
            result_circles = res.x.reshape(-1, 3)
            
            # Refine with improved penalty method
            def refined_objective(x_flat):
                circles = x_flat.reshape(-1, 3)
                # Primary objective: maximize sum of radii
                obj = -np.sum(circles[:, 2])
                
                # Secondary objective: penalize constraint violations
                penalty = 0
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        min_dist_sq = (r1 + r2)**2
                        
                        if dist_sq < min_dist_sq:
                            violation = min_dist_sq - dist_sq
                            penalty += violation**2 * 1000  # Heavy penalty for overlaps
                    
                    # Boundary penalties
                    x, y, r = circles[i]
                    penalty += (max(0, r - x)**2 + 
                               max(0, r - (1 - x))**2 +
                               max(0, r - y)**2 + 
                               max(0, r - (1 - y))**2) * 100
                
                return obj + penalty
            
            # Final optimization with refined objective
            res_final = minimize(refined_objective, res.x, method='L-BFGS-B', 
                               bounds=bounds, options=options, tol=1e-6)
            
            if res_final.success:
                circles = res_final.x.reshape(-1, 3)
            else:
                circles = result_circles
        else:
            # Fall back to initial configuration for safety
            circles = x0.reshape(-1, 3)
            
    except Exception as e:
        # If optimization fails, return the initial Voronoi configuration
        circles = x0.reshape(-1, 3)

    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Use quadratic programming approach
    circles = _quadratic_programming_approach(n_circles=26, max_iter=1000)
    
    # Validate solution
    if not _validate_circle_placement(circles):
        # If invalid, use fallback
        circles = _greedy_fallback(26)
            
    # Ensure final validation
    if not _validate_circle_placement(circles):
        circles = _generate_voronoi_initialization(26)
    
    return circles

# EVOLVE-BLOCK-END