# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math

def is_valid_placement(circles, x, y, r):
    """Check if placing a circle at (x,y) with radius r is valid."""
    # Check boundary constraints
    if r > x or r > y or r > (1-x) or r > (1-y):
        return False

    # Check overlap with existing circles
    for i in range(len(circles)):
        cx, cy, cr = circles[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < (r + cr):
            return False

    return True

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k-nearest neighbors."""
    if len(circles) < 2:
        return 0.0

    # Convert to numpy array for efficient processing
    pts = np.array(circles)[:, :2]
    tree = cKDTree(pts)

    # Query k nearest neighbors (excluding the point itself if it exists)
    distances, indices = tree.query(point, k=min(k+1, len(pts)), p=2)

    # Average distance to neighbors (excluding self if present)
    if len(distances) > 1:
        avg_distance = np.mean(distances[1:])  # Skip the first (distance to itself)
    else:
        avg_distance = distances[0]

    # Density is inversely proportional to average distance
    if avg_distance > 0:
        return 1.0 / avg_distance
    else:
        return float('inf')

def initialize_circles_heuristic(n=32):
    """Initialize circle positions using a heuristic approach with density awareness."""
    circles = []

    # Start with a coarse grid and then refine
    # Try to place circles in a hexagonal-like pattern
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)

    # Create initial placements with decreasing radii
    for i in range(grid_size):
        for j in range(grid_size):
            if len(circles) >= n:
                break
            x = (i + 1) * spacing
            y = (j + 1) * spacing

            # Initial radius estimate based on available space
            r_min = min(x, y, 1-x, 1-y)
            r = min(r_min * 0.3, 0.15)

            # Only add if valid
            if is_valid_placement(circles, x, y, r):
                circles.append([x, y, r])

    # Fill remaining spots with smaller circles using density-aware approach
    while len(circles) < n:
        best_r = 0
        best_x, best_y = 0, 0

        # Sample potential positions
        for _ in range(1000):  # Sample many points
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)

            # Estimate max radius at this location
            r_max = min(x, y, 1-x, 1-y)
            if r_max <= 0:
                continue

            # Compute local density at this point for adaptive sizing
            density = compute_local_density(circles, [x, y], k=5)
            radius_adjustment = 1.0 / (1.0 + 0.3 * density)
            r_adjusted = min(r_max * 0.4 * radius_adjustment, r_max * 0.4)

            # Try different radii
            test_radii = np.linspace(0.01, r_adjusted, 10)
            for r in test_radii:
                if is_valid_placement(circles, x, y, r):
                    if r > best_r:
                        best_r = r
                        best_x, best_y = x, y
                        break

        if best_r > 0:
            circles.append([best_x, best_y, best_r])

    return np.array(circles)

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
        
        # Find nearby circles using cKDTree for efficiency
        neighbors = tree.query_ball_point([x, y], 2 * r)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                overlap_violation = max(0, (r + r2) - distance)
                violations[i] += overlap_violation
    
    return violations

def optimize_individual_circle(circle_idx, circles):
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
    # Initialize using enhanced heuristic with density awareness
    circles = initialize_circles_heuristic(32)
    
    # Refine the configuration
    circles = refine_configuration(circles)
    
    # Final optimization pass
    circles = refine_configuration(circles, max_iterations=20)
    
    return circles

# EVOLVE-BLOCK-END