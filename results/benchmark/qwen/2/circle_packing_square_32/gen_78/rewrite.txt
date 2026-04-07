# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def _compute_voronoi_constraints(circles):
    """Compute Voronoi-based constraints for circle placement."""
    n = len(circles)
    if n == 0:
        return []
    
    # Get circle centers
    points = circles[:, :2]
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
    except:
        # Fallback to simple approach if Voronoi fails
        return [(i, i+1) for i in range(n-1)]
    
    # Extract Voronoi edges
    constraints = []
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:  # Skip infinite edges
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    # Add constraint between vertices
                    pass
    
    # Simplified: return all pairs for overlap checking
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _calculate_voronoi_radius(circles, idx, boundary_distance):
    """Calculate optimal radius based on Voronoi geometry and boundary constraints."""
    n = len(circles)
    if n == 0:
        return boundary_distance * 0.4
    
    # Get current circle
    x, y, current_r = circles[idx]
    
    # Find nearest neighbors
    distances = []
    for i in range(n):
        if i != idx:
            x2, y2, r2 = circles[i]
            dist = np.sqrt((x-x2)**2 + (y-y2)**2)
            distances.append((dist, r2, i))
    
    # Sort by distance
    distances.sort()
    
    # Calculate minimum safe radius considering neighbors
    min_radius = boundary_distance
    
    if distances:
        # Consider the closest neighbor
        closest_dist, closest_r, closest_idx = distances[0]
        # Radius must be such that circle doesn't overlap
        # Distance between centers >= sum of radii
        max_radius_from_neighbor = closest_dist - closest_r
        if max_radius_from_neighbor > 0:
            min_radius = min(min_radius, max_radius_from_neighbor * 0.9)
    
    # Ensure reasonable bounds
    return max(0.005, min(0.15, min_radius))

def _evaluate_voronoi_fitness(circles):
    """Evaluate fitness using Voronoi-based constraints."""
    n = len(circles)
    if n == 0:
        return 0.0
    
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for containment violations
    penalty = 0
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 10000
    
    # Check overlap constraints using Voronoi approach
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                penalty += 10000 * (r1 + r2 - distance)
    
    return total_radius - penalty

def _optimize_single_circle(circles, idx, max_iter=50):
    """Optimize a single circle's position and radius using constrained optimization."""
    n = len(circles)
    
    # Get current state
    x, y, r = circles[idx]
    
    # Define objective for this circle
    def objective(params):
        new_x, new_y, new_r = params
        
        # Check boundary constraints
        if new_x - new_r < 0 or new_x + new_r > 1 or new_y - new_r < 0 or new_y + new_r > 1:
            return 1000000  # Large penalty for boundary violation
            
        # Calculate overlap penalties with all others
        penalty = 0
        for i in range(n):
            if i != idx:
                x2, y2, r2 = circles[i]
                dist = np.sqrt((new_x-x2)**2 + (new_y-y2)**2)
                if dist < new_r + r2:
                    penalty += 10000 * (new_r + r2 - dist)
        
        # Return negative of total radius (since we maximize)
        return -new_r - penalty
    
    # Initial guess with small perturbation
    initial_guess = [x + np.random.normal(0, 0.01), 
                     y + np.random.normal(0, 0.01), 
                     r]
    
    # Bounds
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)]
    
    # Optimization
    try:
        result = minimize(objective, initial_guess, 
                         method='L-BFGS-B', 
                         bounds=bounds,
                         options={'maxiter': max_iter})
        
        if result.success:
            return result.x
    except:
        pass
    
    # Return original if optimization fails
    return [x, y, r]

def _initialize_with_voronoi(n=32):
    """Initialize circles using Voronoi-inspired approach."""
    # Create hexagonal grid with some randomness
    rows = 6  # 6x6 = 36 positions
    cols = 6
    
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x
    
    # Generate base grid positions
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            if x <= 1.0 and y <= 1.0:
                grid_points.append([x, y])
    
    # Take first n points
    points = np.array(grid_points[:n])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = points[i][0]
        circles[i][1] = points[i][1]
        circles[i][2] = 0.05  # Initial small radius
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi-guided evolutionary optimization approach.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with Voronoi-inspired approach
    circles = _initialize_with_voronoi(n)
    
    # Multi-scale optimization
    for iteration in range(3):  # Three refinement levels
        # First, optimize each circle individually with boundary constraints
        for i in range(n):
            new_params = _optimize_single_circle(circles, i, max_iter=20)
            circles[i] = new_params
        
        # Update radii based on Voronoi geometry
        for i in range(n):
            # Calculate boundary distance
            x, y = circles[i][0], circles[i][1]
            boundary_distance = min(x, 1-x, y, 1-y)
            
            # Recalculate radius considering neighbors and boundaries
            new_radius = _calculate_voronoi_radius(circles, i, boundary_distance)
            circles[i][2] = new_radius
    
    # Final comprehensive optimization
    for _ in range(10):
        # Randomly select circles to re-optimize
        selected_indices = np.random.choice(n, size=min(10, n//2), replace=False)
        for i in selected_indices:
            new_params = _optimize_single_circle(circles, i, max_iter=30)
            circles[i] = new_params
    
    # Final boundary adjustment
    for i in range(n):
        x, y, r = circles[i]
        boundary_distance = min(x, 1-x, y, 1-y)
        if r > boundary_distance:
            circles[i][2] = boundary_distance
    
    return circles


# EVOLVE-BLOCK-END