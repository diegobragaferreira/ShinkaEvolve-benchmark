# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_circles_hexagonal(n=32):
    """Initialize circle positions using hexagonal grid for better spatial distribution."""
    # Use hexagonal grid pattern
    sqrt3 = np.sqrt(3)
    
    # Determine grid dimensions to fit 32 circles
    rows = int(np.ceil(np.sqrt(n) * 1.2))
    cols = int(np.ceil(n / rows))
    
    # Create hexagonal grid points
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x
    
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for hexagonal pattern
            x_offset = (j + 0.5) if (i % 2 == 0) else (j + 1.0)
            x = x_offset * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Only include points within the unit square
            if 0 <= x <= 1 and 0 <= y <= 1:
                grid_points.append([x, y])
                
    # Take exactly n points
    if len(grid_points) >= n:
        points = np.array(grid_points[:n])
    else:
        points = np.array(grid_points)
        # Pad with center points if needed
        while len(points) < n:
            points = np.vstack([points, [0.5, 0.5]])
    
    # Assign initial radii based on distance to boundaries and neighbors
    initial_radii = []
    for i in range(n):
        x, y = points[i]
        
        # Distance to boundaries
        dist_to_boundaries = min(x, 1-x, y, 1-y)
        
        # Calculate average distance to neighbors (if any)
        distances = []
        for j in range(n):
            if i != j:
                x2, y2 = points[j]
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                distances.append(dist)
        
        avg_dist = np.mean(distances) if distances else 0.1
        
        # Base radius: bounded by boundary distances and influenced by neighbor density
        base_radius = min(0.1, dist_to_boundaries * 0.4)
        density_factor = 1.0 / (avg_dist + 0.01)
        scaled_radius = base_radius * (1.0 / (1.0 + density_factor * 0.3))
        final_radius = max(0.005, min(0.15, scaled_radius))
        
        initial_radii.append(final_radius)
    
    # Combine into circles array
    circles = np.column_stack([points, initial_radii])
    return circles

def evaluate_fitness(circles):
    """Evaluate fitness of circle configuration with proper penalty system."""
    n = len(circles)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for overlaps and containment violations
    penalty = 0
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 10000  # Large penalty for containment violation
    
    # Check overlap constraints using vectorized operations for efficiency
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Compute pairwise sums of radii
        radii_matrix = np.tile(radii, (n, 1))
        radius_sums = radii_matrix + radii_matrix.T
        
        # Identify overlapping pairs
        overlap_mask = (distances < radius_sums) & (distances > 0)  # Exclude self-comparisons
        
        # Sum up overlap violations
        overlap_violations = np.sum(radius_sums[overlap_mask] - distances[overlap_mask])
        penalty += 1000 * overlap_violations
    
    return total_radius - penalty

def optimize_circles(circles):
    """Optimize circle positions using scipy optimization with proper constraints."""
    def objective(params):
        # Reshape params back to circles array
        circles_flat = params.reshape(-1, 3)
        return -evaluate_fitness(circles_flat)  # Negative because we maximize
    
    # Flatten the circles array for optimization
    initial_params = circles.flatten()
    
    # Define bounds for optimization
    bounds = []
    for i in range(len(initial_params)):
        if i % 3 == 2:  # radius parameter
            bounds.append((0.001, 0.45))  # Radius between 0.001 and 0.45
        else:  # x and y parameters
            bounds.append((0.001, 0.999))  # Position within the unit square
    
    try:
        # Use L-BFGS-B which handles bounds well
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 500, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Return original if optimization failed
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Initialize with hexagonal grid approach
    circles = initialize_circles_hexagonal(n)
    
    # Optimize the configuration
    circles = optimize_circles(circles)
    
    return circles

# EVOLVE-BLOCK-END