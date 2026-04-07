# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_circles_hexagonal(n=32):
    """Initialize circle positions using hexagonal grid pattern."""
    # Create hexagonal grid points for 32 circles
    rows = 6
    cols = 6  # 6x6 = 36 positions, more than enough for 32 circles

    # Create hexagonal grid points
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

    # Take first n points and add some randomness
    points = np.array(grid_points[:n])
    
    # Add slight jitter to avoid perfect grid artifacts
    jitter_strength = 0.1
    points[:, 0] += np.random.uniform(-jitter_strength * spacing_x, jitter_strength * spacing_x, n)
    points[:, 1] += np.random.uniform(-jitter_strength * spacing_y, jitter_strength * spacing_y, n)
    
    # Clamp to valid range
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    # Initial radius estimation based on density and boundary awareness
    initial_radii = []
    for i in range(n):
        x, y = points[i]
        
        # Calculate distance to boundaries
        dist_to_boundaries = min(x, 1-x, y, 1-y)
        
        # Estimate neighborhood influence
        distances = []
        for j in range(n):
            if i != j:
                x2, y2 = points[j]
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                distances.append(dist)
                
        avg_dist = np.mean(distances) if distances else 0.1
        
        # Density-based radius adjustment
        density_factor = 1.0 / (avg_dist + 0.01)
        boundary_factor = dist_to_boundaries
        
        # Base radius based on boundary proximity and density
        base_radius = min(0.1, boundary_factor * 0.3)
        scaled_radius = base_radius * (1.0 / (1.0 + density_factor * 0.3))
        
        final_radius = max(0.005, min(0.15, scaled_radius))
        initial_radii.append(final_radius)

    # Combine into circles array
    circles = np.column_stack([points, initial_radii])
    return circles

def evaluate_fitness(circles):
    """Evaluate fitness of circle configuration with proper penalties."""
    n = len(circles)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for constraint violations
    penalty = 0
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000  # Large penalty for containment violation
            
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                overlap_amount = (r1 + r2 - distance)
                penalty += 1000 * overlap_amount
                
    return total_radius - penalty

def optimize_circles(circles, max_iter=1000):
    """Optimize circle positions and radii using scipy optimization."""
    def objective(params):
        # Reshape params back to circles array
        circles_flat = params.reshape(-1, 3)
        return -evaluate_fitness(circles_flat)  # Negative because we maximize

    # Flatten the circles array for optimization
    initial_params = circles.flatten()
    
    # Define bounds for optimization (radius: 0.001-0.5, positions: 0.001-0.999)
    bounds = []
    for i in range(len(initial_params)):
        if i % 3 == 2:  # radius parameter
            bounds.append((0.001, 0.5))  # Radius between 0.001 and 0.5
        else:  # x and y parameters
            bounds.append((0.001, 0.999))  # Position within the unit square

    try:
        # Use L-BFGS-B which handles bounds well
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': max_iter})

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