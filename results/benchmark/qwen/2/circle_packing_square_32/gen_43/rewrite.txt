# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_hexagonal_grid(n_circles):
    """Initialize circle positions using a denser hexagonal grid pattern."""
    # Calculate optimal grid dimensions
    sqrt_n = math.ceil(math.sqrt(n_circles))
    rows = sqrt_n
    cols = math.ceil(n_circles / rows)
    
    # Create a hexagonal grid with proper spacing
    positions = []
    spacing = 0.15  # Initial spacing guess
    
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for hexagonal packing
            x_offset = j * spacing
            y_offset = i * spacing * math.sqrt(3) / 2
            
            # Add offset for even/odd rows
            if i % 2 == 1:
                x_offset += spacing / 2
                
            positions.append([x_offset, y_offset])
            
    # Adjust to fit within unit square
    positions = np.array(positions[:n_circles])
    
    if len(positions) > 0:
        # Normalize to fit in [0.05, 0.95] x [0.05, 0.95] to ensure boundary room
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        # Avoid division by zero
        scale_x = 0.9 / (max_x - min_x) if (max_x - min_x) > 0 else 1
        scale_y = 0.9 / (max_y - min_y) if (max_y - min_y) > 0 else 1
        
        # Apply scaling and translation to center in unit square with padding
        positions[:, 0] = (positions[:, 0] - min_x) * scale_x * 0.9 + 0.05
        positions[:, 1] = (positions[:, 1] - min_y) * scale_y * 0.9 + 0.05
    
    return positions

def compute_radius_at_position(pos, circles, min_radius=0.001):
    """Compute maximum possible radius at given position without overlapping existing circles."""
    x, y = pos
    
    # Initial estimate of radius based on distance to nearest boundary
    r_boundary = min(x, 1-x, y, 1-y)
    
    if r_boundary <= min_radius:
        return min_radius
    
    # Check overlap with existing circles using KDTree for efficiency
    if len(circles) > 0:
        tree = cKDTree(circles[:, :2])  # Only use x,y coordinates for tree
        distances, _ = tree.query([pos], k=min(5, len(circles)))  # Query nearby circles
        
        min_dist = float('inf')
        for dist in distances[0]:
            if dist > 0:  # Skip self-distance
                min_dist = min(min_dist, dist)
        
        # Maximum radius is limited by both boundary and existing circles
        if min_dist != float('inf'):
            r = min(r_boundary, min_dist/2 - 0.001)
            return max(min_radius, r)
    
    return max(min_radius, r_boundary)

def smooth_penalty(distance, threshold=0.0):
    """Smooth exponential penalty function."""
    # Exponential penalty that becomes very steep as constraints are violated
    if distance <= threshold:
        return 1000 * math.exp(10 * (distance - threshold))
    return 0

def evaluate_fitness(circles_flat, n_circles):
    """Evaluate the fitness of a circle configuration using smooth penalties."""
    # Reshape flat array back into circles
    circles = circles_flat.reshape((n_circles, 3))
    
    # Calculate sum of radii (this is what we want to maximize)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty terms for constraint violations
    penalty = 0
    
    # Boundary penalties using smooth exponential function
    for i in range(n_circles):
        x, y, r = circles[i]
        # Check boundaries with smooth penalties
        left_pen = smooth_penalty(x - r, 0)
        right_pen = smooth_penalty(1 - (x + r), 0)
        bottom_pen = smooth_penalty(y - r, 0)
        top_pen = smooth_penalty(1 - (y + r), 0)
        
        penalty += left_pen + right_pen + bottom_pen + top_pen
    
    # Overlap penalties using smooth exponential function
    # Use KDTree for efficient overlap checking
    if len(circles) > 1:
        tree = cKDTree(circles[:, :2])
        
        # Query pairs within a reasonable radius
        pairs = tree.query_pairs(r=0.01, output_type='ndarray')  # Adjust radius as needed
        
        for i, j in pairs:
            if i < j:  # Avoid duplicate pairs
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # Calculate overlap penalty
                overlap = max(0, r1 + r2 - distance)
                if overlap > 0:
                    # Use smooth exponential penalty
                    penalty += 1000 * math.exp(10 * overlap)
    
    return -(total_radius + penalty)  # Negative because we minimize in scipy

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Initialize using hexagonal grid
    initial_positions = initialize_hexagonal_grid(n)
    
    # Set initial radii based on positions and constraints
    initial_radii = np.array([compute_radius_at_position(pos, initial_positions[:i]) 
                             for i, pos in enumerate(initial_positions)])
    
    # Combine positions and radii into a single flat array for optimization
    initial_solution = np.column_stack([initial_positions, initial_radii]).flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # x, y, r bounds
    
    # Optimize using scipy minimize
    try:
        result = minimize(
            evaluate_fitness, 
            initial_solution, 
            args=(n,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=None
        )
        
        if result.success:
            # Extract final solution
            final_circles = result.x.reshape((n, 3))
            return final_circles
        else:
            # If optimization fails, return the initial configuration
            print("Optimization failed:", result.message)
            return np.column_stack([initial_positions, initial_radii])
    except Exception as e:
        print(f"Optimization error: {e}")
        # Return initial configuration if anything goes wrong
        return np.column_stack([initial_positions, initial_radii])

# EVOLVE-BLOCK-END