# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
from scipy.optimize import minimize
import time

def compute_distance_matrix(points):
    """Compute pairwise distance matrix for given points."""
    return squareform(pdist(points))

def calculate_min_max_ratio(distance_matrix):
    """Calculate the ratio of minimum to maximum distances."""
    # Exclude diagonal (distance to self)
    off_diagonal = distance_matrix[distance_matrix > 0]
    if len(off_diagonal) == 0:
        return 0.0
    d_min = np.min(off_diagonal)
    d_max = np.max(off_diagonal)
    return d_min / d_max if d_max > 0 else 0.0

def generate_hexagonal_lattice(n_points=16):
    """Generate precise hexagonal lattice with optimal spacing."""
    # Create a hexagonal lattice that naturally maximizes minimum distances
    sqrt3 = math.sqrt(3)
    row_spacing = sqrt3 / 2
    col_spacing = 1.0
    
    # Determine grid dimensions to accommodate n_points
    rows = int(math.ceil(math.sqrt(n_points)))
    cols = int(math.ceil(n_points / rows))
    
    # Ensure we have enough points
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n_points:
                break
            # Offset odd rows for proper hexagonal packing
            x = j * col_spacing + (i % 2) * col_spacing / 2
            y = i * row_spacing
            points.append([x, y])
        if len(points) >= n_points:
            break
    
    # Convert to numpy array
    points = np.array(points[:n_points])
    
    # Normalize to fit perfectly in [0,1] with proper scaling and centering
    if len(points) > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # Avoid division by zero
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
        # Scale to proper size and center in unit square
        scale_factor = 0.9
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        points[:, 0] = 0.05 + scale_factor * (points[:, 0] - center_x) + 0.5
        points[:, 1] = 0.05 + scale_factor * (points[:, 1] - center_y) + 0.5
    
    return points

def initialize_geometric_structure():
    """Initialize points using geometric principles for better distribution."""
    # Start with a hexagonal lattice and enhance with geometric constraints
    points = generate_hexagonal_lattice(16)
    
    # Apply geometric-based symmetry breaking
    np.random.seed(42)
    
    # Apply controlled perturbations that respect geometric relationships
    for i in range(len(points)):
        # Use geometrically meaningful perturbation pattern
        angle = i * 0.314159  # ~pi/10 increments for good distribution
        magnitude = 0.005 + 0.003 * math.sin(angle)
        
        # Apply perturbation in a way that enhances distribution
        noise_x = np.random.normal(0, magnitude, 1)[0]
        noise_y = np.random.normal(0, magnitude, 1)[0]
        
        points[i] += [noise_x, noise_y]
    
    # Ensure all points stay within bounds
    points = np.clip(points, 0, 1)
    
    return points

def geometric_local_optimization(points, max_iter=100):
    """Apply geometric-based local optimization that respects point relationships."""
    current_points = points.copy()
    
    for iteration in range(max_iter):
        # Calculate current distances
        try:
            dist_matrix = compute_distance_matrix(current_points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            if ratio < 1e-10:
                break
                
        except Exception:
            break
        
        # For each point, compute optimal adjustment direction
        new_points = current_points.copy()
        updated = False
        
        for i in range(len(current_points)):
            # Store original point
            original_point = current_points[i].copy()
            
            # Calculate gradients based on distances to neighbors
            best_direction = None
            best_improvement = 0
            
            # Try several directions for small perturbations
            directions = [
                [0.001, 0], [0, 0.001], [-0.001, 0], [0, -0.001],
                [0.000707, 0.000707], [-0.000707, 0.000707],
                [0.000707, -0.000707], [-0.000707, -0.000707]
            ]
            
            for dx, dy in directions:
                test_point = original_point + [dx, dy]
                test_point = np.clip(test_point, 0, 1)
                
                # Create test configuration
                test_points = current_points.copy()
                test_points[i] = test_point
                
                try:
                    test_dist_matrix = compute_distance_matrix(test_points)
                    test_ratio = calculate_min_max_ratio(test_dist_matrix)
                    
                    if test_ratio > ratio + best_improvement:
                        best_improvement = test_ratio - ratio
                        best_direction = [dx, dy]
                        
                except Exception:
                    continue
            
            # Apply best direction if beneficial
            if best_direction is not None and best_improvement > 1e-12:
                new_points[i] = original_point + best_direction
                updated = True
        
        # Update points if any improvements were made
        if updated:
            current_points = new_points.copy()
        else:
            break
    
    return current_points

def advanced_boundary_handling(points):
    """Improve point distribution near boundaries using geometric constraints."""
    # Check for boundary violations and handle them appropriately
    boundary_buffer = 0.01
    handled_points = points.copy()
    
    for i in range(len(handled_points)):
        point = handled_points[i]
        
        # Apply boundary correction if needed
        if point[0] < boundary_buffer:
            handled_points[i][0] = boundary_buffer + np.random.uniform(0, 0.005)
        elif point[0] > 1 - boundary_buffer:
            handled_points[i][0] = 1 - boundary_buffer - np.random.uniform(0, 0.005)
            
        if point[1] < boundary_buffer:
            handled_points[i][1] = boundary_buffer + np.random.uniform(0, 0.005)
        elif point[1] > 1 - boundary_buffer:
            handled_points[i][1] = 1 - boundary_buffer - np.random.uniform(0, 0.005)
    
    return handled_points

def multi_scale_geometric_optimization(initial_points, max_iter=500):
    """Perform multi-scale geometric optimization with different refinement stages."""
    current_points = initial_points.copy()
    
    # Stage 1: Coarse global optimization
    try:
        # Apply basic geometric optimization
        current_points = geometric_local_optimization(current_points, max_iter // 4)
        current_points = advanced_boundary_handling(current_points)
    except Exception:
        pass
    
    # Stage 2: Fine-grained optimization
    try:
        # Apply more detailed geometric optimization
        current_points = geometric_local_optimization(current_points, max_iter // 2)
        current_points = advanced_boundary_handling(current_points)
    except Exception:
        pass
    
    # Stage 3: Final refinement with boundary awareness
    try:
        # Apply final boundary-aware optimization
        current_points = geometric_local_optimization(current_points, max_iter // 4)
        current_points = advanced_boundary_handling(current_points)
    except Exception:
        pass
    
    return current_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize with geometrically sound configuration
    initial_points = initialize_geometric_structure()
    
    # Apply multi-scale geometric optimization to refine the solution
    optimized_points = multi_scale_geometric_optimization(initial_points, max_iter=500)
    
    # Final validation and boundary correction
    try:
        dist_matrix = compute_distance_matrix(optimized_points)
        ratio = calculate_min_max_ratio(dist_matrix)
        
        # If ratio is extremely low, fallback to a better configuration
        if ratio < 1e-10:
            initial_points = generate_hexagonal_lattice(16)
            optimized_points = multi_scale_geometric_optimization(initial_points, max_iter=500)
            
    except Exception:
        # Fallback to simple hexagonal lattice if anything goes wrong
        optimized_points = generate_hexagonal_lattice(16)
    
    # Ensure final points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

# EVOLVE-BLOCK-END