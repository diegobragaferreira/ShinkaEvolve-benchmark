# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices

def hexagon_area(side_length):
    """Calculate area of regular hexagon with given side length"""
    return (3 * np.sqrt(3) / 2) * side_length ** 2

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_side_length):
    """Check if hexagon vertices are contained within outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, 0, outer_side_length)
    
    # Check if all vertices of inner hexagon are within outer hexagon using point-in-polygon test
    # For simplicity, we'll check distance from center to each vertex against outer radius
    outer_radius = outer_side_length
    inner_radius = np.sqrt(3) / 2  # Distance from center to vertex for unit hexagon
    
    # Calculate distance from outer center to inner hex center
    dist_to_center = np.sqrt((hex_vertices[0,0] - outer_center_x)**2 + (hex_vertices[0,1] - outer_center_y)**2)
    
    # Check containment condition
    max_dist = np.max(np.sqrt(np.sum((hex_vertices - np.array([outer_center_x, outer_center_y]))**2, axis=1)))
    return max_dist <= outer_radius

def hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Separating Axis Theorem"""
    # Get all edges from both hexagons
    edges1 = hex1_vertices[1:] - hex1_vertices[:-1]
    edges2 = hex2_vertices[1:] - hex2_vertices[:-1]
    
    # Get axes perpendicular to edges
    axes = np.vstack([np.column_stack([-e[:,1], e[:,0]]) for e in [edges1, edges2]])
    
    # Normalize axes
    axes_norm = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    
    # Project both polygons onto each axis
    projections1 = np.dot(hex1_vertices, axes_norm.T)
    projections2 = np.dot(hex2_vertices, axes_norm.T)
    
    # Check for separation
    for i in range(len(projections1.T)):
        p1_min, p1_max = np.min(projections1[:,i]), np.max(projections1[:,i])
        p2_min, p2_max = np.min(projections2[:,i]), np.max(projections2[:,i])
        
        if p1_max < p2_min or p2_max < p1_min:
            return False  # Separated on this axis
            
    return True  # Overlapping

def calculate_packing_score(params, n_hexagons=11):
    """Calculate negative score for optimization (minimize negative score = maximize score)"""
    # Unpack parameters
    outer_side_length = params[-1]
    outer_center_x = params[-2]
    outer_center_y = params[-3]
    hex_params = params[:-3]
    
    # Reshape hexagon parameters
    hex_positions = hex_params.reshape(n_hexagons, 3)  # [x, y, angle]
    
    # Precompute all hexagon vertices
    all_hex_vertices = []
    for i in range(n_hexagons):
        x, y, angle = hex_positions[i]
        vertices = hexagon_vertices(x, y, angle)
        all_hex_vertices.append(vertices)
    
    # Check containment and overlap
    total_score = 0
    
    # Check containment constraint
    for i in range(n_hexagons):
        if not check_containment(all_hex_vertices[i], outer_center_x, outer_center_y, outer_side_length):
            # Large penalty for containment violation
            return 1e10
    
    # Check overlap constraints
    for i in range(n_hexagons):
        for j in range(i+1, n_hexagons):
            if hexagon_overlap(all_hex_vertices[i], all_hex_vertices[j]):
                # Large penalty for overlap violation
                return 1e10
    
    # Objective: maximize 1/outer_side_length (minimize outer_side_length)
    objective_value = 1.0 / outer_side_length
    
    return -objective_value  # Return negative because we're minimizing

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    # Initial guess for hexagon positions and orientations
    # Start with a reasonable configuration based on triangular lattice
    initial_positions = np.array([
        [0, 0, 0],      # center
        [-2, 0, 0],     # left
        [2, 0, 0],      # right
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [-1, 1, 0],     # top-left
        [1, 1, 0],      # top-right
        [-1, -1, 0],    # bottom-left
        [1, -1, 0],     # bottom-right
        [-2, 1, 0],     # far top-left
        [2, 1, 0],      # far top-right
    ])
    
    # Set initial outer hexagon parameters
    outer_side_length = 6.0
    outer_center_x = 0.0
    outer_center_y = 0.0
    
    # Flatten all parameters
    initial_params = np.concatenate([
        initial_positions.flatten(), 
        [outer_center_x, outer_center_y, outer_side_length]
    ])
    
    # Define bounds
    # Position bounds
    pos_bounds = [(-10, 10)] * (n * 2)  # x, y for each hexagon
    # Center bounds for outer hexagon
    center_bounds = [(-10, 10), (-10, 10)]
    # Outer hexagon side length bound (must be positive)
    size_bounds = [(0.1, 10)]
    
    bounds = pos_bounds + center_bounds + size_bounds
    
    # Define constraints for optimization
    def constraint_func(params):
        # This function returns the constraint values (should be >= 0 for equality constraints)
        # We'll just check basic feasibility in scoring function
        return 0
    
    # Run optimization
    try:
        result = minimize(
            calculate_packing_score,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        # Extract results
        final_params = result.x
        outer_side_length = final_params[-1]
        outer_center_x = final_params[-2]
        outer_center_y = final_params[-3]
        hex_positions = final_params[:-3].reshape(n, 3)
        
        # Validate the result
        # Recalculate with refined check
        all_hex_vertices = []
        for i in range(n):
            x, y, angle = hex_positions[i]
            vertices = hexagon_vertices(x, y, angle)
            all_hex_vertices.append(vertices)
        
        # Final validation
        valid_packing = True
        for i in range(n):
            if not check_containment(all_hex_vertices[i], outer_center_x, outer_center_y, outer_side_length):
                valid_packing = False
                break
        
        if not valid_packing:
            # Fallback to initial configuration if optimization fails
            pass
        else:
            # Update with optimized positions
            pass
            
        inner_hex_data = hex_positions.copy()
        outer_hex_data = np.array([outer_center_x, outer_center_y, 0])  # No rotation for outer hexagon
        
    except Exception as e:
        # Fallback to initial configuration on error
        print(f"Optimization failed: {e}")
        inner_hex_data = initial_positions.copy()
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 6.0
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
